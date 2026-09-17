#!/usr/bin/env python3
"""Standalone XR-1 SF_control checkpoint/loss sanity diagnostics.

This script intentionally does not import or edit any formal training entrypoint
state.  It reuses the existing data adapter and model construction, writes only
to a new diagnostic output directory, and performs a 100-step local optimizer
smoke test (no Lightning/DDP launcher).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_xr1_baseline_official20 as common
import spatial_forcing_xr1 as sf


def clone_value(value):
    if isinstance(value, dict):
        return {k: clone_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_value(v) for v in value)
    if isinstance(value, torch.Tensor):
        return value.clone()
    return value


def move_value(value, device):
    if isinstance(value, dict):
        return {k: move_value(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move_value(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_value(v, device) for v in value)
    return value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def tensor_digest(x: torch.Tensor) -> str:
    # NumPy does not expose bfloat16 on all versions; float32 is sufficient
    # for a stable diagnostic fingerprint and preserves the exact values used
    # for comparing the two seeded forwards.
    y = x.detach().float().cpu().contiguous()
    return hashlib.sha256(y.numpy().tobytes()).hexdigest()


def tensor_summary(x: torch.Tensor):
    y = x.detach().float().cpu()
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "min": float(y.min()) if y.numel() else None,
        "max": float(y.max()) if y.numel() else None,
        "mean": float(y.mean()) if y.numel() else None,
        "std": float(y.std(unbiased=False)) if y.numel() else None,
        "sha256": tensor_digest(x),
    }


def summarize_tensors(value, prefix=""):
    out = {}
    if isinstance(value, dict):
        for k, v in value.items():
            out.update(summarize_tensors(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            out.update(summarize_tensors(v, f"{prefix}[{i}]"))
    elif isinstance(value, torch.Tensor):
        out[prefix] = tensor_summary(value)
    return out


def scalar(value):
    return float(value.detach().float().cpu()) if isinstance(value, torch.Tensor) else float(value)


def policy_result(result):
    return {
        "loss_policy": scalar(0.5 * result["loss_mse"] + result["loss_freq"]),
        "loss_mse": scalar(result["loss_mse"]),
        "loss_freq": scalar(result["loss_freq"]),
        "loss_official_raw": scalar(result.get("loss", 0.5 * result["loss_mse"] + result["loss_freq"])),
        "loss_choice_audit": scalar(result.get("loss_choice", 0.0)),
        "loss_score_audit": scalar(result.get("loss_score", 0.0)),
    }


class RNGProbe:
    def __init__(self, model):
        self.model = model
        self.beta_values = []
        self.noise_values = []
        self.orig_beta = None
        self.orig_randn = None

    def __enter__(self):
        self.orig_beta = self.model.beta.sample
        self.orig_randn = torch.randn_like

        def beta_sample(shape, *args, **kwargs):
            value = self.orig_beta(shape, *args, **kwargs)
            self.beta_values.append(value.detach().cpu().clone())
            return value

        def randn_like(input_tensor, *args, **kwargs):
            value = self.orig_randn(input_tensor, *args, **kwargs)
            self.noise_values.append({
                "shape": list(input_tensor.shape),
                "dtype": str(input_tensor.dtype),
                "device": str(input_tensor.device),
                "mean": float(value.float().mean()),
                "std": float(value.float().std(unbiased=False)),
                "sha256": tensor_digest(value),
            })
            return value

        self.model.beta.sample = beta_sample
        torch.randn_like = randn_like
        return self

    def __exit__(self, exc_type, exc, tb):
        self.model.beta.sample = self.orig_beta
        torch.randn_like = self.orig_randn

    def json(self):
        return {
            "beta_samples": [tensor_summary(x) for x in self.beta_values],
            "randn_like_calls": self.noise_values,
        }


def run_forward(runner, batch, seed, probe=False, grad=False):
    seed_all(seed)
    runner.model.train()
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        if probe:
            with RNGProbe(runner.model) as p:
                result = runner.model(batch, return_loss=True)
            return policy_result(result), p.json()
        result = runner.model(batch, return_loss=True)
        return policy_result(result), None


def checkpoint_audit(runner, args, out):
    ckpt = torch.load(str(args.model_ckpt), map_location="cpu", mmap=True, weights_only=False)
    source = ckpt["module"]
    target = runner.state_dict()
    missing = sorted(k for k in target if k not in source)
    unexpected = sorted(k for k in source if k not in target)
    mismatch = sorted(k for k in target if k in source and tuple(target[k].shape) != tuple(source[k].shape))
    matched = [k for k in target if k in source and tuple(target[k].shape) == tuple(source[k].shape)]

    def category(key):
        k = key.removeprefix("model.")
        if k.startswith("dit."):
            return "DiT_action_expert"
        if k.startswith("vlm.") or k.startswith("vlm_"):
            return "VLM_Qwen"
        if any(x in k for x in ("projector", "action_output", "action_head", "choice", "score")):
            return "projector_action_head"
        return "other"

    def counts(keys):
        result = {}
        for k in keys:
            c = category(k)
            result[c] = result.get(c, 0) + 1
        return result

    loaded_numel = sum(target[k].numel() for k in matched)
    total_numel = sum(v.numel() for v in target.values())
    report = {
        "requested_checkpoint": str(args.requested_checkpoint),
        "actual_policy_state_checkpoint_loaded_by_current_xr1": str(args.model_ckpt),
        "processor_model_directory": str(args.processor),
        "path_note": "RoboCasa365 is the local processor/model directory; current XR-1 code loads full policy weights from model_states.pt.",
        "source_state_keys": len(source),
        "target_state_keys": len(target),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatch_keys": mismatch,
        "loaded_parameter_count": int(loaded_numel),
        "total_parameter_count": int(total_numel),
        "loaded_fraction": loaded_numel / max(total_numel, 1),
        "matched_key_counts_by_component": counts(matched),
        "missing_key_counts_by_component": counts(missing),
        "unexpected_key_counts_by_component": counts(unexpected),
        "shape_mismatch_counts_by_component": counts(mismatch),
        "processor_files": sorted(p.name for p in args.processor.iterdir())[:100],
    }
    lines = [
        "XR-1 SF_control checkpoint loading audit",
        f"requested checkpoint: {args.requested_checkpoint}",
        f"actual full policy state loaded: {args.model_ckpt}",
        f"processor/model directory: {args.processor}",
        "NOTE: RoboCasa365 is a local HF-style processor/model directory; current XR-1 loads policy state from model_states.pt.",
        f"source keys: {len(source)}",
        f"target keys: {len(target)}",
        f"loaded parameter count: {loaded_numel}",
        f"total parameter count: {total_numel}",
        f"loaded fraction: {loaded_numel / max(total_numel, 1):.9f}",
        f"missing keys: {len(missing)} by component {counts(missing)}",
        f"unexpected keys: {len(unexpected)} by component {counts(unexpected)}",
        f"shape mismatches: {len(mismatch)} by component {counts(mismatch)}",
        "",
        "missing key list:",
        *missing,
        "",
        "unexpected key list:",
        *unexpected,
        "",
        "shape mismatch key list:",
        *mismatch,
    ]
    (out / "checkpoint_load_report.txt").write_text("\n".join(lines) + "\n")
    (out / "checkpoint_load_report.json").write_text(json.dumps(report, indent=2) + "\n")
    del ckpt, source, target
    return report


def inject_control_lora(model):
    model.requires_grad_(False)
    qwen = model.vlm.model.language_model
    qwen_targets = sf.discover_linear_targets(qwen, sf.QWEN_LORA_SUFFIXES)
    dit_targets = sf.discover_linear_targets(model.dit, sf.DIT_LORA_SUFFIXES)
    if len(qwen_targets) != sf.EXPECTED_QWEN_LORA_TARGETS or len(dit_targets) != sf.EXPECTED_DIT_LORA_TARGETS:
        raise RuntimeError(f"LoRA target mismatch qwen={len(qwen_targets)} dit={len(dit_targets)}")
    sf.inject_lora(qwen, qwen_targets)
    sf.inject_lora(model.dit, dit_targets)
    return qwen_targets, dit_targets


def lora_report(model, qwen_targets, dit_targets):
    values = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            values[name] = {
                "numel": int(param.numel()),
                "norm": float(param.detach().float().norm()),
                "requires_grad": bool(param.requires_grad),
            }
    a_norm = sum(v["norm"] for k, v in values.items() if "lora_A" in k)
    b_norm = sum(v["norm"] for k, v in values.items() if "lora_B" in k)
    return {
        "qwen_target_count": len(qwen_targets),
        "dit_target_count": len(dit_targets),
        "rank": sf.LORA_RANK,
        "alpha": sf.LORA_ALPHA,
        "scaling": sf.LORA_ALPHA / sf.LORA_RANK,
        "dropout": sf.LORA_DROPOUT,
        "lora_parameter_count": sum(v["numel"] for v in values.values()),
        "lora_A_norm_sum": a_norm,
        "lora_B_norm_sum": b_norm,
        "initial_delta_norm": 0.0 if b_norm == 0.0 else None,
        "parameters": values,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--model-ckpt", type=Path, required=True)
    ap.add_argument("--requested-checkpoint", type=Path, default=Path("/data-cfs/data3/models/Xiaomi-Robotics-1-RoboCasa365"))
    ap.add_argument("--processor", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--target-gpu-gb", type=float, default=67.0)
    ap.add_argument("--limit-gpu-gb", type=float, default=70.0)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest = args.manifest.resolve()
    args.model_ckpt = args.model_ckpt.resolve()
    args.processor = args.processor.resolve()
    args.dataset_root = args.dataset_root.resolve()
    common.DATA_ROOT = args.dataset_root
    common.MANIFEST = args.manifest
    common.META = args.manifest.with_name("manifest.meta.json")
    common.MODEL_CKPT = args.model_ckpt
    common.PROCESSOR = args.processor
    common.memory_policy(args.target_gpu_gb, args.limit_gpu_gb)
    seed_all(20260917)

    stats = common.build_action_stats_processor()
    processor = common.build_official_qwen_processor()
    rows = sf.load_six_rows(args.manifest)
    data = sf.SixTaskData(rows, processor, args.batch_size, args.workers, False, Path("/tmp/no-esm"), gpu_prefetch=False, stats_processor=stats)
    data.setup("fit")
    val_loader = DataLoader(data.val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=data.collate)
    train_loader = DataLoader(data.train_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=data.collate)
    val_batch_cpu = next(iter(val_loader))
    train_iter = iter(train_loader)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args_ns = SimpleNamespace(max_steps=20000, target_gpu_gb=args.target_gpu_gb, limit_gpu_gb=args.limit_gpu_gb)
    bare = common.Official20Runner(common.MODEL_CKPT, 20000, args.target_gpu_gb, args.limit_gpu_gb)
    bare.configure_model()
    bare.model.to(device)
    bare.model.train()
    report = checkpoint_audit(bare, args, args.output_dir)

    # Original checkpoint forward and fixed step-0 evaluation, before adapters.
    base_batch = move_value(clone_value(val_batch_cpu), device)
    base_input_summary = summarize_tensors(base_batch)
    base_eval, base_probe = run_forward(bare, clone_value(base_batch), sf.VAL_SEED, probe=True, grad=False)

    qwen_targets, dit_targets = inject_control_lora(bare.model)
    for p in bare.model.parameters():
        p.requires_grad_("lora_" in next((n for n, q in bare.model.named_parameters() if q is p), ""))
    # The identity-initialized adapters should preserve the base function.
    lora_batch = move_value(clone_value(val_batch_cpu), device)
    with_lora_eval, lora_probe = run_forward(bare, clone_value(lora_batch), sf.VAL_SEED, probe=True, grad=False)
    lora_info = lora_report(bare.model, qwen_targets, dit_targets)
    effect = {
        "seed": sf.VAL_SEED,
        "same_batch": True,
        "without_lora": base_eval,
        "with_lora": with_lora_eval,
        "absolute_loss_differences": {k: abs(with_lora_eval[k] - base_eval[k]) for k in base_eval},
        "rng_probe_without_lora": base_probe,
        "rng_probe_with_lora": lora_probe,
        "noise_identical": base_probe["randn_like_calls"] == lora_probe["randn_like_calls"],
        "timestep_identical": base_probe["beta_samples"] == lora_probe["beta_samples"],
        "lora": lora_info,
    }
    (args.output_dir / "lora_init_effect.json").write_text(json.dumps(effect, indent=2) + "\n")

    step0 = {
        "checkpoint": str(args.model_ckpt),
        "requested_checkpoint": str(args.requested_checkpoint),
        "manifest": str(args.manifest),
        "split": "val",
        "batch_index": 0,
        "batch_size": int(val_batch_cpu["action"].shape[0]),
        "seed": sf.VAL_SEED,
        "optimizer_step": False,
        "model_training_flag": True,
        "loss_definition": "loss_policy = 0.5 * loss_mse + loss_freq; choice/score detached audit only",
        "without_lora": base_eval,
        "with_zero_initialized_lora": with_lora_eval,
        "choice_score_audit": {"choice": base_eval["loss_choice_audit"], "score": base_eval["loss_score_audit"]},
        "fixed_timestep_probe": base_probe["beta_samples"],
        "fixed_action_noise_probe": base_probe["randn_like_calls"],
    }
    (args.output_dir / "step0_eval.json").write_text(json.dumps(step0, indent=2) + "\n")

    pipeline = {
        "same_batch_reused": True,
        "input_tensor_summary_before_forward": base_input_summary,
        "input_tensor_summary_current_forward": summarize_tensors(move_value(clone_value(val_batch_cpu), device)),
        "input_tensor_exact_match": base_input_summary == summarize_tensors(move_value(clone_value(val_batch_cpu), device)),
        "difference_note": "The original checkpoint and current control forward use the same collated batch; no extra crop/normalization/action transform is applied by this diagnostic.",
        "policy_crop_contract": {"video_history_frames": 4, "official_center_crop_ratio": 0.95, "processor_do_resize": False, "dataset_current_spatial_rgb": "raw current frame for SF payload"},
        "action_contract": {"batch_action_summary": base_input_summary.get("action"), "vlm_action_target_summary": base_input_summary.get("vlm_action_target"), "action_stats_source": str(args.processor / "config.json"), "manual_action_renormalization_in_forward": False},
        "random_contract": {"seed": sf.VAL_SEED, "timestep_same": effect["timestep_identical"], "noise_same": effect["noise_identical"], "base_probe": base_probe, "lora_probe": lora_probe},
    }
    (args.output_dir / "pipeline_diff_report.json").write_text(json.dumps(pipeline, indent=2) + "\n")

    # 100-step control short test with the exact 20k optimizer/scheduler contract.
    bare.args = args_ns
    optimizer_cfg = sf.ForcingRunner.configure_optimizers(bare)
    optimizer = optimizer_cfg["optimizer"]
    scheduler = optimizer_cfg["lr_scheduler"]["scheduler"]
    train_log, val_log = [], []
    seed_all(424242)
    start = time.time()
    for step in range(1, 101):
        try:
            train_cpu = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            train_cpu = next(train_iter)
        train_batch = move_value(train_cpu, device)
        bare.model.train()
        optimizer.zero_grad(set_to_none=True)
        result = bare.model(train_batch, return_loss=True)
        losses = policy_result(result)
        loss = 0.5 * result["loss_mse"] + result["loss_freq"]
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % 10 == 0:
            train_log.append({"step": step, "train/loss": float(loss.detach().cpu()), "train/loss_mse": losses["loss_mse"], "train/loss_freq": losses["loss_freq"], "lr": float(optimizer.param_groups[0]["lr"])})
        if step % 20 == 0:
            state = capture_rng()
            seed_all(sf.VAL_SEED)
            bare.model.train()
            with torch.no_grad():
                val_result = bare.model(move_value(clone_value(val_batch_cpu), device), return_loss=True)
            val_losses = policy_result(val_result)
            restore_rng(state)
            val_log.append({"step": step, "val/loss_policy": val_losses["loss_policy"], "val/loss_mse": val_losses["loss_mse"], "val/loss_freq": val_losses["loss_freq"], "choice": val_losses["loss_choice_audit"], "score": val_losses["loss_score_audit"], "seed": sf.VAL_SEED})
            print(json.dumps({"short_test_step": step, "train": train_log[-1], "val": val_log[-1]}), flush=True)
    short = {
        "steps": 100,
        "batch_size": args.batch_size,
        "optimizer": "torch.optim.AdamW with current SF_control trainable scope",
        "scheduler": {"num_training_steps": 20000, "warmup_steps": 500, "max_lr": sf.LR, "min_lr": sf.MIN_LR},
        "train_every_10": train_log,
        "fixed_val_every_20": val_log,
        "elapsed_seconds": time.time() - start,
        "interpretation": "Compare step-10 train/loss against checkpoint step0 loss and inspect whether the large drop occurs before meaningful adapter updates.",
    }
    (args.output_dir / "short_test_100steps.json").write_text(json.dumps(short, indent=2) + "\n")
    print(json.dumps({"diagnostic_complete": True, "output_dir": str(args.output_dir), "step0": step0, "short_test_last": short["train_every_10"][-1] if train_log else None}), flush=True)


if __name__ == "__main__":
    main()
