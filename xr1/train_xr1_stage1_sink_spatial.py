#!/usr/bin/env python3
"""Stage1 Sink Spatial: frozen RoboCasa365 XR-1 + one spatial sink adapter.

The policy weights are loaded directly from the published RoboCasa365
checkpoint.  Only ``sink_spatial_adapter`` is trainable.  The ESM encoder is
frozen and runs once per raw sample before XR-1's native training repeat.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch
from lightning import Trainer
from lightning.pytorch import LightningDataModule
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.strategies import DeepSpeedStrategy
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

import train_xr1_baseline_official20 as common

BASE_CHECKPOINT = Path("/data-cfs/data3/models/Xiaomi-Robotics-1-RoboCasa365")
ESM_CHECKPOINT = Path("/data-cfs/data3/shurenbin/Xiaomi-Robotics-Spatial/train/esm_sink_token/runs/esm1b_robocasa365_geometry_fresh_400k_p4_retry2/checkpoint_014000.pt")
FALCON_ROOT = Path("/data-cfs/data3/shurenbin/Xiaomi-Robotics-Spatial/train/esm_sink_token/vendor/FALCON")
TASKS = ("CloseBlenderLid", "CoffeeSetupMug", "TurnOffStove", "TurnOnMicrowave", "SlideDishwasherRack", "PickPlaceCounterToCabinet")
ESM_HIDDEN, DIT_HIDDEN = 2048, 1024
TRAINING_REPEAT = 4
DIAGNOSTIC_EVERY = 500
PER_GPU_BATCH = 64
WORLD_SIZE = 2
PROGRESS = (0.0, 0.25, 0.5, 0.75)


def assert_policy_checkpoint(path: Path) -> dict:
    expected = BASE_CHECKPOINT.resolve()
    if path.resolve() != expected:
        raise RuntimeError(f"MODEL_CKPT must be exactly {expected}, got {path.resolve()}")
    index_path = path / "model.safetensors.index.json"
    if not path.is_dir() or not (path / "config.json").is_file() or not index_path.is_file():
        raise RuntimeError(f"RoboCasa365 policy checkpoint is incomplete: {path}")
    index = json.loads(index_path.read_text())
    shards = sorted(set(index["weight_map"].values()))
    for shard in shards:
        if not (path / shard).is_file():
            raise RuntimeError(f"missing RoboCasa365 policy shard: {path / shard}")
    return index


def load_robocasa_policy(model: nn.Module, checkpoint: Path) -> dict:
    """Load every shared policy tensor from the RoboCasa365 safetensors.

    The released inference checkpoint intentionally omits the training-only
    choice/score heads and their two special token embeddings.  Those exact
    15 tensors are allowed to remain freshly initialized; all policy tensors
    must come from the requested RoboCasa365 checkpoint.
    """
    from safetensors.torch import load_file

    index = assert_policy_checkpoint(checkpoint)
    weight_map = index["weight_map"]
    model_state = model.state_dict()
    model_keys, checkpoint_keys = set(model_state), set(weight_map)
    unexpected = sorted(checkpoint_keys - model_keys)
    missing = sorted(model_keys - checkpoint_keys)
    expected_missing = sorted(
        [key for key in model_keys if key.startswith(("state_projector_choice.", "action_projector_choice.", "score_projector_choice."))]
        + ["vlm.lm_head.weight", "vlm.model.action_embed.weight", "vlm.model.score_embed.weight"]
    )
    if unexpected or missing != expected_missing:
        raise RuntimeError(f"RoboCasa365 state contract mismatch: missing={missing}, unexpected={unexpected[:8]}")

    for shard in sorted(set(weight_map.values())):
        tensors = load_file(str(checkpoint / shard), device="cpu")
        for key, value in tensors.items():
            model_state[key].copy_(value.to(dtype=model_state[key].dtype))
        del tensors
    return {"checkpoint": str(checkpoint), "loaded_tensors": len(checkpoint_keys), "allowed_training_only_missing": expected_missing}


class SinkSpatialAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(ESM_HIDDEN, DIT_HIDDEN), nn.GELU(), nn.Linear(DIT_HIDDEN, DIT_HIDDEN))
        last_linear = self.layers[-1]
        nn.init.zeros_(last_linear.weight)
        if last_linear.bias is not None:
            nn.init.zeros_(last_linear.bias)

    def forward(self, esm_global):
        if tuple(esm_global.shape[1:]) != (ESM_HIDDEN,):
            raise RuntimeError(f"adapter input must be [B,2048], got {tuple(esm_global.shape)}")
        return self.layers(esm_global)


def sink_spatial_dit_forward(self, noisy_action, timestep, action_mask, state_embed, position_embeds, past_key_values, attn_mask, prefix_length):
    timestep = self.t_projector(self.t_embedder(timestep[:, 0, 0] * 1000)).view(-1, 6, DIT_HIDDEN)
    noisy_action = self.action_projector(noisy_action * action_mask)
    spatial_delta = self._sink_spatial_delta
    if spatial_delta is None:
        raise RuntimeError("spatial delta was not prepared before XR-1 forward")
    if spatial_delta.ndim != 2 or spatial_delta.shape[1] != DIT_HIDDEN:
        raise RuntimeError(f"spatial delta must be [B,1024], got {tuple(spatial_delta.shape)}")
    if spatial_delta.shape[0] != state_embed.shape[0]:
        if state_embed.shape[0] % spatial_delta.shape[0]:
            raise RuntimeError(f"training-repeat mismatch: delta={spatial_delta.shape[0]}, dit={state_embed.shape[0]}")
        spatial_delta = spatial_delta.repeat_interleave(state_embed.shape[0] // spatial_delta.shape[0], dim=0)
    sink = self.sink.weight[None].repeat(state_embed.shape[0], 1, 1)
    sink = sink + spatial_delta[:, None, :].to(dtype=sink.dtype)
    hidden_states = torch.cat([sink, state_embed, noisy_action], dim=1).contiguous()
    hidden_states = self.dit(hidden_states, past_key_values, attn_mask, position_embeds, timestep)
    output = self.action_output_layer(hidden_states[:, -noisy_action.shape[1]:])
    if prefix_length:
        output[:, :prefix_length] = 0.0
    return output


def load_rows(manifest: Path) -> dict[str, list[dict]]:
    meta_path = Path(str(manifest).replace(".jsonl", ".meta.json"))
    meta = json.loads(meta_path.read_text())
    if tuple(meta.get("task_names", ())) != TASKS:
        raise RuntimeError("Stage1 Sink Spatial manifest task contract mismatch")
    rows = {"train": [], "val": []}
    for line in manifest.open():
        row = json.loads(line)
        if row.get("task") in TASKS and row.get("split") in rows:
            rows[row["split"]].append(row)
    if not rows["train"] or not rows["val"]:
        raise RuntimeError("Stage1 Sink Spatial manifest has an empty split")
    train_eps = {(r["task"], r["episode_key"]) for r in rows["train"]}
    val_eps = {(r["task"], r["episode_key"]) for r in rows["val"]}
    if train_eps & val_eps:
        raise RuntimeError("Stage1 Sink Spatial train/val episode leakage")
    return rows


def build_diagnostic_specs(rows: dict[str, list[dict]]) -> list[dict]:
    by_episode = {(r["task"], r["episode_key"], int(r["frame_index"])): r for r in rows["val"]}
    specs = []
    for task in TASKS:
        episodes = sorted({r["episode_key"] for r in rows["val"] if r["task"] == task})
        if len(episodes) != 2:
            raise RuntimeError(f"{task} must have exactly 2 validation episodes, got {len(episodes)}")
        lengths = {ep: max(int(r["frame_index"]) for r in rows["val"] if r["task"] == task and r["episode_key"] == ep) + 1 for ep in episodes}
        for progress in PROGRESS:
            for target_ep, donor_ep in zip(episodes, episodes[::-1]):
                frame = round(progress * (lengths[target_ep] - 1))
                donor_frame = round(progress * (lengths[donor_ep] - 1))
                target = by_episode[(task, target_ep, frame)]
                donor = by_episode[(task, donor_ep, donor_frame)]
                specs.append({"task": task, "progress": progress, "target": target["sample_id"], "donor": donor["sample_id"]})
    if len(specs) != 48:
        raise RuntimeError(f"Stage1 Sink Spatial diagnostic must contain 48 samples, got {len(specs)}")
    return specs


class SpatialDataset(common.RoboCasa20):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        from PIL import Image
        from decord import VideoReader
        if str(FALCON_ROOT) not in sys.path:
            sys.path.insert(0, str(FALCON_ROOT))
        from falcon.model.policy_head.esm_utils.vggt.utils.load_fn import load_and_preprocess_images_square_new

        row = self.rows[index]
        raw = VideoReader(row["camera_paths"][common.RIGHT], num_threads=1).get_batch([int(row["frame_index"]) ]).asnumpy()[0]
        processed, _ = load_and_preprocess_images_square_new([Image.fromarray(raw.astype(np.uint8), mode="RGB")], target_size=518)
        item["esm_image"] = processed[0]
        if tuple(item["esm_image"].shape) != (3, 518, 518):
            raise RuntimeError(f"ESM input contract failed: {tuple(item['esm_image'].shape)}")
        return item


class Stage1Data(LightningDataModule):
    def __init__(self, rows, processor, batch_size, workers, stats_processor):
        super().__init__()
        self.rows, self.processor, self.batch_size, self.workers, self.stats_processor = rows, processor, batch_size, workers, stats_processor
        self.collate = common.LocalOfficialCollate(processor)

    def setup(self, stage=None):
        if stage in (None, "fit") and not hasattr(self, "train_set"):
            self.train_set = common.OfficialSampleDataset(SpatialDataset(self.rows["train"]), self.stats_processor)
            self.val_set = common.OfficialSampleDataset(SpatialDataset(self.rows["val"]), self.stats_processor)

    def _loader(self, dataset, sampler, drop_last):
        kwargs = dict(batch_size=self.batch_size, sampler=sampler, drop_last=drop_last, num_workers=self.workers, pin_memory=True, persistent_workers=self.workers > 0, collate_fn=self.collate)
        if self.workers:
            kwargs["prefetch_factor"] = 2
        loader = DataLoader(dataset, **kwargs)
        return common.CUDAPrefetcher(loader) if torch.cuda.is_available() else loader

    def train_dataloader(self):
        self.setup("fit")
        return self._loader(self.train_set, DistributedSampler(self.train_set, shuffle=True, seed=common.SEED), True)

    def val_dataloader(self):
        self.setup("fit")
        return self._loader(self.val_set, DistributedSampler(self.val_set, shuffle=False, seed=common.SEED), False)


class FrozenRightESM(nn.Module):
    def __init__(self):
        super().__init__()
        if not ESM_CHECKPOINT.is_file() or not FALCON_ROOT.is_dir():
            raise FileNotFoundError(f"missing ESM checkpoint/FALCON: {ESM_CHECKPOINT}, {FALCON_ROOT}")
        sys.path.insert(0, str(FALCON_ROOT))
        os.environ["FALCON_DINOV2_REPO"] = "/data-cfs/data3/models/dinov2"
        os.environ["TORCH_HOME"] = "/data-cfs/data3/models/torch"
        from falcon.model.policy_head.esm_utils.vggt.models.vggt_camera_replace_depth import VGGT_Camera_Replace_Depth

        checkpoint = torch.load(str(ESM_CHECKPOINT), map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        self.net = VGGT_Camera_Replace_Depth()
        missing, unexpected = self.net.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"ESM checkpoint mismatch: missing={missing[:4]}, unexpected={unexpected[:4]}")
        self.net.requires_grad_(False).eval()
        del checkpoint, state

    @torch.no_grad()
    def forward(self, image):
        self.eval()
        if tuple(image.shape[1:]) != (3, 518, 518):
            raise RuntimeError(f"ESM input must be [B,3,518,518], got {tuple(image.shape)}")
        dtype = next(self.net.parameters()).dtype
        _, aggregated = self.net.inference(images=image.to(dtype=dtype).unsqueeze(1), camera_gt_pt=0.0, depth_gt_pt=0.0)
        feature = aggregated[-1]
        if int(getattr(self.net.aggregator, "patch_start_idx", 5)) != 5 or feature.ndim != 4 or feature.shape[1] != 1 or feature.shape[-1] != ESM_HIDDEN:
            raise RuntimeError(f"ESM final/stage23 contract failed: {tuple(feature.shape)}")
        tokens = feature[:, 0, 5:].contiguous()
        if tuple(tokens.shape[1:]) != (1369, ESM_HIDDEN) or not torch.isfinite(tokens).all():
            raise RuntimeError(f"ESM token contract failed: {tuple(tokens.shape)}")
        return tokens


class Stage1Runner(common.Official20Runner):
    def __init__(self, *args, diagnostic_specs, diagnostic_rows, output_dir, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__["_esm"] = FrozenRightESM()
        self.diagnostic_specs = diagnostic_specs
        self._diagnostic_rows = diagnostic_rows
        self.output_dir = Path(output_dir)
        self._diag_pairs = None
        self._diag_token_cache = None

    def configure_model(self):
        common.use_local_vlm_config()
        from mibot.models.VLA.XR1 import xr1

        common.MODEL_CKPT = BASE_CHECKPOINT
        common.PROCESSOR = BASE_CHECKPOINT
        assert_policy_checkpoint(common.MODEL_CKPT)
        self.model = xr1(freq_coefficient=1.0, freq_excluded_dims=[17, 18, 19], ffn_gradient_checkpointing=True, async_train=True)
        self.model.state_shape = (4, 60)
        self.model.action_shape = (16, 60)
        load_info = load_robocasa_policy(self.model, BASE_CHECKPOINT)
        self.model.sink_spatial_adapter = SinkSpatialAdapter()
        self.model._sink_spatial_delta = None
        self.model.dit_forward = types.MethodType(sink_spatial_dit_forward, self.model)
        self.model.requires_grad_(False)
        self.model.sink_spatial_adapter.requires_grad_(True)
        if int(self.model.training_repeat) != TRAINING_REPEAT:
            raise RuntimeError(f"XR-1 training_repeat must be {TRAINING_REPEAT}, got {self.model.training_repeat}")
        trainable = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        expected = {f"sink_spatial_adapter.layers.0.{x}" for x in ("weight", "bias")} | {f"sink_spatial_adapter.layers.2.{x}" for x in ("weight", "bias")}
        if {n for n, _ in trainable} != expected:
            raise RuntimeError(f"trainable scope mismatch: {[n for n, _ in trainable]}")
        if self.model.sink_spatial_adapter.layers[-1].weight.abs().sum().item() != 0:
            raise RuntimeError("last spatial adapter Linear is not zero initialized")
        print(json.dumps({"stage1_sink_spatial": True, "policy_checkpoint": load_info, "esm_frozen": True, "esm_contract": [1369, 2048], "pool": "max", "adapter": "2048->1024->1024", "last_linear_zero": True, "gate": None, "dit_attention_patched": False, "training_repeat": TRAINING_REPEAT, "trainable": [n for n, _ in trainable]}), flush=True)

    def configure_optimizers(self):
        if os.environ.get("STAGE1_OPTIMIZER") == "adamw":
            cfg = {"type": "torch.optim.AdamW", "params": dict(self._optimizer["params"])}
            optimizer = self.build_optimizer(cfg, self.named_parameters())
            scheduler = self.build_scheduler(self._scheduler, optimizer)
            result = {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        else:
            result = super().configure_optimizers()
        actual = {id(p) for group in result["optimizer"].param_groups for p in group["params"]}
        expected = {id(p) for p in self.model.sink_spatial_adapter.parameters()}
        if actual != expected:
            raise RuntimeError("optimizer is not restricted to sink_spatial_adapter")
        return result

    def _esm_tokens(self, image):
        device = next(self.model.parameters()).device
        self._esm.to(device=device, dtype=torch.bfloat16)
        self._esm.eval()
        tokens = self._esm(image.to(device=device, dtype=torch.bfloat16))
        if tuple(tokens.shape[1:]) != (1369, ESM_HIDDEN):
            raise RuntimeError(f"ESM token contract failed before pooling: {tuple(tokens.shape)}")
        return tokens

    def _forward_with_spatial(self, batch, mode="correct", wrong_image=None, cached_tokens=None):
        work = dict(batch)
        image = work.pop("esm_image")
        device = next(self.model.parameters()).device
        if mode == "noesm":
            spatial_delta = torch.zeros((image.shape[0], DIT_HIDDEN), device=device, dtype=self.model.sink.weight.dtype)
        else:
            with torch.no_grad():
                tokens = cached_tokens if cached_tokens is not None else self._esm_tokens(image)
                esm_global = tokens.to(device=device).max(dim=1).values
            spatial_delta = self.model.sink_spatial_adapter(esm_global)
        self.model._sink_spatial_delta = spatial_delta
        try:
            result = self.model(work, return_loss=True)
            # The released inference checkpoint has no trained choice/score
            # heads.  Keep the official total for audit, but do not optimize
            # it: only the DiT losses can carry gradients to this adapter.
            result["loss_official_raw"] = result["loss"].detach()
            result["loss_stage1"] = 0.5 * result["loss_mse"] + result["loss_freq"]
            return result
        finally:
            self.model._sink_spatial_delta = None

    @staticmethod
    def _move(value, device):
        if isinstance(value, dict):
            return {key: Stage1Runner._move(item, device) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(Stage1Runner._move(item, device) for item in value)
        return value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value

    @staticmethod
    def _capture_rng():
        return random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()

    @staticmethod
    def _restore_rng(state):
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
        torch.cuda.set_rng_state_all(state[3])

    def _prepare_diag_pairs(self):
        if self._diag_pairs is not None:
            return
        processor = common.build_official_qwen_processor()
        stats = common.build_action_stats_processor()
        by_id = {row["sample_id"]: row for row in self._diagnostic_rows}
        targets = [by_id[item["target"]] for item in self.diagnostic_specs]
        donors = [by_id[item["donor"]] for item in self.diagnostic_specs]
        target_ds = common.OfficialSampleDataset(SpatialDataset(targets), stats)
        donor_ds = SpatialDataset(donors)
        target_items = [target_ds[index] for index in range(len(target_ds))]
        donor_items = [donor_ds[index] for index in range(len(donor_ds))]
        collate = common.LocalOfficialCollate(processor)
        self._diag_pairs = []
        for index, spec in enumerate(self.diagnostic_specs):
            target_batch = collate([target_items[index]])
            wrong_image = donor_items[index]["esm_image"].unsqueeze(0)
            self._diag_pairs.append((spec, target_batch, wrong_image))
        if len(self._diag_pairs) != 48:
            raise RuntimeError(f"diagnostic pair count must be 48, got {len(self._diag_pairs)}")

    def _cache_diag_tokens(self):
        if self._diag_token_cache is not None:
            return
        device = next(self.model.parameters()).device
        self._diag_token_cache = []
        with torch.no_grad():
            for _, cpu_batch, cpu_wrong in self._diag_pairs:
                correct = self._esm_tokens(cpu_batch["esm_image"].to(device)).detach().cpu()
                wrong = self._esm_tokens(cpu_wrong.to(device)).detach().cpu()
                self._diag_token_cache.append((correct, wrong))

    @staticmethod
    def _avg(records, mode, key):
        return sum(record[mode][key] for record in records) / len(records)

    def _summarize_diag(self, records):
        def deltas(key):
            return [record["wrong"][key] - record["correct"][key] for record in records]

        def summarize(values):
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
            margin = 1.96 * (variance / len(values)) ** 0.5
            return {"mean": mean, "low": mean - margin, "high": mean + margin, "n": len(values)}

        result = {
            mode: {key: self._avg(records, mode, key) for key in ("loss_stage1", "loss_official_raw", "loss_mse", "loss_freq")}
            for mode in ("correct", "wrong", "noesm")
        }
        wrong_stage1 = deltas("loss_stage1")
        noesm_stage1 = [record["noesm"]["loss_stage1"] - record["correct"]["loss_stage1"] for record in records]
        result["wrong_minus_correct"] = summarize(wrong_stage1)
        result["noesm_minus_correct"] = summarize(noesm_stage1)
        result["fraction_wrong_gt_correct"] = sum(value > 0 for value in wrong_stage1) / len(wrong_stage1)
        result["fraction_noesm_gt_correct"] = sum(value > 0 for value in noesm_stage1) / len(noesm_stage1)
        return result

    def _write_diagnostics(self, records):
        payload = {
            "step": int(self.global_step),
            "sample_count": len(records),
            "objective": "0.5*loss_mse + loss_freq",
            "official_total_loss_is_audit_only": True,
            "overall": self._summarize_diag(records),
            "by_task": {task: self._summarize_diag([r for r in records if r["spec"]["task"] == task]) for task in TASKS},
            "by_progress": {f"{int(progress * 100)}%": self._summarize_diag([r for r in records if r["spec"]["progress"] == progress]) for progress in PROGRESS},
            "records": records,
        }
        path = self.output_dir / "diagnostics"
        path.mkdir(parents=True, exist_ok=True)
        (path / f"step_{int(self.global_step):06d}.json").write_text(json.dumps(payload, indent=2) + "\n")
        with (path / "diagnostics.jsonl").open("a") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _run_diagnostics(self, force=False):
        if not force and (self.global_step <= 0 or self.global_step % DIAGNOSTIC_EVERY):
            return
        self._prepare_diag_pairs()
        self._cache_diag_tokens()
        outer_rng = self._capture_rng()
        was_training = self.model.training
        self.model.train()
        records = []
        try:
            device = next(self.model.parameters()).device
            for index, (spec, cpu_batch, cpu_wrong) in enumerate(self._diag_pairs):
                batch = self._move(cpu_batch, device)
                correct_tokens, wrong_tokens = self._diag_token_cache[index]
                rng = self._capture_rng()
                outputs = {}
                with torch.no_grad():
                    for mode, tokens in (("correct", correct_tokens), ("wrong", wrong_tokens), ("noesm", None)):
                        self._restore_rng(rng)
                        result = self._forward_with_spatial(batch, mode=mode, cached_tokens=None if tokens is None else tokens.to(device))
                        outputs[mode] = {key: float(result[key].detach().cpu()) for key in ("loss_stage1", "loss_official_raw", "loss_mse", "loss_freq")}
                records.append({"spec": spec, **outputs})
        finally:
            self._restore_rng(outer_rng)
            if not was_training:
                self.model.eval()
        if self.global_rank == 0:
            self._write_diagnostics(records)
            print(json.dumps({"diagnostic_step": int(self.global_step), "sample_count": len(records), "wrong_minus_correct": self._summarize_diag(records)["wrong_minus_correct"], "noesm_minus_correct": self._summarize_diag(records)["noesm_minus_correct"]}), flush=True)

    def on_train_start(self):
        self._esm.eval()
        self._run_diagnostics(force=True)
        if self.global_rank == 0:
            print(json.dumps({"training_start": True, "devices": torch.cuda.device_count(), "max_steps": self.max_steps, "training_repeat": TRAINING_REPEAT, "per_gpu_batch": self.trainer.datamodule.batch_size, "optimized_loss": "0.5*loss_mse+loss_freq", "official_total_loss": "audit_only_due_to_random_frozen_choice_score_heads"}), flush=True)

    def training_step(self, batch, batch_idx):
        result = self._forward_with_spatial(batch)
        for name in ("loss_stage1", "loss_official_raw", "loss_mse", "loss_freq", "loss_l1", "loss_score"):
            self.log(f"train/{name}", result[name].detach(), sync_dist=True)
        return result["loss_stage1"]

    def validation_step(self, batch, batch_idx):
        was_training = self.model.training
        self.model.train()
        with torch.no_grad():
            result = self._forward_with_spatial(batch)
        if not was_training:
            self.model.eval()
        for name, value in result.items():
            if name in ("loss_stage1", "loss_official_raw", "loss_mse", "loss_freq", "loss_l1", "loss_score"):
                self.log(f"val/{name}", value.detach(), sync_dist=True, prog_bar=name == "loss_stage1")
        return result["loss_stage1"]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self._run_diagnostics()

    def on_after_backward(self):
        if self.global_step == 0:
            bad = [n for n, p in self.model.named_parameters() if p.grad is not None and not n.startswith("sink_spatial_adapter.")]
            if bad or any(p.grad is not None for p in self._esm.parameters()):
                raise RuntimeError(f"frozen parameter received gradient: {bad[:8]}")


class StepMarker(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if trainer.is_global_zero and step and step % 100 == 0:
            print(json.dumps({"step_marker": step}), flush=True)


class AdapterCheckpoint(Callback):
    """Save only the lightweight trainable interface, not the 5B frozen model."""
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir) / "checkpoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if not trainer.is_global_zero or step <= 0 or step % 1000:
            return
        state = {key: value.detach().cpu().clone() for key, value in pl_module.model.sink_spatial_adapter.state_dict().items()}
        torch.save({"state_dict": state, "metadata": {"step": step, "base_policy": str(BASE_CHECKPOINT), "esm_checkpoint": str(ESM_CHECKPOINT), "objective": "0.5*loss_mse + loss_freq"}}, self.output_dir / f"sink_spatial_adapter_step_{step:06d}.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=PER_GPU_BATCH)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--val-batches", type=int, default=16)
    parser.add_argument("--target-gpu-gb", type=float, default=68.0)
    parser.add_argument("--limit-gpu-gb", type=float, default=70.0)
    parser.add_argument("--acceptance-only", action="store_true")
    args = parser.parse_args()
    if args.devices != WORLD_SIZE:
        raise ValueError(f"Stage1 Sink Spatial requires exactly {WORLD_SIZE} devices")
    if not args.acceptance_only and (args.batch_size != PER_GPU_BATCH or args.max_steps != 20000):
        raise ValueError(f"formal Stage1 Sink Spatial requires batch_size={PER_GPU_BATCH}, devices={WORLD_SIZE}, max_steps=20000")
    common.MODEL_CKPT = BASE_CHECKPOINT
    common.PROCESSOR = BASE_CHECKPOINT
    assert_policy_checkpoint(BASE_CHECKPOINT)
    rows = load_rows(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_processor = common.build_action_stats_processor()
    processor = common.build_official_qwen_processor()
    specs = build_diagnostic_specs(rows)
    (args.output_dir / "experiment_contract.json").write_text(json.dumps({"experiment": "Stage1 Sink Spatial", "model_ckpt": str(BASE_CHECKPOINT), "esm_ckpt": str(ESM_CHECKPOINT), "devices": args.devices, "batch_size": args.batch_size, "max_steps": args.max_steps, "workers_per_process": args.workers, "prefetch_factor": 2, "training_repeat": TRAINING_REPEAT, "esm_view": "right", "esm_frame": "current", "esm_input": [3, 518, 518], "esm_tokens": [1369, 2048], "pool": "max", "adapter": [2048, 1024, 1024], "last_linear_zero_init": True, "gate": None, "diT_attention_math": "official_untouched", "diagnostic": {"samples": 48, "at": [0, 500, 1000], "modes": ["correct", "wrong", "noesm"], "delta": "wrong_minus_correct"}, "optimized_loss": "0.5*loss_mse + loss_freq", "official_total_loss": "audit_only_due_to_random_frozen_choice_score_heads", "train_rows": len(rows["train"]), "val_rows": len(rows["val"])}, indent=2) + "\n")
    (args.output_dir / "diagnostic_sample_ids.json").write_text(json.dumps(specs, indent=2) + "\n")
    if args.acceptance_only:
        max_steps, val_interval, limit_train = 2, 999999, 2
        callbacks = [StepMarker()]
    else:
        max_steps, val_interval, limit_train = args.max_steps, 1000, None
        callbacks = [AdapterCheckpoint(args.output_dir), StepMarker()]
    data = Stage1Data(rows, processor, args.batch_size, args.workers, stats_processor)
    runner = Stage1Runner(BASE_CHECKPOINT, max_steps, args.target_gpu_gb, args.limit_gpu_gb, diagnostic_specs=specs, diagnostic_rows=rows["val"], output_dir=args.output_dir)
    trainer = Trainer(accelerator="cuda", devices=args.devices, num_nodes=1, precision="bf16-mixed", strategy=DeepSpeedStrategy(), max_steps=max_steps, default_root_dir=str(args.output_dir), accumulate_grad_batches=1, gradient_clip_val=1.0, log_every_n_steps=10, val_check_interval=val_interval, check_val_every_n_epoch=None, limit_train_batches=limit_train, limit_val_batches=args.val_batches, num_sanity_val_steps=0, callbacks=callbacks, enable_checkpointing=False)
    trainer.fit(runner, datamodule=data)


if __name__ == "__main__":
    main()
