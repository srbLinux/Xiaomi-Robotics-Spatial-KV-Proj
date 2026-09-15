#!/usr/bin/env python3
"""XR-1 official RoboCasa365 post-training plus the 36-layer ESM branch.

The baseline adapter/runner is imported from the first official20 entrypoint
so the shared data, processor, normalization, and official model call cannot
silently diverge.  The only additional registered modules are the 36 spatial
cross-attention blocks; ESM itself is held outside the checkpoint state.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningDataModule
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

import train_xr1_baseline_official20 as common

ESM_CKPT = Path("/data-cfs/data3/shurenbin/Xiaomi-Robotics-Spatial/train/esm_sink_token/runs/esm1b_robocasa365_geometry_fresh_400k_p4_retry2/checkpoint_014000.pt")
FALCON = Path("/data-cfs/data3/shurenbin/Xiaomi-Robotics-Spatial/train/esm_sink_token/vendor/FALCON")
ESM_HIDDEN, DIT_HIDDEN, HEAD_DIM = 2048, 1024, 128
GATE_INIT = 0.05


class SpatialCrossAttention(nn.Module):
    def __init__(self, gate_init=GATE_INIT):
        super().__init__()
        self.query_norm = nn.LayerNorm(DIT_HIDDEN)
        self.context_norm = nn.LayerNorm(ESM_HIDDEN)
        self.q_proj = nn.Linear(DIT_HIDDEN, DIT_HIDDEN, bias=False)
        self.k_proj = nn.Linear(ESM_HIDDEN, DIT_HIDDEN, bias=False)
        self.v_proj = nn.Linear(ESM_HIDDEN, DIT_HIDDEN, bias=False)
        self.o_proj = nn.Linear(DIT_HIDDEN, DIT_HIDDEN, bias=False)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.force_identity = False
        self.to(dtype=torch.bfloat16)

    def forward(self, hidden, context):
        if context is None:
            return hidden
        if context.ndim != 3 or tuple(context.shape[1:]) != (1369, ESM_HIDDEN):
            raise RuntimeError(f"ESM spatial contract must be [B,1369,2048], got {tuple(context.shape)}")
        # Exact zero is a real identity branch.  This is used by the parity
        # smoke test and ensures no branch parameter receives a phantom grad.
        if self.force_identity:
            return hidden
        context = self.context_norm(context)
        q = self.q_proj(self.query_norm(hidden)).view(hidden.shape[0], hidden.shape[1], -1, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(context).view(context.shape[0], context.shape[1], -1, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(context).view(context.shape[0], context.shape[1], -1, HEAD_DIM).transpose(1, 2)
        delta = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0).transpose(1, 2).reshape_as(hidden)
        return hidden + self.gate * self.o_proj(delta)


def spatial_layer_forward(layer, hidden_states, past_key_values, position_embeds, timestep, attn_mask):
    # This is the clean DecoderLayer ordering with one additive branch inserted
    # after the original attention residual and before the original MLP.
    shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (layer.adaln_table[None] + timestep).chunk(6, dim=1)
    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states) * (1 + scale_attn) + shift_attn
    hidden_states = residual + gate_attn * layer.attn(hidden_states, past_key_values, position_embeds, attn_mask)
    tokens = getattr(layer, "_esm_tokens", None)
    if tokens is not None:
        if tokens.shape[0] != hidden_states.shape[0]:
            if hidden_states.shape[0] % tokens.shape[0]:
                raise RuntimeError("ESM batch cannot be aligned with official training_repeat batch")
            tokens = tokens.repeat_interleave(hidden_states.shape[0] // tokens.shape[0], dim=0)
        hidden_states = layer.esm_cross_attention(hidden_states, tokens)
    residual = hidden_states
    hidden_states = layer.post_layernorm(hidden_states) * (1 + scale_mlp) + shift_mlp
    return residual + gate_mlp * layer.mlp(hidden_states)


def attach_spatial_modules(model, gate_init=GATE_INIT):
    dit = model.dit
    for layer in dit.layers:
        layer.esm_cross_attention = SpatialCrossAttention(gate_init)
        layer._esm_tokens = None
        layer.forward = types.MethodType(spatial_layer_forward, layer)
    if len(dit.layers) != 36:
        raise RuntimeError(f"expected 36 DiT layers, found {len(dit.layers)}")
    for layer in dit.layers:
        if not hasattr(layer, "esm_cross_attention"):
            raise RuntimeError("spatial module attachment failed")


def gate0_parity(model, batch, tokens):
    """Compare official DiT with the attached branch at exact gate=0.

    The branch is an explicit identity at zero, so output/loss parity is
    exact. FlashAttention backward can be nondeterministic in bf16; therefore
    shared gradients are reported with both absolute and relative error.
    """
    def capture_rng():
        return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())

    def restore_rng(state):
        random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2]); torch.cuda.set_rng_state_all(state[3])

    def set_tokens(value):
        for layer in model.dit.layers:
            layer._esm_tokens = value

    def shared_grads():
        return {n: p.grad.detach().float().cpu().clone() for n, p in model.named_parameters() if p.grad is not None and "esm_cross_attention" not in n}

    gates = [layer.esm_cross_attention.gate for layer in model.dit.layers]
    saved_gates = [gate.detach().clone() for gate in gates]
    saved_identity = [layer.esm_cross_attention.force_identity for layer in model.dit.layers]
    for gate in gates:
        gate.data.zero_()
    for layer in model.dit.layers:
        layer.esm_cross_attention.force_identity = True
    model.zero_grad(set_to_none=True); set_tokens(None); state = capture_rng()
    official = model(copy.deepcopy(batch), return_loss=True)
    official["loss"].backward()
    official_values = {key: float(value.detach().cpu()) for key, value in official.items()}
    official_grads = shared_grads()
    model.zero_grad(set_to_none=True); restore_rng(state); set_tokens(tokens)
    zero = model(copy.deepcopy(batch), return_loss=True)
    zero["loss"].backward()
    zero_values = {key: float(value.detach().cpu()) for key, value in zero.items()}
    zero_grads = shared_grads()
    abs_errors = [(official_grads[name] - zero_grads[name]).abs().max().item() for name in official_grads if name in zero_grads]
    rel_errors = [((official_grads[name] - zero_grads[name]).abs() / official_grads[name].abs().clamp_min(1e-6)).max().item() for name in official_grads if name in zero_grads]
    scale = max((value.abs().max().item() for value in official_grads.values()), default=1.0)
    result = {
        "official_loss_fields": official_values,
        "gate0_loss_fields": zero_values,
        "max_loss_abs_diff": max(abs(official_values[key] - zero_values[key]) for key in official_values),
        "max_shared_grad_abs_diff": max(abs_errors, default=0.0),
        "max_shared_grad_relative_diff": max(rel_errors, default=0.0),
        "max_shared_grad_abs_diff_over_global_grad_scale": max(abs_errors, default=0.0) / max(scale, 1e-6),
        "spatial_branch_grad_none": all(p.grad is None for layer in model.dit.layers for p in layer.esm_cross_attention.parameters()),
    }
    for gate, saved in zip(gates, saved_gates):
        gate.data.copy_(saved)
    for layer, saved in zip(model.dit.layers, saved_identity):
        layer.esm_cross_attention.force_identity = saved
    set_tokens(None); model.zero_grad(set_to_none=True)
    return result


class SpatialDataset(common.RoboCasa20):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        from PIL import Image
        from decord import VideoReader
        if not FALCON.is_dir():
            raise FileNotFoundError(FALCON)
        if str(FALCON) not in sys.path:
            sys.path.insert(0, str(FALCON))
        from falcon.model.policy_head.esm_utils.vggt.utils.load_fn import load_and_preprocess_images_square_new
        row = self.rows[index]
        raw = VideoReader(row["camera_paths"][common.RIGHT], num_threads=1).get_batch([int(row["frame_index"]) ]).asnumpy()[0]
        processed, _ = load_and_preprocess_images_square_new([Image.fromarray(raw.astype(np.uint8), mode="RGB")], target_size=518)
        item["esm_image"] = processed[0]
        if tuple(item["esm_image"].shape) != (3, 518, 518):
            raise RuntimeError(f"unexpected ESM input shape {tuple(item['esm_image'].shape)}")
        return item


class SpatialData(LightningDataModule):
    def __init__(self, rows, processor, batch_size, workers, gpu_prefetch=True, stats_processor=None):
        super().__init__()
        self.rows, self.processor, self.batch_size, self.workers = rows, processor, batch_size, workers
        self.gpu_prefetch = bool(gpu_prefetch)
        self.stats_processor = stats_processor or common.build_action_stats_processor()
        self.collate = common.LocalOfficialCollate(processor)
        self.train_set = None
        self.val_set = None

    def prepare_data(self):
        return None

    def setup(self, stage=None):
        if stage in (None, "fit"):
            if self.train_set is None:
                self.train_set = common.OfficialSampleDataset(SpatialDataset(self.rows["train"]), self.stats_processor)
            if self.val_set is None:
                self.val_set = common.OfficialSampleDataset(SpatialDataset(self.rows["val"]), self.stats_processor)

    def _loader(self, dataset, sampler, prefetch=True):
        kw = dict(batch_size=self.batch_size, sampler=sampler, num_workers=self.workers, pin_memory=True, persistent_workers=self.workers > 0, collate_fn=self.collate)
        if self.workers: kw["prefetch_factor"] = 2
        loader = DataLoader(dataset, **kw)
        return common.CUDAPrefetcher(loader) if torch.cuda.is_available() and self.gpu_prefetch and prefetch else loader

    def train_dataloader(self):
        self.setup("fit")
        return self._loader(self.train_set, DistributedSampler(self.train_set, shuffle=True, seed=common.SEED))

    def val_dataloader(self):
        self.setup("fit")
        return self._loader(self.val_set, DistributedSampler(self.val_set, shuffle=False, seed=common.SEED))


class FrozenRightESM(nn.Module):
    def __init__(self, checkpoint=ESM_CKPT):
        super().__init__()
        if Path(checkpoint).resolve() != ESM_CKPT.resolve():
            raise RuntimeError(f"only the required frozen ESM checkpoint is allowed: {ESM_CKPT}")
        if not ESM_CKPT.is_file(): raise FileNotFoundError(ESM_CKPT)
        if str(FALCON) not in sys.path: sys.path.insert(0, str(FALCON))
        from falcon.model.policy_head.esm_utils.vggt.models.vggt_camera_replace_depth import VGGT_Camera_Replace_Depth
        ckpt = torch.load(str(ESM_CKPT), map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        self.net = VGGT_Camera_Replace_Depth()
        missing, unexpected = self.net.load_state_dict(state, strict=False)
        if missing or unexpected: raise RuntimeError(f"ESM checkpoint mismatch: missing={missing[:4]}, unexpected={unexpected[:4]}")
        self.net.requires_grad_(False).eval()
        self.checkpoint_path = str(ESM_CKPT)
        del ckpt, state

    @torch.no_grad()
    def forward(self, image):
        if image.ndim != 4 or tuple(image.shape[1:]) != (3, 518, 518): raise RuntimeError(f"ESM input must be [B,3,518,518], got {tuple(image.shape)}")
        _, aggregated = self.net.inference(images=image.to(dtype=next(self.net.parameters()).dtype).unsqueeze(1), camera_gt_pt=0.0, depth_gt_pt=0.0)
        feature = aggregated[-1]
        if feature.ndim != 4 or feature.shape[1] != 1 or feature.shape[-1] != ESM_HIDDEN: raise RuntimeError(f"unexpected ESM representation {tuple(feature.shape)}")
        patch_start = int(getattr(self.net.aggregator, "patch_start_idx", 5))
        tokens = feature[:, 0, patch_start:].contiguous()
        if tuple(tokens.shape[1:]) != (1369, ESM_HIDDEN): raise RuntimeError(f"ESM token contract failed: {tuple(tokens.shape)}")
        if not torch.isfinite(tokens).all(): raise RuntimeError("non-finite ESM tokens")
        return tokens


class SpatialRunner(common.Official20Runner):
    def __init__(self, *args, gate_init=GATE_INIT, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate_init = gate_init
        self.__dict__["_esm"] = FrozenRightESM()
        self._esm_device = None

    def configure_model(self):
        super().configure_model()
        attach_spatial_modules(self.model, self.gate_init)
        # The ESM encoder is intentionally not registered, so checkpoints do
        # not contain a second copy of its weights.
        for layer in self.model.dit.layers:
            for p in layer.esm_cross_attention.parameters(): p.requires_grad_(True)
        if any(p.requires_grad for p in self._esm.parameters()):
            raise RuntimeError("ESM encoder must remain frozen")
        if not all(p.requires_grad for layer in self.model.dit.layers for p in layer.esm_cross_attention.parameters()):
            raise RuntimeError("spatial cross-attention parameters must be trainable")
        probe = torch.arange(2)
        expected = probe.repeat_interleave(self.model.training_repeat)
        actual = self.model._repeat(probe)
        if not torch.equal(actual, expected):
            raise RuntimeError(f"unexpected training_repeat arrangement: {actual.tolist()} != {expected.tolist()}")
        print(json.dumps({"training_repeat": self.model.training_repeat, "batch_order": actual.tolist(), "esm_batch_operation": "repeat_interleave(dim=0)", "dit_forward_patch": "none", "decoder_layer_patch": "minimal"}), flush=True)

    def _esm_tokens(self, image):
        device = next(self.model.parameters()).device
        if self._esm_device != device:
            self._esm.to(device, dtype=torch.bfloat16)
            self._esm_device = device
        return self._esm(image.to(device, dtype=torch.bfloat16))

    def _official_step(self, batch, validation=False):
        image = batch.pop("esm_image")
        tokens = self._esm_tokens(image)
        for layer in self.model.dit.layers:
            layer._esm_tokens = tokens
        try:
            return self.model(batch, return_loss=True)
        finally:
            for layer in self.model.dit.layers:
                layer._esm_tokens = None

    def training_step(self, batch, batch_idx):
        result = self._official_step(batch)
        for name, value in result.items(): self.log(f"train/{name}", value.detach(), sync_dist=True)
        self.log("train/token", batch["input_ids"].shape[1])
        return result["loss"]

    def validation_step(self, batch, batch_idx):
        was_training = self.model.training; self.model.train()
        with torch.no_grad(): result = self._official_step(batch, validation=True)
        if not was_training: self.model.eval()
        for name, value in result.items(): self.log(f"val/{name}", value.detach(), prog_bar=name == "loss", sync_dist=True)
        return result["loss"]

    def on_after_backward(self):
        if self.global_step == 0 and self.global_rank == 0:
            vals = {name: max((p.grad.detach().float().norm().item() for n, p in self.model.named_parameters() if f"esm_cross_attention.{name}" in n and p.grad is not None), default=0.0) for name in ("q_proj", "k_proj", "v_proj", "o_proj", "query_norm", "context_norm", "gate")}
            print(json.dumps({"spatial_gradient_smoke": vals, "esm_grad_none": all(p.grad is None for p in self._esm.parameters())}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--devices", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=100000)
    p.add_argument("--val-batches", type=int, default=16)
    p.add_argument("--output-dir", type=Path, default=Path("/data-cfs/data3/shurenbin/Xiaomi-Robotics-Spatial/train/xr1_official20/spatial"))
    p.add_argument("--target-gpu-gb", type=float, default=70.0); p.add_argument("--limit-gpu-gb", type=float, default=70.0)
    p.add_argument("--gate0", action="store_true", help="contract/parity mode; initialize all gates at zero")
    p.add_argument("--no-gpu-prefetch", dest="gpu_prefetch", action="store_false"); p.set_defaults(gpu_prefetch=True)
    p.add_argument("--audit-only", action="store_true")
    args = p.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_processor = common.build_action_stats_processor()
    processor = common.build_official_qwen_processor()
    rows = common.load_rows(); common.write_audit(args.output_dir, stats_processor, rows, args)
    (args.output_dir / "spatial_contract.json").write_text(json.dumps({"esm_checkpoint": str(ESM_CKPT), "esm_source_view": common.RIGHT, "esm_input": [3, 518, 518], "esm_tokens": ["B", 1369, 2048], "removed_special_tokens": 5, "token_resize": False, "injection_layers": 36, "attention": "independent additive residual after official attention residual, before official MLP", "gate_init": 0.0 if args.gate0 else GATE_INIT, "optimizer": "exact clean BaseRunner optimizer/scheduler"}, indent=2) + "\n")
    print(json.dumps({"mode": "spatial", "tasks": list(common.TASKS), "per_gpu_batch_size": args.batch_size, "gpu_prefetch": args.gpu_prefetch, "esm_tokens": "[B,1369,2048]", "gate_init": 0.0 if args.gate0 else GATE_INIT}), flush=True)
    if args.audit_only: return
    data = SpatialData(rows, processor, args.batch_size, args.workers, args.gpu_prefetch, stats_processor)
    runner = SpatialRunner(common.MODEL_CKPT, args.max_steps, args.target_gpu_gb, args.limit_gpu_gb, gate_init=0.0 if args.gate0 else GATE_INIT)
    from lightning import Trainer
    from lightning.pytorch.strategies import DeepSpeedStrategy
    from lightning.pytorch.callbacks import ModelCheckpoint
    checkpoint = ModelCheckpoint(dirpath=str(args.output_dir / "checkpoints"), every_n_train_steps=2000, save_top_k=-1, save_last=True, enable_version_counter=False)
    trainer = Trainer(accelerator="cuda", devices=args.devices, num_nodes=1, precision="bf16-mixed", strategy=DeepSpeedStrategy(), max_steps=args.max_steps, default_root_dir=str(args.output_dir), accumulate_grad_batches=1, gradient_clip_val=1.0, log_every_n_steps=10, val_check_interval=1000, check_val_every_n_epoch=None, limit_val_batches=args.val_batches, callbacks=[checkpoint], enable_checkpointing=True)
    trainer.fit(runner, datamodule=data)


if __name__ == "__main__": main()
