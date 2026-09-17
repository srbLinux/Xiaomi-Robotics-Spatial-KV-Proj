#!/usr/bin/env python3
"""Stage1: frozen XR-1 plus a trainable 36-layer ESM spatial interface."""
from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from lightning import Trainer
from lightning.pytorch import LightningDataModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

import train_xr1_baseline_official20 as common

ESM_CKPT = Path("/data-cfs/data3/shurenbin/Xiaomi-Robotics-Spatial/train/esm_sink_token/runs/esm1b_robocasa365_geometry_fresh_400k_p4_retry2/checkpoint_014000.pt")
FALCON = Path("/data-cfs/data3/shurenbin/Xiaomi-Robotics-Spatial/train/esm_sink_token/vendor/FALCON")
TASKS = ("CloseBlenderLid", "CoffeeSetupMug", "TurnOffStove", "TurnOnMicrowave", "SlideDishwasherRack", "PickPlaceCounterToCabinet")
PROGRESS = (0.0, 0.25, 0.5, 0.75)
ESM_HIDDEN, DIT_HIDDEN, HEAD_DIM = 2048, 1024, 128
GATE = 0.02
BASE_CHECKPOINT = Path("/data-cfs/data3/models/Xiaomi-Robotics-1-RoboCasa365")
TRAINING_REPEAT = 4
DIAGNOSTIC_EVERY = 500
VAL_EVERY = CHECKPOINT_EVERY = 1000
PER_GPU_BATCH = 64
WORLD_SIZE = 2


def assert_base_checkpoint():
    expected = Path("/data-cfs/data3/models/Xiaomi-Robotics-1-RoboCasa365")
    if BASE_CHECKPOINT != expected or not BASE_CHECKPOINT.is_dir():
        raise RuntimeError(f"Stage1 base checkpoint must be exactly {expected}")
    if common.PROCESSOR.resolve() != expected.resolve():
        raise RuntimeError(f"official RoboCasa base checkpoint mismatch: {common.PROCESSOR}")


def load_rows(manifest: Path) -> dict[str, list[dict]]:
    meta = json.loads(Path(str(manifest).replace(".jsonl", ".meta.json")).read_text())
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if meta.get("manifest_sha256") != digest or tuple(meta.get("task_names", ())) != TASKS:
        raise RuntimeError("spatial6 manifest metadata/hash mismatch")
    rows = {"train": [], "val": []}
    for line in manifest.open():
        row = json.loads(line)
        if row.get("task") in TASKS and row.get("split") in rows:
            rows[row["split"]].append(row)
    if not all(rows.values()):
        raise RuntimeError("spatial6 train/val split is empty")
    train_eps = {(r["task"], r["episode_key"]) for r in rows["train"]}
    val_eps = {(r["task"], r["episode_key"]) for r in rows["val"]}
    if train_eps & val_eps:
        raise RuntimeError("episode leakage detected")
    for task in TASKS:
        eps = sorted({r["episode_key"] for r in rows["val"] if r["task"] == task})
        if len(eps) != 2:
            raise RuntimeError(f"{task} has {len(eps)} val episodes, expected exactly 2")
    return rows


def build_diagnostic_specs(rows: dict[str, list[dict]]) -> list[dict]:
    by_episode = {(r["task"], r["episode_key"], int(r["frame_index"])): r for r in rows["val"]}
    specs = []
    for task in TASKS:
        episodes = sorted({r["episode_key"] for r in rows["val"] if r["task"] == task})
        lengths = {ep: max(int(r["frame_index"]) for r in rows["val"] if r["task"] == task and r["episode_key"] == ep) + 1 for ep in episodes}
        for progress in PROGRESS:
            target_rows = []
            donor_rows = []
            for target_ep, donor_ep in zip(episodes, episodes[::-1]):
                frame = round(progress * (lengths[target_ep] - 1))
                donor_frame = round(progress * (lengths[donor_ep] - 1))
                target = by_episode[(task, target_ep, frame)]
                donor = by_episode[(task, donor_ep, donor_frame)]
                target_rows.append(target)
                donor_rows.append(donor)
                specs.append({"task": task, "progress": progress, "target": target["sample_id"], "target_episode": target_ep, "donor": donor["sample_id"], "donor_episode": donor_ep})
            # Keep the two episode rows together: it gives paired correct/wrong
            # comparisons while covering exactly 2*4*6 fixed samples.
            specs[-2]["pair_id"] = f"{task}@{progress:g}"
            specs[-1]["pair_id"] = f"{task}@{progress:g}"
    return specs


def expected_trainable_names() -> list[str]:
    names = []
    for index in range(36):
        prefix = f"model.dit.layers.{index}.esm_cross_attention"
        for module in ("query_norm", "context_norm"):
            names.extend([f"{prefix}.{module}.weight", f"{prefix}.{module}.bias"])
        for module in ("q_proj", "k_proj", "v_proj", "o_proj"):
            names.append(f"{prefix}.{module}.weight")
    return names


class SpatialCrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_norm = nn.LayerNorm(DIT_HIDDEN)
        self.context_norm = nn.LayerNorm(ESM_HIDDEN)
        self.q_proj = nn.Linear(DIT_HIDDEN, DIT_HIDDEN, bias=False)
        self.k_proj = nn.Linear(ESM_HIDDEN, DIT_HIDDEN, bias=False)
        self.v_proj = nn.Linear(ESM_HIDDEN, DIT_HIDDEN, bias=False)
        self.o_proj = nn.Linear(DIT_HIDDEN, DIT_HIDDEN, bias=False)
        # Keep this fixed scalar outside Module buffers: DeepSpeed's global
        # bf16 cast cannot represent decimal 0.02 exactly.
        self.__dict__["gate"] = torch.tensor(GATE, dtype=torch.float32)
        self.force_identity = False
        self.to(dtype=torch.bfloat16)
        self.gate.fill_(GATE)

    def forward(self, hidden, context):
        if self.force_identity or context is None:
            return hidden
        if context.ndim != 3 or tuple(context.shape[1:]) != (1369, ESM_HIDDEN):
            raise RuntimeError(f"ESM contract must be [B,1369,2048], got {tuple(context.shape)}")
        if hidden.shape[0] % context.shape[0]:
            raise RuntimeError(f"hidden/context batch mismatch: {hidden.shape[0]} vs {context.shape[0]}")
        repeat = hidden.shape[0] // context.shape[0]
        context = self.context_norm(context)
        if repeat > 1:
            # Official XR-1 repeats each raw sample four times.  Keep the
            # ESM K/V tensors at raw batch size and compute each repeat group
            # separately; repeat_interleave here would materialize ~4x the
            # 1369x2048 context on every DiT layer.
            grouped_hidden = hidden.view(context.shape[0], repeat, hidden.shape[1], hidden.shape[2])
            q = self.q_proj(self.query_norm(grouped_hidden)).view(context.shape[0], repeat, hidden.shape[1], -1, HEAD_DIM).transpose(2, 3)
        else:
            q = self.q_proj(self.query_norm(hidden)).view(hidden.shape[0], hidden.shape[1], -1, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(context).view(context.shape[0], context.shape[1], -1, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(context).view(context.shape[0], context.shape[1], -1, HEAD_DIM).transpose(1, 2)
        if repeat > 1:
            delta = torch.stack([
                F.scaled_dot_product_attention(q[:, index], k, v, dropout_p=0.0).transpose(1, 2).reshape(context.shape[0], hidden.shape[1], DIT_HIDDEN)
                for index in range(repeat)
            ], dim=1).reshape_as(hidden)
        else:
            delta = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0).transpose(1, 2).reshape_as(hidden)
        return hidden + self.gate.to(device=delta.device, dtype=delta.dtype) * self.o_proj(delta)


def spatial_layer_forward(layer, hidden_states, past_key_values, position_embeds, timestep, attn_mask):
    shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (layer.adaln_table[None] + timestep).chunk(6, dim=1)
    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states) * (1 + scale_attn) + shift_attn
    if layer.training and torch.is_grad_enabled():
        attn_output = torch.utils.checkpoint.checkpoint(
            lambda x: layer.attn(x, past_key_values, position_embeds, attn_mask),
            hidden_states,
            use_reentrant=False,
        )
    else:
        attn_output = layer.attn(hidden_states, past_key_values, position_embeds, attn_mask)
    hidden_states = residual + gate_attn * attn_output
    tokens = getattr(layer, "_esm_tokens", None)
    hidden_states = layer.esm_cross_attention(hidden_states, tokens)
    residual = hidden_states
    hidden_states = layer.post_layernorm(hidden_states) * (1 + scale_mlp) + shift_mlp
    return residual + gate_mlp * layer.mlp(hidden_states)


def spatial_attention_forward(attention, hidden_states, past_key_values, position_embeds, attn_mask):
    """Official attention math with non-materialized training-repeat cache.

    XR-1's stock attention repeats the frozen VLM KV cache along batch before
    every DiT layer.  This new-script-only bound method computes each repeat
    group against the same cache, which is mathematically identical and keeps
    the official source under ``mibot/**`` untouched.
    """
    from mibot.models.VLA.XR1 import apply_rotary_pos_emb, repeat_kv

    batch_size, seq_len, _ = hidden_states.shape
    cache_batch = past_key_values[0].shape[0]
    if cache_batch == batch_size:
        repeat = 1
    elif batch_size % cache_batch == 0:
        repeat = batch_size // cache_batch
    else:
        raise RuntimeError(f"Cannot align official attention cache {cache_batch} with hidden batch {batch_size}")
    qkv = attention.qkv_proj(hidden_states).view(batch_size, seq_len, 3, attention.num_heads, attention.head_dim)
    query, key, value = qkv.unbind(2)
    query = attention.q_norm(query).transpose(1, 2)
    key = attention.k_norm(key).transpose(1, 2)
    value = value.transpose(1, 2)
    cos, sin = position_embeds
    if cos.ndim == 4:
        cos, sin = cos[0], sin[0]
    query, key = apply_rotary_pos_emb(query, key, cos, sin)
    cache_key, cache_value = past_key_values
    if attention.kv_group != 1:
        cache_key = repeat_kv(cache_key, attention.kv_group)
        cache_value = repeat_kv(cache_value, attention.kv_group)
    if repeat == 1:
        combined_key = torch.cat([cache_key, key], dim=-2)
        combined_value = torch.cat([cache_value, value], dim=-2)
        output = F.scaled_dot_product_attention(query, combined_key, combined_value, attn_mask=attn_mask, dropout_p=0.0)
    else:
        query = query.reshape(cache_batch, repeat, attention.num_heads, seq_len, attention.head_dim)
        key = key.reshape(cache_batch, repeat, attention.num_heads, seq_len, attention.head_dim)
        value = value.reshape(cache_batch, repeat, attention.num_heads, seq_len, attention.head_dim)
        mask = attn_mask.reshape(cache_batch, repeat, *attn_mask.shape[1:])
        outputs = []
        for index in range(repeat):
            combined_key = torch.cat([cache_key, key[:, index]], dim=-2)
            combined_value = torch.cat([cache_value, value[:, index]], dim=-2)
            outputs.append(F.scaled_dot_product_attention(query[:, index], combined_key, combined_value, attn_mask=mask[:, index], dropout_p=0.0))
        output = torch.stack(outputs, dim=1).reshape(batch_size, attention.num_heads, seq_len, attention.head_dim)
    output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, attention.hidden_size)
    return attention.o_proj(output)


def attach_spatial_modules(model):
    if len(model.dit.layers) != 36:
        raise RuntimeError(f"expected 36 DiT layers, found {len(model.dit.layers)}")
    for layer in model.dit.layers:
        layer.esm_cross_attention = SpatialCrossAttention()
        layer._esm_tokens = None
        layer.forward = spatial_layer_forward.__get__(layer, type(layer))
        layer.attn.forward = spatial_attention_forward.__get__(layer.attn, type(layer.attn))


class SpatialDataset(common.RoboCasa20):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        from PIL import Image
        from decord import VideoReader
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


class Stage1Data(LightningDataModule):
    def __init__(self, rows, processor, batch_size, workers, stats_processor):
        super().__init__()
        self.rows, self.processor, self.batch_size, self.workers = rows, processor, batch_size, workers
        self.stats_processor = stats_processor
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
        return self._loader(self.train_set, DistributedSampler(self.train_set, shuffle=True, seed=common.SEED), drop_last=True)

    def val_dataloader(self):
        self.setup("fit")
        return self._loader(self.val_set, DistributedSampler(self.val_set, shuffle=False, seed=common.SEED), drop_last=False)


class FrozenRightESM(nn.Module):
    def __init__(self):
        super().__init__()
        if not ESM_CKPT.is_file() or not FALCON.is_dir():
            raise FileNotFoundError(f"missing ESM checkpoint or FALCON source: {ESM_CKPT}, {FALCON}")
        if str(FALCON) not in sys.path:
            sys.path.insert(0, str(FALCON))
        # FALCON's constructor supports this local override; keep DINOv2
        # entirely offline and do not alter the official FALCON checkout.
        os.environ["FALCON_DINOV2_REPO"] = "/data-cfs/data3/models/dinov2"
        os.environ["TORCH_HOME"] = "/data-cfs/data3/models/torch"
        from falcon.model.policy_head.esm_utils.vggt.models.vggt_camera_replace_depth import VGGT_Camera_Replace_Depth
        ckpt = torch.load(str(ESM_CKPT), map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        self.net = VGGT_Camera_Replace_Depth()
        missing, unexpected = self.net.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"ESM checkpoint mismatch: missing={missing[:4]}, unexpected={unexpected[:4]}")
        self.net.requires_grad_(False).eval()
        del ckpt, state

    @torch.no_grad()
    def forward(self, image):
        if image.ndim != 4 or tuple(image.shape[1:]) != (3, 518, 518):
            raise RuntimeError(f"ESM input must be [B,3,518,518], got {tuple(image.shape)}")
        dtype = next(self.net.parameters()).dtype
        _, aggregated = self.net.inference(images=image.to(dtype=dtype).unsqueeze(1), camera_gt_pt=0.0, depth_gt_pt=0.0)
        feature = aggregated[-1]
        patch_start = int(getattr(self.net.aggregator, "patch_start_idx", 5))
        if patch_start != 5 or feature.ndim != 4 or feature.shape[1] != 1 or feature.shape[-1] != ESM_HIDDEN:
            raise RuntimeError(f"ESM final/stage23 contract failed: patch_start={patch_start}, feature={tuple(feature.shape)}")
        tokens = feature[:, 0, 5:].contiguous()
        if tuple(tokens.shape[1:]) != (1369, ESM_HIDDEN) or not torch.isfinite(tokens).all():
            raise RuntimeError(f"ESM token contract failed: {tuple(tokens.shape)}")
        return tokens


class Stage1Runner(common.Official20Runner):
    def __init__(self, *args, diagnostic_specs, output_dir, **kwargs):
        assert_base_checkpoint()
        super().__init__(*args, **kwargs)
        self.__dict__["_esm"] = FrozenRightESM()
        self._esm_device = None
        self.diagnostic_specs = diagnostic_specs
        self.output_dir = Path(output_dir)
        self._diag_pairs = None
        self._diag_token_cache = None
        self._acceptance_done = False

    def configure_model(self):
        super().configure_model()
        attach_spatial_modules(self.model)
        self.model.requires_grad_(False)
        for layer in self.model.dit.layers:
            layer.esm_cross_attention.requires_grad_(True)
            layer.esm_cross_attention.gate.requires_grad_(False)
        if any(p.requires_grad for p in self._esm.parameters()):
            raise RuntimeError("ESM encoder must remain frozen")
        if int(self.model.training_repeat) != TRAINING_REPEAT:
            raise RuntimeError(f"official training_repeat must be {TRAINING_REPEAT}, got {self.model.training_repeat}")
        actual = [n for n, p in self.named_parameters() if p.requires_grad]
        expected = expected_trainable_names()
        if actual != expected:
            raise RuntimeError(f"trainable parameter contract mismatch: actual={actual[:8]}... expected={expected[:8]}...")
        print(json.dumps({"trainable_parameter_names": actual, "trainable_parameter_count": sum(self.get_parameter(n).numel() for n in actual), "gate": GATE, "gate_trainable": False}, indent=2), flush=True)

    def configure_optimizers(self):
        actual = [n for n, p in self.named_parameters() if p.requires_grad]
        if actual != expected_trainable_names():
            raise RuntimeError("optimizer parameter list is not exactly the six spatial modules per layer")
        config = super().configure_optimizers()
        optimizer = config["optimizer"]
        expected_ids = {id(self.get_parameter(name)) for name in expected_trainable_names()}
        actual_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
        if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids or len(actual_ids) != len(expected_ids):
            raise RuntimeError("optimizer param_groups are not exactly the 36 spatial cross-attention parameter set")
        if self.global_rank == 0:
            print(json.dumps({"optimizer_param_groups": [len(group["params"]) for group in optimizer.param_groups], "optimizer_parameter_count": len(actual_ids)}), flush=True)
        return config

    def assert_gates(self):
        for layer in self.model.dit.layers:
            gate = layer.esm_cross_attention.gate
            if abs(float(gate.item()) - GATE) > 1e-8 or gate.requires_grad:
                raise RuntimeError(f"gate contract violated: {gate.item()}, requires_grad={gate.requires_grad}")

    def on_train_start(self):
        common.memory_policy(self.target_gb, self.limit_gb)
        self.assert_gates()
        if int(self.model.training_repeat) != TRAINING_REPEAT:
            raise RuntimeError(f"official training_repeat must be {TRAINING_REPEAT}, got {self.model.training_repeat}")
        if self.global_rank == 0:
            print(json.dumps({"stage1_start": True, "per_gpu_raw_batch": PER_GPU_BATCH, "world_size": WORLD_SIZE, "global_raw_batch": PER_GPU_BATCH * WORLD_SIZE, "training_repeat": self.model.training_repeat, "max_steps": self.max_steps, "val_every": VAL_EVERY, "checkpoint_every": CHECKPOINT_EVERY, "diagnostic_every": DIAGNOSTIC_EVERY}), flush=True)
        self._run_diagnostics(force=True)

    def _esm_tokens(self, image):
        device = next(self.model.parameters()).device
        if self._esm_device != device:
            self._esm.to(device, dtype=torch.bfloat16)
            self._esm_device = device
        return self._esm(image.to(device, dtype=torch.bfloat16))

    @staticmethod
    def _move(value, device):
        if isinstance(value, dict):
            return {k: Stage1Runner._move(v, device) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(Stage1Runner._move(v, device) for v in value)
        return value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value

    def _forward_variant(self, batch, mode, wrong_image=None, cached_tokens=None):
        work = dict(batch)
        image = work.pop("esm_image", None)
        if mode == "noesm":
            for layer in self.model.dit.layers:
                layer.esm_cross_attention.force_identity = True
            tokens = None
        else:
            for layer in self.model.dit.layers:
                layer.esm_cross_attention.force_identity = False
            tokens = cached_tokens if cached_tokens is not None else self._esm_tokens(wrong_image if mode == "wrong" else image)
        for layer in self.model.dit.layers:
            layer._esm_tokens = tokens
        try:
            return self.model(work, return_loss=True)
        finally:
            for layer in self.model.dit.layers:
                layer._esm_tokens = None
                layer.esm_cross_attention.force_identity = False

    def _capture_rng(self):
        return random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()

    def _restore_rng(self, state):
        random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2]); torch.cuda.set_rng_state_all(state[3])

    def on_train_batch_start(self, batch, batch_idx):
        if self._acceptance_done:
            return
        state = self._capture_rng()
        was_training = self.model.training
        spatial = [p for n, p in self.named_parameters() if "esm_cross_attention" in n and p.requires_grad]
        try:
            self.model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                result = self._forward_variant(batch, "correct")
                # Route the probe through Lightning's strategy/precision
                # hooks so DeepSpeed handles scaling and partitioned grads.
                # ``manual_backward`` is not valid here because this runner
                # intentionally uses Lightning automatic optimization.
                self.trainer.strategy.backward(result["loss"], None)
            if not result["loss"].requires_grad:
                raise RuntimeError("spatial acceptance loss is not differentiable")
            visible_grads = [p.grad for p in spatial if p.grad is not None]
            if any(not torch.isfinite(grad).all() for grad in visible_grads):
                raise RuntimeError("spatial acceptance test produced a non-finite visible gradient")
            if visible_grads and not any(grad.detach().float().abs().sum().item() > 0 for grad in visible_grads):
                raise RuntimeError("spatial acceptance test produced only all-zero visible gradients")
            frozen = [p for n, p in self.model.named_parameters() if "esm_cross_attention" not in n]
            if any(p.grad is not None for p in frozen) or any(p.grad is not None for p in self._esm.parameters()):
                raise RuntimeError("frozen backbone or ESM received a gradient")
            self.assert_gates()
            self._acceptance_done = True
            if self.global_rank == 0:
                print(json.dumps({"acceptance_test": "passed", "spatial_trainable": len(spatial), "visible_spatial_grads": len(visible_grads), "deepspeed_partition_aware": True, "frozen_backbone_grad_none": True, "esm_grad_none": True, "gate": GATE, "backward": "strategy_backward_before_training_step"}), flush=True)
        finally:
            self.model.zero_grad(set_to_none=True)
            self._esm.zero_grad(set_to_none=True)
            self._restore_rng(state)
            if not was_training:
                self.model.eval()

    def training_step(self, batch, batch_idx):
        result = self._forward_variant(batch, "correct")
        for name, value in result.items():
            self.log(f"train/{name}", value.detach(), sync_dist=True)
        return result["loss"]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.global_step > 0 and self.global_step % DIAGNOSTIC_EVERY == 0:
            self._run_diagnostics()

    def validation_step(self, batch, batch_idx):
        was_training = self.model.training
        self.model.train()
        with torch.no_grad():
            result = self._forward_variant(batch, "correct")
        if not was_training:
            self.model.eval()
        for name, value in result.items():
            self.log(f"val/{name}", value.detach(), prog_bar=name == "loss", sync_dist=True)
        return result["loss"]

    def _prepare_diag_pairs(self):
        if self._diag_pairs is not None:
            return
        processor = common.build_official_qwen_processor()
        stats = common.build_action_stats_processor()
        target_rows = []
        donor_rows = []
        for spec in self.diagnostic_specs:
            target_rows.append(next(r for r in self._diag_val_rows if r["sample_id"] == spec["target"]))
            donor_rows.append(next(r for r in self._diag_val_rows if r["sample_id"] == spec["donor"]))
        target_ds = common.OfficialSampleDataset(SpatialDataset(target_rows), stats)
        donor_ds = SpatialDataset(donor_rows)
        target_items = [target_ds[i] for i in range(len(target_ds))]
        donor_items = [donor_ds[i] for i in range(len(donor_ds))]
        collate = common.LocalOfficialCollate(processor)
        self._diag_pairs = []
        for index, spec in enumerate(self.diagnostic_specs):
            batch = collate([target_items[index]])
            wrong_image = donor_items[index]["esm_image"].unsqueeze(0)
            self._diag_pairs.append((spec, batch, wrong_image))
        if len(self._diag_pairs) != 48:
            raise RuntimeError(f"diagnostic must contain 48 individual samples, got {len(self._diag_pairs)}")

    def _cache_diagnostic_tokens(self):
        if self._diag_token_cache is not None:
            return
        device = next(self.model.parameters()).device
        cache = []
        with torch.no_grad():
            for spec, cpu_batch, cpu_wrong in self._diag_pairs:
                correct_image = cpu_batch["esm_image"].to(device, non_blocking=True)
                wrong_image = cpu_wrong.to(device, non_blocking=True)
                # ESM is frozen and evaluated before policy RNG capture.  The
                # cached CPU copies make all three policy variants use the
                # identical precomputed context tokens.
                correct_tokens = self._esm_tokens(correct_image).detach().cpu()
                wrong_tokens = self._esm_tokens(wrong_image).detach().cpu()
                cache.append((correct_tokens, wrong_tokens))
        self._diag_token_cache = cache

    def _run_diagnostics(self, force=False):
        if not force and (self.global_step <= 0 or self.global_step % DIAGNOSTIC_EVERY):
            return
        self._prepare_diag_pairs()
        self._cache_diagnostic_tokens()
        outer_rng = self._capture_rng()
        was_training = self.model.training
        self.model.train()
        records = []
        try:
            device = next(self.model.parameters()).device
            for index, (spec, cpu_batch, cpu_wrong) in enumerate(self._diag_pairs):
                batch = self._move(cpu_batch, next(self.model.parameters()).device)
                correct_tokens, wrong_tokens = self._diag_token_cache[index]
                correct_tokens = correct_tokens.to(device, non_blocking=True)
                wrong_tokens = wrong_tokens.to(device, non_blocking=True)
                rng = self._capture_rng()
                outputs = {}
                with torch.no_grad():
                    for mode, tokens in (("correct", correct_tokens), ("wrong", wrong_tokens), ("noesm", None)):
                        self._restore_rng(rng)
                        out = self._forward_variant(batch, mode, cached_tokens=tokens)
                        outputs[mode] = {key: float(value.detach().cpu()) for key, value in out.items()}
                records.append({"sample_id": spec["target"], "task": spec["task"], "progress": spec["progress"], "count": 1, **outputs})
        finally:
            self._restore_rng(outer_rng)
            if not was_training:
                self.model.eval()
        if self.global_rank == 0:
            self._write_diagnostics(records)

    @staticmethod
    def _summary(records):
        total = sum(item["count"] for item in records)
        def avg(mode, key):
            return sum(item[mode][key] * item["count"] for item in records) / total
        result = {mode: {key: avg(mode, key) for key in ("loss", "loss_mse", "loss_freq")} for mode in ("correct", "wrong", "noesm")}
        wrong_deltas = [item["wrong"]["loss"] - item["correct"]["loss"] for item in records for _ in range(item["count"])]
        noesm_deltas = [item["noesm"]["loss"] - item["correct"]["loss"] for item in records for _ in range(item["count"])]

        def paired_stats(deltas):
            n = len(deltas)
            mean = sum(deltas) / n
            if n > 1:
                variance = sum((value - mean) ** 2 for value in deltas) / (n - 1)
                margin = 1.96 * (variance / n) ** 0.5
            else:
                margin = 0.0
            return {"mean": mean, "low": mean - margin, "high": mean + margin, "n": n}

        result["delta_wrong"] = paired_stats(wrong_deltas)["mean"]
        result["delta_noesm"] = paired_stats(noesm_deltas)["mean"]
        result["delta_wrong_95ci"] = paired_stats(wrong_deltas)
        result["delta_noesm_95ci"] = paired_stats(noesm_deltas)
        result["fraction_wrong_gt_correct"] = sum(delta > 0 for delta in wrong_deltas) / len(wrong_deltas)
        result["fraction_noesm_gt_correct"] = sum(delta > 0 for delta in noesm_deltas) / len(noesm_deltas)
        result["count"] = total
        return result

    def _write_diagnostics(self, records):
        step = int(self.global_step)
        by_task = {task: self._summary([r for r in records if r["task"] == task]) for task in TASKS}
        by_progress = {f"{int(progress * 100)}%": self._summary([r for r in records if r["progress"] == progress]) for progress in PROGRESS}
        payload = {"step": step, "sample_count": 48, "comparison_unit": "individual sample-level paired correct/wrong/noESM evaluation", "paired_ci": "normal-approximation 95% CI", "overall": self._summary(records), "by_task": by_task, "by_progress": by_progress}
        path = self.output_dir / "diagnostics"
        path.mkdir(parents=True, exist_ok=True)
        (path / f"step_{step:06d}.json").write_text(json.dumps(payload, indent=2) + "\n")
        with (path / "diagnostics.jsonl").open("a") as handle:
            handle.write(json.dumps(payload) + "\n")
        csv_path = path / "diagnostics.csv"
        fields = ["step", "scope", "correct_loss", "wrong_loss", "noesm_loss", "correct_loss_mse", "wrong_loss_mse", "noesm_loss_mse", "correct_loss_freq", "wrong_loss_freq", "noesm_loss_freq", "delta_wrong", "delta_wrong_ci_low", "delta_wrong_ci_high", "delta_noesm", "delta_noesm_ci_low", "delta_noesm_ci_high", "fraction_wrong_gt_correct", "fraction_noesm_gt_correct", "count"]
        with csv_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if csv_path.stat().st_size == 0:
                writer.writeheader()
            for scope, summary in [("overall", payload["overall"]), *[(f"task/{k}", v) for k, v in by_task.items()], *[(f"progress/{k}", v) for k, v in by_progress.items()]]:
                writer.writerow({"step": step, "scope": scope, "correct_loss": summary["correct"]["loss"], "wrong_loss": summary["wrong"]["loss"], "noesm_loss": summary["noesm"]["loss"], "correct_loss_mse": summary["correct"]["loss_mse"], "wrong_loss_mse": summary["wrong"]["loss_mse"], "noesm_loss_freq": summary["noesm"]["loss_freq"], "correct_loss_freq": summary["correct"]["loss_freq"], "wrong_loss_freq": summary["wrong"]["loss_freq"], "delta_wrong": summary["delta_wrong"], "delta_wrong_ci_low": summary["delta_wrong_95ci"]["low"], "delta_wrong_ci_high": summary["delta_wrong_95ci"]["high"], "delta_noesm": summary["delta_noesm"], "delta_noesm_ci_low": summary["delta_noesm_95ci"]["low"], "delta_noesm_ci_high": summary["delta_noesm_95ci"]["high"], "fraction_wrong_gt_correct": summary["fraction_wrong_gt_correct"], "fraction_noesm_gt_correct": summary["fraction_noesm_gt_correct"], "count": summary["count"]})
        if self.logger is not None and hasattr(self.logger, "experiment"):
            for scope, summary in [("overall", payload["overall"]), *[(f"task/{k}", v) for k, v in by_task.items()], *[(f"progress/{k}", v) for k, v in by_progress.items()]]:
                for mode in ("correct", "wrong", "noesm"):
                    for key in ("loss", "loss_mse", "loss_freq"):
                        self.logger.experiment.add_scalar(f"diagnostic/{scope}/{mode}/{key}", summary[mode][key], step)
                for key in ("delta_wrong", "delta_noesm", "fraction_wrong_gt_correct", "fraction_noesm_gt_correct"):
                    self.logger.experiment.add_scalar(f"diagnostic/{scope}/{key}", summary[key], step)
        print(json.dumps({"diagnostic_step": step, "overall": payload["overall"]}), flush=True)

    def on_validation_end(self):
        self.assert_gates()


def write_contract(output, manifest, meta, specs, args):
    names = expected_trainable_names()
    contract = {
        "base_checkpoint": str(BASE_CHECKPOINT),
        "xr1_training_state_checkpoint": str(common.MODEL_CKPT),
        "base_checkpoint_hard_assert": str(BASE_CHECKPOINT),
        "base_policy_reference": str(BASE_CHECKPOINT),
        "esm_checkpoint": str(ESM_CKPT),
        "task_names": list(TASKS),
        "manifest_path": str(manifest),
        "manifest_sha256": meta["manifest_sha256"],
        "train_episode_counts": {task: len(meta["per_task"][task]["train_episode_ids"]) for task in TASKS},
        "val_episode_ids": {task: meta["per_task"][task]["val_episode_ids"] for task in TASKS},
        "val_episode_keys": {task: meta["per_task"][task]["val_episode_keys"] for task in TASKS},
        "per_gpu_batch": args.batch_size,
        "world_size": args.devices,
        "global_batch": args.batch_size * args.devices,
        "training_repeat": TRAINING_REPEAT,
        "gate": GATE,
        "gate_trainable": False,
        "trainable_parameter_names": names,
        "trainable_parameter_count": 226713600,
        "max_steps": args.max_steps,
        "scheduler_steps": args.max_steps,
        "val_every": VAL_EVERY,
        "checkpoint_every": CHECKPOINT_EVERY,
        "diagnostic_every": DIAGNOSTIC_EVERY,
        "contract": {"views": ["left", "right", "wrist"], "visual_history": 4, "state_history": 4, "interval": 2, "crop_ratio": 0.95, "action_horizon": 16, "esm_view": "right", "esm_current_frame_only": True, "esm_input": [3, 518, 518], "esm_feature": "final/stage23", "esm_tokens": [1369, 2048], "removed_tokens": {"camera": 1, "register": 4}, "pool": False, "interpolate": False, "spatial_layers": 36, "objective": "official flow matching + frequency + choice + score", "extra_esm_loss": False},
        "diagnostic_sample_count": 48,
        "diagnostic_sample_ids": [s["target"] for s in specs],
    }
    (output / "training_contract.json").write_text(json.dumps(contract, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=PER_GPU_BATCH)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--target-gpu-gb", type=float, default=70.0)
    parser.add_argument("--limit-gpu-gb", type=float, default=70.0)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size != PER_GPU_BATCH or args.devices != WORLD_SIZE or args.max_steps != 20000:
        raise ValueError(f"Stage1 contract requires --batch-size {PER_GPU_BATCH} --devices {WORLD_SIZE} --max-steps 20000")
    assert_base_checkpoint()
    random.seed(common.SEED); np.random.seed(common.SEED); torch.manual_seed(common.SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("manifests", "checkpoints", "tensorboard", "diagnostics", "logs"):
        (args.output_dir / name).mkdir(exist_ok=True)
    rows = load_rows(args.manifest)
    meta = json.loads(Path(str(args.manifest).replace(".jsonl", ".meta.json")).read_text())
    specs = build_diagnostic_specs(rows)
    (args.output_dir / "diagnostics" / "diagnostic_sample_ids.json").write_text(json.dumps([s["target"] for s in specs], indent=2) + "\n")
    (args.output_dir / "diagnostics" / "wrong_donor_mapping.json").write_text(json.dumps(specs, indent=2) + "\n")
    write_contract(args.output_dir, args.manifest, meta, specs, args)
    if args.audit_only:
        print(json.dumps({"audit": "passed", "train_rows": len(rows["train"]), "val_rows": len(rows["val"]), "diagnostic_samples": len(specs), "manifest_sha256": meta["manifest_sha256"]}, indent=2))
        return
    stats = common.build_action_stats_processor()
    processor = common.build_official_qwen_processor()
    data = Stage1Data(rows, processor, args.batch_size, args.workers, stats)
    runner = Stage1Runner(common.MODEL_CKPT, args.max_steps, args.target_gpu_gb, args.limit_gpu_gb, diagnostic_specs=specs, output_dir=args.output_dir)
    runner._diag_val_rows = rows["val"]
    checkpoint = ModelCheckpoint(dirpath=str(args.output_dir / "checkpoints"), every_n_train_steps=CHECKPOINT_EVERY, save_top_k=-1, save_last=True, enable_version_counter=False)
    logger = TensorBoardLogger(save_dir=str(args.output_dir / "tensorboard"), name="stage1_spatial6", version="")
    trainer = Trainer(accelerator="cuda", devices=args.devices, num_nodes=1, precision="bf16-mixed", strategy=__import__("lightning.pytorch.strategies", fromlist=["DeepSpeedStrategy"]).DeepSpeedStrategy(), max_steps=args.max_steps, default_root_dir=str(args.output_dir), accumulate_grad_batches=1, gradient_clip_val=1.0, log_every_n_steps=10, val_check_interval=VAL_EVERY, check_val_every_n_epoch=None, limit_val_batches=1.0, num_sanity_val_steps=0, callbacks=[checkpoint], logger=logger, enable_checkpointing=True)
    trainer.fit(runner, datamodule=data)


if __name__ == "__main__":
    main()
