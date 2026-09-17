#!/usr/bin/env python3
"""XR-1 RoboCasa365 Spatial Forcing ablation.

The ESM branch in this file is a frozen teacher only.  It never enters the
XR-1/DiT computation graph and is used only to form an auxiliary cosine loss
against the selected intermediate Qwen visual tokens.

The runner deliberately reuses the official XR-1 data adapter and action-loss
path from ``train_xr1_baseline_official20``.  The only trainable policy
parameters are explicit LoRA adapters in Qwen's text transformer and the 36
DiT blocks; the alignment projector is trainable only for Spatial Forcing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from lightning.pytorch.callbacks import Callback

import train_xr1_baseline_official20 as common

TASKS = (
    "CloseBlenderLid", "CoffeeSetupMug", "TurnOffStove", "TurnOnMicrowave",
    "SlideDishwasherRack", "PickPlaceCounterToCabinet",
)
SEED = 42
VAL_SEED = 12345
TRAINING_REPEAT = 4
TOTAL_DIT_LAYERS = 36
ESM_HIDDEN = 2048
QWEN_LORA_SUFFIXES = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)
DIT_LORA_SUFFIXES = (
    "attn.qkv_proj", "attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)
EXPECTED_QWEN_LORA_TARGETS = TOTAL_DIT_LAYERS * len(QWEN_LORA_SUFFIXES)
EXPECTED_DIT_LORA_TARGETS = TOTAL_DIT_LAYERS * len(DIT_LORA_SUFFIXES)
LORA_RANK = 16
LORA_ALPHA = 32.0
LORA_DROPOUT = 0.05
LR = 2e-5
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 500
WARMUP_LR_START = 5e-7
MIN_LR = 5e-6


def target_lr(step: int, max_steps: int = 20000) -> float:
    step = int(step)
    if step < WARMUP_STEPS:
        return min(LR, WARMUP_LR_START + (LR - WARMUP_LR_START) * step / max(WARMUP_STEPS, 1))
    progress = float(step - WARMUP_STEPS) / float(max(1, max_steps - WARMUP_STEPS))
    return (LR - MIN_LR) * 0.5 * (1.0 + math.cos(math.pi * progress)) + MIN_LR


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def seed_validation_batch(batch_idx: int) -> int:
    seed = VAL_SEED + int(batch_idx)
    seed_all(seed)
    return seed


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_six_rows(manifest: Path):
    rows = {"train": [], "val": []}
    for line in manifest.open():
        row = json.loads(line)
        if row.get("task") in TASKS and row.get("split") in rows:
            rows[row["split"]].append(row)
    if not rows["train"] or not rows["val"]:
        raise RuntimeError(f"empty six-task manifest: train={len(rows['train'])}, val={len(rows['val'])}")
    train_eps = {(r["task"], r.get("episode_key", r.get("episode_id"))) for r in rows["train"]}
    val_eps = {(r["task"], r.get("episode_key", r.get("episode_id"))) for r in rows["val"]}
    if train_eps & val_eps:
        raise RuntimeError("train/val episode leakage")
    for task in TASKS:
        val = [r for r in rows["val"] if r["task"] == task]
        if len({r.get("episode_key", r.get("episode_id")) for r in val}) != 2:
            raise RuntimeError(f"{task}: expected exactly two validation trajectories")
    return rows


class LoRALinear(nn.Module):
    """Frozen Linear plus the standard zero-initialized LoRA residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Linear(base.in_features, rank, bias=False,
                                device=base.weight.device, dtype=base.weight.dtype)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False,
                                device=base.weight.device, dtype=base.weight.dtype)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.base.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.dropout(self.lora_A(x))) * self.scaling


def discover_linear_targets(module: nn.Module, suffixes):
    names = [name for name, child in module.named_modules()
             if isinstance(child, nn.Linear) and any(name.endswith(s) for s in suffixes)]
    if not names:
        raise RuntimeError(f"no LoRA targets found for suffixes={suffixes}")
    return sorted(names)


def inject_lora(module: nn.Module, names, rank=LORA_RANK, alpha=LORA_ALPHA, dropout=LORA_DROPOUT):
    modules = dict(module.named_modules())
    for full_name in names:
        base = modules.get(full_name)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRA target is not Linear: {full_name} ({type(base)})")
        parent_name, child_name = full_name.rsplit(".", 1)
        setattr(modules[parent_name], child_name, LoRALinear(base, rank, alpha, dropout))


def lora_parameters(module: nn.Module):
    return [p for n, p in module.named_parameters() if "lora_" in n and p.requires_grad]


def lora_state(module: nn.Module):
    return {n: p.detach().cpu() for n, p in module.named_parameters() if "lora_" in n}


class FrozenRightESM(nn.Module):
    def __init__(self, checkpoint: Path, falcon_root: Path):
        super().__init__()
        self.checkpoint = Path(checkpoint)
        self.falcon_root = Path(falcon_root)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        if not self.falcon_root.is_dir():
            raise FileNotFoundError(self.falcon_root)
        sys.path.insert(0, str(self.falcon_root))
        from falcon.model.policy_head.esm_utils.vggt.models.vggt_camera_replace_depth import VGGT_Camera_Replace_Depth
        state = torch.load(str(self.checkpoint), map_location="cpu", weights_only=False)
        state = state.get("model", state.get("state_dict", state))
        self.net = VGGT_Camera_Replace_Depth()
        missing, unexpected = self.net.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"ESM checkpoint mismatch: missing={missing[:4]}, unexpected={unexpected[:4]}")
        self.net.requires_grad_(False).eval()
        del state

    @torch.no_grad()
    def forward(self, image):
        if image.ndim != 4 or tuple(image.shape[1:]) != (3, 518, 518):
            raise RuntimeError(f"ESM input must be [B,3,518,518], got {tuple(image.shape)}")
        dtype = next(self.net.parameters()).dtype
        _, aggregated = self.net.inference(
            images=image.to(dtype=dtype).unsqueeze(1), camera_gt_pt=0.0, depth_gt_pt=0.0
        )
        feature = aggregated[-1]
        if feature.ndim != 4 or feature.shape[1] != 1 or feature.shape[-1] != ESM_HIDDEN:
            raise RuntimeError(f"ESM final/stage23 contract failed: {tuple(feature.shape)}")
        patch_start = int(getattr(self.net.aggregator, "patch_start_idx", 5))
        if patch_start != 5:
            raise RuntimeError(f"ESM special-token contract failed: expected camera+4 registers=5, got {patch_start}")
        tokens = feature[:, 0, patch_start:].contiguous()
        if tuple(tokens.shape[1:]) != (1369, ESM_HIDDEN):
            raise RuntimeError(f"ESM token contract failed: {tuple(tokens.shape)}")
        return tokens


class AlignmentProjector(nn.Module):
    """Official-compatible dimension projector, with explicit v2 init."""

    def __init__(self, qwen_dim: int, use_vlm_norm: bool = False):
        super().__init__()
        self.vlm_norm = nn.LayerNorm(qwen_dim) if use_vlm_norm else None
        self.fc1 = nn.Linear(qwen_dim, 4096, bias=True)
        self.fc2 = nn.Linear(4096, ESM_HIDDEN, bias=True)
        self.act = nn.GELU()
        for module in (self.fc1, self.fc2):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        if self.vlm_norm is not None:
            x = self.vlm_norm(x)
        return self.fc2(self.act(self.fc1(x)))


def _grid_count(grid, merge_size):
    t, h, w = [int(x) for x in grid]
    if h % merge_size or w % merge_size:
        raise RuntimeError(f"video_grid_thw is not merge-aligned: {grid.tolist()}")
    return t * (h // merge_size) * (w // merge_size)


def resolve_current_view_tokens(batch, processor, print_once_state=None):
    """Resolve current-only LEFT/RIGHT/WRIST token blocks from packed Qwen input."""
    ids = batch["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise RuntimeError(f"packed input_ids must be [1,total], got {tuple(ids.shape)}")
    segments = batch.get("segments")
    grids = batch.get("video_grid_thw", batch.get("image_grid_thw"))
    if segments is None or grids is None:
        raise RuntimeError("spatial segments and video_grid_thw are required")
    video_token_id = int(getattr(processor, "video_token_id", 151656))
    merge = int(getattr(getattr(processor, "image_processor", None), "merge_size", 2))
    temporal_patch_size = int(getattr(getattr(processor, "video_processor", None), "temporal_patch_size", -1))
    if temporal_patch_size != 2:
        raise RuntimeError(f"unexpected runtime temporal_patch_size={temporal_patch_size}; v2 requires 2")
    positions = (ids[0] == video_token_id).nonzero(as_tuple=False).flatten()
    if grids.ndim != 2 or grids.shape[1] != 3:
        raise RuntimeError(f"video_grid_thw must be [N,3], got {tuple(grids.shape)}")
    grid_cursor, results, debug = 0, [], []
    for sample_index, (start, end) in enumerate(segments.tolist()):
        local = positions[(positions >= start) & (positions < end)] - start
        local_grids, expected = [], 0
        while grid_cursor < grids.shape[0] and expected < local.numel():
            grid = grids[grid_cursor]
            count = _grid_count(grid, merge)
            local_grids.append((grid, count))
            expected += count
            grid_cursor += 1
        if expected != local.numel() or len(local_grids) != 3:
            raise RuntimeError(f"sample {sample_index}: view grid/token mismatch grids={len(local_grids)} tokens={local.numel()} expected={expected}")
        starts, offset, labels = [], 0, ("left camera", "right camera", "wrist camera")
        decoded = []
        view_tokens = []
        for view_index, ((grid, count), label) in enumerate(zip(local_grids, labels)):
            if int(grid[0]) != 1:
                raise RuntimeError(f"sample {sample_index} view {view_index} is not current-only: grid={grid.tolist()}")
            block_count = count
            block = local[offset:offset + block_count]
            if block.numel() != block_count or (block_count > 1 and not torch.all(block[1:] - block[:-1] == 1)):
                raise RuntimeError(f"sample {sample_index} view {view_index} token block is not contiguous")
            global_block = block + int(start)
            if view_tokens and int(global_block[0]) <= int(view_tokens[-1][-1]):
                raise RuntimeError("camera token blocks overlap")
            starts.append(int(global_block[0]))
            view_tokens.append(global_block)
            label_text = processor.tokenizer.decode(ids[0, int(start):int(start) + int(block[0])].tolist(), skip_special_tokens=False).lower()
            decoded.append(label_text[-160:])
            if label not in label_text:
                raise RuntimeError(f"sample {sample_index}: expected {label!r}, decoded tail={label_text[-160:]!r}")
            offset += block_count
        if offset != local.numel() or not (view_tokens[0][-1] < view_tokens[1][0] < view_tokens[1][-1] < view_tokens[2][0]):
            raise RuntimeError(f"sample {sample_index}: camera token ranges are invalid")
        results.append(view_tokens)
        debug.append({
            "sample": sample_index,
            "video_grid_thw": [g[0].tolist() for g in local_grids],
            "video_token_id": video_token_id,
            "temporal_patch_size": temporal_patch_size,
            "merge_size": merge,
            "view_grid": [int(local_grids[0][0][1]) // merge, int(local_grids[0][0][2]) // merge],
            "view_grids": [[int(grid[1]) // merge, int(grid[2]) // merge] for grid, _ in local_grids],
            "view_token_ranges": [[int(v[0]), int(v[-1]) + 1] for v in view_tokens],
            "view_token_counts": [int(v.numel()) for v in view_tokens],
            "view_order": ["LEFT", "RIGHT", "WRIST"],
            "decoded_view_label_tails": decoded,
        })
    if grid_cursor != grids.shape[0]:
        raise RuntimeError(f"unused video_grid_thw rows: consumed={grid_cursor}, total={grids.shape[0]}")
    if print_once_state is not None and not print_once_state.get("printed"):
        print(json.dumps({"selected_alignment_layer": print_once_state.get("layer"), "token_mapping": debug}), flush=True)
        print_once_state["printed"] = True
    return results, debug


def warp_full_teacher_to_crop(teacher, hq, wq, crop_ratio=0.95):
    """Sample full-FOV 37x37 teacher at the official center-crop coordinates."""
    if teacher.ndim != 4 or teacher.shape[2:] != (1369, ESM_HIDDEN):
        raise RuntimeError(f"teacher must be [B,3,1369,2048], got {tuple(teacher.shape)}")
    b = teacher.shape[0]
    feature = teacher.reshape(b * 3, 37, 37, ESM_HIDDEN).permute(0, 3, 1, 2).contiguous()
    yy = (torch.arange(hq, device=teacher.device, dtype=teacher.dtype) + 0.5) / hq
    xx = (torch.arange(wq, device=teacher.device, dtype=teacher.dtype) + 0.5) / wq
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    left = (1.0 - crop_ratio) / 2.0
    full_x = left + crop_ratio * grid_x
    full_y = left + crop_ratio * grid_y
    sample_grid = torch.stack((2.0 * full_x - 1.0, 2.0 * full_y - 1.0), dim=-1)
    warped = F.grid_sample(feature, sample_grid[None].expand(b * 3, -1, -1, -1), mode="bilinear", padding_mode="border", align_corners=False)
    return warped.permute(0, 2, 3, 1).reshape(b, 3, hq * wq, ESM_HIDDEN).contiguous()


class SixTaskData(common.Official20Data):
    """One shared DataModule; SF receives raw current RGB as an extra payload."""
    def __init__(self, rows, processor, batch_size, workers, include_esm, falcon_root,
                 gpu_prefetch=True, stats_processor=None):
        super().__init__(rows, processor, batch_size, workers, gpu_prefetch, stats_processor)
        self.include_esm = bool(include_esm)
        self.falcon_root = Path(falcon_root)

    def setup(self, stage=None):
        if stage not in (None, "fit"):
            return
        if self.train_set is None:
            self.train_set = common.OfficialSampleDataset(common.RoboCasa20(self.rows["train"]), self.stats_processor)
        if self.val_set is None:
            self.val_set = common.OfficialSampleDataset(common.RoboCasa20(self.rows["val"]), self.stats_processor)


class ForcingRunner(common.Official20Runner):
    def __init__(self, args, processor, sf: bool):
        super().__init__(common.MODEL_CKPT, args.max_steps, args.target_gpu_gb, args.limit_gpu_gb)
        self.args = args
        self.processor = processor
        self.sf = sf
        self.layer_number = args.alignment_layer if sf else None
        self.alpha = float(args.alpha if sf else 0.0)
        self.__dict__["_esm"] = None
        self._sf_hook = None
        self._sf_hidden = None
        self._esm_device = None
        self._mapping_print = {"printed": False, "layer": self.layer_number}
        self._spatial_hidden = None
        self._teacher_printed = False
        self._last_student = None
        self._last_teacher = None

    def configure_model(self):
        super().configure_model()
        self.model.requires_grad_(False)
        qwen = self.model.vlm.model.language_model
        qwen_targets = discover_linear_targets(qwen, QWEN_LORA_SUFFIXES)
        dit_targets = discover_linear_targets(self.model.dit, DIT_LORA_SUFFIXES)
        if len(qwen_targets) != EXPECTED_QWEN_LORA_TARGETS:
            raise RuntimeError(
                f"Qwen LoRA target count mismatch: expected {EXPECTED_QWEN_LORA_TARGETS}, got {len(qwen_targets)}"
            )
        if len(dit_targets) != EXPECTED_DIT_LORA_TARGETS:
            raise RuntimeError(
                f"DiT LoRA target count mismatch: expected {EXPECTED_DIT_LORA_TARGETS}, got {len(dit_targets)}"
            )
        inject_lora(qwen, qwen_targets)
        inject_lora(self.model.dit, dit_targets)
        self._qwen_targets = qwen_targets
        self._dit_targets = dit_targets
        if self.sf:
            if self.layer_number not in (24, 27, 32):
                raise ValueError("Spatial Forcing layer must be 24, 27, or 32")
            layer = self.model.vlm.model.language_model.layers[self.layer_number - 1]

            def capture(_module, _args, output):
                self._sf_hidden = output[0] if isinstance(output, tuple) else output

            self._sf_hook = layer.register_forward_hook(capture)
            qwen_dim = int(self.model.vlm.config.text_config.hidden_size)
            self.model.spatial_alignment_projector = AlignmentProjector(qwen_dim, self.args.use_vlm_norm).to(
                dtype=next(self.model.parameters()).dtype
            )
            self.model.spatial_alignment_projector.requires_grad_(True)
            self.__dict__["_esm"] = FrozenRightESM(self.args.esm_checkpoint, self.args.falcon_root)
        trainable = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        bad = [n for n, _ in trainable if "lora_" not in n and not n.startswith("spatial_alignment_projector.")]
        if bad:
            raise RuntimeError(f"unexpected trainable base parameters: {bad[:12]}")
        if not trainable:
            raise RuntimeError("no trainable LoRA/projector parameters")
        if self.sf and any(p.requires_grad for p in self._esm.parameters()):
            raise RuntimeError("ESM must be frozen")
        print(json.dumps({
            "experiment": self.args.experiment,
            "spatial_forcing": self.sf,
            "alignment_layer": self.layer_number,
            "alpha": self.alpha,
            "qwen_lora_targets": len(qwen_targets),
            "dit_lora_targets": len(dit_targets),
            "trainable_numel": sum(p.numel() for _, p in trainable),
            "esm": "frozen_eval_no_grad_not_in_optimizer" if self.sf else "disabled",
            "official_training_repeat": TRAINING_REPEAT,
        }), flush=True)

    def configure_optimizers(self):
        cfg = {
            "type": "torch.optim.AdamW",
            "params": {"lr": LR, "betas": [0.9, 0.95], "weight_decay": WEIGHT_DECAY, "eps": 1e-8},
        }
        out = self.build_optimizer(cfg, self.named_parameters())
        scheduler = self.build_scheduler({
            "type": "mibot.utils.cosine_warmup.get_cosine_schedule_with_warmup",
            "params": {"num_training_steps": self.args.max_steps, "num_warmup_steps": WARMUP_STEPS,
                       "warmup_lr_start": WARMUP_LR_START, "max_lr": LR, "min_lr": MIN_LR},
        }, out)
        optimizer_ids = {id(p) for g in out.param_groups for p in g["params"]}
        allowed = {id(p) for p in self.model.parameters() if p.requires_grad}
        if optimizer_ids != allowed:
            raise RuntimeError(f"optimizer scope mismatch missing={len(allowed - optimizer_ids)} extra={len(optimizer_ids - allowed)}")
        lr_audit = []
        base_lr = float(out.param_groups[0]["initial_lr"])
        for step in (0, 1, 100, 500, 1000, 6000, self.args.max_steps):
            expected = target_lr(step, self.args.max_steps)
            actual = base_lr * float(scheduler.lr_lambdas[0](step))
            if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-12):
                raise RuntimeError(f"LR audit failed at step {step}: actual={actual} target={expected}")
            lr_audit.append({"step": step, "target_lr": expected, "actual_lr": actual})
        if not math.isclose(float(out.param_groups[0]["lr"]), target_lr(0, self.args.max_steps), rel_tol=1e-6, abs_tol=1e-12):
            raise RuntimeError(f"initial optimizer LR mismatch: {out.param_groups[0]['lr']} vs {target_lr(0, self.args.max_steps)}")
        print(json.dumps({"runtime_lr_audit": lr_audit, "optimizer_base_lr": base_lr}), flush=True)
        return {"optimizer": out, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def _prepare_current_esm(self, batch):
        from PIL import Image
        from falcon.model.policy_head.esm_utils.vggt.utils.load_fn import load_and_preprocess_images_square_new

        raw = batch.get("spatial_images_current")
        if not isinstance(raw, torch.Tensor) or raw.ndim != 5 or raw.shape[1] != 3 or raw.shape[2] != 3:
            raise RuntimeError(f"spatial_images_current must be [B,3,3,H,W], got {type(raw)} {getattr(raw, 'shape', None)}")
        if raw.dtype != torch.uint8:
            raise RuntimeError(f"raw current ESM images must remain uint8 before original preprocessing, got {raw.dtype}")
        images = []
        for sample in raw.detach().cpu():
            for view in sample:
                array = view.permute(1, 2, 0).contiguous().numpy()
                images.append(Image.fromarray(array, mode="RGB"))
        processed, _ = load_and_preprocess_images_square_new(images, target_size=518)
        processed = processed if isinstance(processed, torch.Tensor) else torch.stack(tuple(processed), dim=0)
        expected = (raw.shape[0] * 3, 3, 518, 518)
        if tuple(processed.shape) != expected:
            raise RuntimeError(f"ESM current preprocessing shape mismatch: got {tuple(processed.shape)} expected {expected}")
        device = raw.device
        if self._esm_device != device:
            self._esm.to(device=device, dtype=torch.bfloat16)
            self._esm_device = device
        teacher = self._esm(processed.to(device=device, dtype=torch.bfloat16))
        teacher = teacher.reshape(raw.shape[0], 3, 1369, ESM_HIDDEN)
        if not self._teacher_printed:
            print(json.dumps({"esm_input_raw_batch_shape": list(raw.shape), "esm_preprocessed_shape": list(processed.shape), "esm_output_shape": list(teacher.shape), "esm_special_tokens_removed": {"camera": 1, "register": 4}, "esm_spatial_grid": [37, 37], "esm_fov": "full_raw_current_frame"}), flush=True)
            self._teacher_printed = True
        return teacher

    def _alignment_features(self, batch):
        spatial_batch = batch["spatial_batch"]
        mapping, debug = resolve_current_view_tokens(spatial_batch, self.processor, self._mapping_print)
        self._sf_hidden = None
        vlm_batch = {key: value for key, value in spatial_batch.items() if key != "segments"}
        self.model.vlm(**vlm_batch, use_cache=True, skip_logits=True)
        hidden = self._sf_hidden
        if hidden is None:
            raise RuntimeError("selected Qwen current-only layer hook did not capture output")
        if hidden.ndim == 3 and hidden.shape[0] == 1:
            hidden = hidden[0]
        if hidden.ndim != 2:
            raise RuntimeError(f"current Qwen hidden must be [total_tokens,D], got {tuple(hidden.shape)}")
        self._spatial_hidden = hidden
        student = torch.stack([
            torch.stack([hidden[pos.to(hidden.device)] for pos in sample_mapping], dim=0)
            for sample_mapping in mapping
        ], dim=0)
        grid_set = {tuple(grid) for debug_item in debug for grid in debug_item["view_grids"]}
        if len(grid_set) != 1:
            raise RuntimeError(f"current spatial grids must match across all views/samples, got {sorted(grid_set)}")
        hq, wq = next(iter(grid_set))
        teacher = self._prepare_current_esm(batch)
        teacher = warp_full_teacher_to_crop(teacher, hq, wq).to(dtype=student.dtype)
        if tuple(student.shape[:3]) != tuple(teacher.shape[:3]):
            raise RuntimeError(f"student/teacher shape mismatch: {tuple(student.shape)} vs {tuple(teacher.shape)}")
        self._last_student, self._last_teacher = student, teacher
        return student, teacher, debug

    def _forward(self, batch):
        student = teacher = debug = None
        if self.sf:
            student, teacher, debug = self._alignment_features(batch)
        # These payloads are for the v2 diagnostic/alignment branch only and
        # must not be forwarded as unknown XR-1 VLM kwargs.
        batch.pop("spatial_batch", None)
        batch.pop("spatial_images_current", None)
        result = self.model(batch, return_loss=True)
        # Published inference checkpoint has training-only choice/score heads
        # that are not restored. Keep their values for audit, but make the
        # policy objective independent of those randomly initialized heads so
        # they cannot contaminate Qwen-LoRA gradients.
        loss_choice_audit = result["loss_l1"].detach()
        loss_score_audit = result["loss_score"].detach()
        policy_loss = 0.5 * result["loss_mse"] + result["loss_freq"]
        result["loss_l1"] = loss_choice_audit
        result["loss_score"] = loss_score_audit
        result["loss_choice_audit"] = loss_choice_audit
        result["loss_score_audit"] = loss_score_audit
        result["loss_policy"] = policy_loss
        result["loss"] = policy_loss
        if not self.sf:
            result["loss_align"] = result["loss"].new_zeros(())
            result["loss_official"] = policy_loss
            return result
        projected = self.model.spatial_alignment_projector(student)
        projected = F.normalize(projected.float(), dim=-1)
        teacher = F.normalize(teacher.detach().float(), dim=-1)
        align = (1.0 - (projected * teacher).sum(-1)).mean()
        result["loss_official"] = policy_loss
        result["loss_align"] = align
        result["loss"] = policy_loss + self.alpha * align
        return result

    def training_step(self, batch, batch_idx):
        result = self._forward(batch)
        if self.sf and self.args.negative_controls and self.global_step == 0:
            with torch.no_grad():
                projected = F.normalize(self.model.spatial_alignment_projector(self._last_student).float(), dim=-1)
                normalized_teacher = F.normalize(self._last_teacher.float(), dim=-1)
                controls = {
                    "loss_correct": (1.0 - (projected * normalized_teacher).sum(-1)).mean(),
                    "loss_spatial_shuffle": (1.0 - (projected * normalized_teacher.flip(2)).sum(-1)).mean(),
                    "loss_view_shuffle": (1.0 - (projected * normalized_teacher.roll(1, 1)).sum(-1)).mean(),
                    "loss_sample_shuffle": (1.0 - (projected * normalized_teacher.roll(1, 0)).sum(-1)).mean(),
                }
                print(json.dumps({key: float(value) for key, value in controls.items()}), flush=True)
        for name, value in result.items():
            self.log(f"train/{name}", value.detach(), sync_dist=True)
        self.log("train/token", batch["input_ids"].shape[1])
        return result["loss"]

    def validation_step(self, batch, batch_idx):
        rng_state = capture_rng_state()
        was_training = self.model.training
        val_seed = seed_validation_batch(batch_idx)
        try:
            # Keep the official XR-1 training loss path (including model.train()),
            # while making prefix/noise/timestep/dropout deterministic per batch.
            self.model.train()
            with torch.no_grad():
                result = self._forward(batch)
        finally:
            restore_rng_state(rng_state)
            if not was_training:
                self.model.eval()
        self.log("val/rng_seed", val_seed, sync_dist=False)
        for name, value in result.items():
            self.log(f"val/{name}", value.detach(), prog_bar=name == "loss", sync_dist=True)
        return result["loss"]

    def on_after_backward(self):
        super().on_after_backward()
        if self.global_step == 0 and self.global_rank == 0:
            trainable_grads = {"qwen_lora": 0.0, "dit_lora": 0.0, "alignment_projector": 0.0}
            for name, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                key = "alignment_projector" if name.startswith("spatial_alignment_projector.") else ("qwen_lora" if "vlm.model.language_model" in name else "dit_lora")
                trainable_grads[key] = max(trainable_grads[key], float(p.grad.detach().float().norm().cpu()))
            print(json.dumps({"gradient_smoke": trainable_grads, "esm_grad_none": self.sf and all(p.grad is None for p in self._esm.parameters())}), flush=True)


class MemoryCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.global_step or trainer.global_step % 1 != 0:
            return
        device = pl_module.device if torch.cuda.is_available() else torch.device("cpu")
        local = torch.tensor(
            [torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30],
            device=device,
            dtype=torch.float64,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(local, op=torch.distributed.ReduceOp.MAX)
        if trainer.is_global_zero:
            print(json.dumps({
                "memory_step": int(trainer.global_step),
                "peak_allocated_GB_max_rank": float(local[0].item()),
                "peak_reserved_GB_max_rank": float(local[1].item()),
                "memory_ranks": int(trainer.world_size),
            }), flush=True)


def _move_to_device(value, device):
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_move_to_device(item, device) for item in value)
    return value.to(device) if isinstance(value, torch.Tensor) else value


def _zero_grads(module):
    for parameter in module.parameters():
        parameter.grad = None


def _gradient_report(runner):
    report = {"qwen_lora": {"count": 0, "max_norm": 0.0}, "dit_lora": {"count": 0, "max_norm": 0.0}, "alignment_projector": {"count": 0, "max_norm": 0.0}}
    for name, parameter in runner.model.named_parameters():
        if "lora_" not in name and not name.startswith("spatial_alignment_projector."):
            continue
        key = "alignment_projector" if name.startswith("spatial_alignment_projector.") else ("qwen_lora" if "vlm.model.language_model" in name else "dit_lora")
        if parameter.grad is not None:
            report[key]["count"] += 1
            report[key]["max_norm"] = max(report[key]["max_norm"], float(parameter.grad.detach().float().norm().cpu()))
    return report


def _delta_report(before, module):
    report = {}
    for group, prefixes in {
        "qwen_lora": ("vlm.model.language_model.",),
        "dit_lora": ("dit.",),
        "alignment_projector": ("spatial_alignment_projector.",),
    }.items():
        values = []
        for name, parameter in module.named_parameters():
            if "lora_" not in name and not name.startswith("spatial_alignment_projector."):
                continue
            if not any(name.startswith(prefix) for prefix in prefixes):
                continue
            delta = (parameter.detach().float() - before[name]).abs()
            values.append(delta.reshape(-1))
        if values:
            delta = torch.cat(values)
            report[group] = {"delta_norm": float(torch.linalg.vector_norm(delta).cpu()), "max_abs_delta": float(delta.max().cpu()), "mean_abs_delta": float(delta.mean().cpu())}
        else:
            report[group] = {"delta_norm": 0.0, "max_abs_delta": 0.0, "mean_abs_delta": 0.0}
    return report


def _make_diagnostic_batch(args, processor, rows, stats_processor):
    from torch.utils.data import DataLoader
    data = SixTaskData(rows, processor, batch_size=1, workers=0, include_esm=True, falcon_root=args.falcon_root, gpu_prefetch=False, stats_processor=stats_processor)
    data.setup("fit")
    loader = DataLoader(data.train_set, batch_size=1, shuffle=False, num_workers=0, collate_fn=data.collate)
    batch = next(iter(loader))
    return _move_to_device(batch, torch.device("cuda:0"))


def run_gradient_audit(args, processor, rows, stats_processor):
    if not torch.cuda.is_available():
        raise RuntimeError("gradient audit requires one CUDA device")
    device = torch.device("cuda:0")
    batch = _make_diagnostic_batch(args, processor, rows, stats_processor)
    runner = ForcingRunner(args, processor, sf=True)
    runner.configure_model()
    runner.to(device)
    runner.train()
    result = runner._forward(batch)
    hidden = runner._spatial_hidden
    print(json.dumps({"gradient_graph": {"selected_hidden_requires_grad": bool(hidden is not None and hidden.requires_grad), "selected_hidden_grad_fn": type(hidden.grad_fn).__name__ if hidden is not None and hidden.grad_fn is not None else None, "student_requires_grad": bool(runner._last_student.requires_grad), "projected_requires_grad": bool(runner.model.spatial_alignment_projector(runner._last_student).requires_grad), "loss_align_requires_grad": bool(result["loss_align"].requires_grad)}}), flush=True)
    _zero_grads(runner.model)
    result["loss_align"].backward(retain_graph=True)
    print(json.dumps({"backward_alignment_only": _gradient_report(runner), "esm_grad_none": all(parameter.grad is None for parameter in runner._esm.parameters())}), flush=True)
    _zero_grads(runner.model)
    result["loss_policy"].backward()
    print(json.dumps({"backward_policy_only": _gradient_report(runner), "esm_grad_none": all(parameter.grad is None for parameter in runner._esm.parameters())}), flush=True)
    with torch.no_grad():
        projected = F.normalize(runner.model.spatial_alignment_projector(runner._last_student).float(), dim=-1)
        normalized_teacher = F.normalize(runner._last_teacher.float(), dim=-1)
        controls = {
            "loss_correct": (1.0 - (projected * normalized_teacher).sum(-1)).mean(),
            "loss_spatial_shuffle": (1.0 - (projected * normalized_teacher.flip(2)).sum(-1)).mean(),
            "loss_view_shuffle": (1.0 - (projected * normalized_teacher.roll(1, 1)).sum(-1)).mean(),
            "loss_sample_shuffle": (1.0 - (projected * normalized_teacher.roll(1, 0)).sum(-1)).mean(),
        }
    print(json.dumps({"negative_controls": {key: float(value) for key, value in controls.items()}}), flush=True)

    # A fresh batch is required because XR-1's official forward intentionally
    # pops action/state fields from its input dictionary.
    batch = _make_diagnostic_batch(args, processor, rows, stats_processor)
    _zero_grads(runner.model)
    before = {name: parameter.detach().float().clone() for name, parameter in runner.model.named_parameters() if "lora_" in name or name.startswith("spatial_alignment_projector.")}
    # A direct AdamW step on bf16 module weights can be rounded away at the
    # real 2e-5 training LR, even though DeepSpeed's fp32 master weights will
    # update normally.  Use a separately-labelled visible diagnostic step so
    # this audit checks the update path without conflating it with the runtime
    # LR audit above.
    diagnostic_lr = 1e-3
    optimizer = torch.optim.AdamW([parameter for parameter in runner.model.parameters() if parameter.requires_grad], lr=diagnostic_lr, betas=(0.9, 0.95), weight_decay=0.0, eps=1e-8)
    result = runner._forward(batch)
    result["loss"].backward()
    optimizer.step()
    print(json.dumps({"parameter_update_audit": {"configured_training_lr": LR, "diagnostic_update_lr": diagnostic_lr, "delta": _delta_report(before, runner.model)}}), flush=True)


def run_alignment_overfit_test(args, processor, rows, stats_processor):
    if not torch.cuda.is_available():
        raise RuntimeError("alignment overfit test requires one CUDA device")
    device = torch.device("cuda:0")
    batch = _make_diagnostic_batch(args, processor, rows, stats_processor)
    runner = ForcingRunner(args, processor, sf=True)
    runner.configure_model()
    runner.to(device)
    for parameter in runner.model.parameters():
        parameter.requires_grad_(False)
    runner.model.spatial_alignment_projector.requires_grad_(True)
    runner.eval()
    with torch.no_grad():
        student, teacher, debug = runner._alignment_features(batch)
    optimizer = torch.optim.AdamW(runner.model.spatial_alignment_projector.parameters(), lr=1e-3, weight_decay=0.0)
    initial = None
    last = None
    for step in range(args.overfit_steps + 1):
        projected = F.normalize(runner.model.spatial_alignment_projector(student).float(), dim=-1)
        normalized_teacher = F.normalize(teacher.float(), dim=-1)
        cosine = (projected * normalized_teacher).sum(-1)
        loss = (1.0 - cosine).mean()
        if initial is None:
            initial = float(loss.detach())
        last = float(loss.detach())
        if step == 0 or step == args.overfit_steps or step % max(1, args.overfit_steps // 10) == 0:
            print(json.dumps({"alignment_overfit_step": step, "loss_align": last, "mean_cosine": float(cosine.mean().detach()), "per_view_loss": [float(value.detach()) for value in (1.0 - cosine).mean(dim=(0, 2))], "per_view_cosine": [float(value.detach()) for value in cosine.mean(dim=(0, 2))]}), flush=True)
        if step == args.overfit_steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    if not (last < initial - 0.05 and last < initial * 0.9):
        raise RuntimeError(f"alignment overfit failed: initial={initial} final={last}")
    print(json.dumps({"alignment_overfit_pass": True, "initial_loss": initial, "final_loss": last}), flush=True)


def write_contract(out: Path, args, rows, sf: bool):
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": args.experiment, "spatial_forcing": sf,
        "base_checkpoint": str(args.model_ckpt), "esm_checkpoint": str(args.esm_checkpoint) if sf else None,
        "tasks": list(TASKS), "manifest": str(args.manifest), "manifest_sha256": sha256(args.manifest),
        "train_rows": len(rows["train"]), "val_rows": len(rows["val"]),
        "per_gpu_raw_batch": args.batch_size, "global_raw_batch": args.batch_size * args.devices,
        "accumulate_grad_batches": 1, "training_repeat": TRAINING_REPEAT, "max_steps": args.max_steps,
        "seed": SEED, "val_seed_base": VAL_SEED, "validation_deterministic": True,
        "alignment_layer": args.alignment_layer if sf else None,
        "alpha": args.alpha if sf else 0.0, "alignment_loss": "tokenwise cosine over B x 3 current views x spatial tokens" if sf else None,
        "config_lock": {
            "SF_control": {"alignment_layer": None, "alpha": 0.0},
            "SF_qwen_L24": {"alignment_layer": 24, "alpha": 0.5},
            "SF_qwen_L27": {"alignment_layer": 27, "alpha": 0.5},
            "SF_qwen_L32": {"alignment_layer": 32, "alpha": 0.5},
        },
        "student_views": "LEFT/RIGHT/WRIST current-only one-frame videos" if sf else None, "esm_view": "LEFT/RIGHT/WRIST current raw RGB, original FALCON preprocessing" if sf else None,
        "temporal_contract": "Spatial branch is current-only; policy branch remains history=4 interval=2" if sf else None,
        "geometry_contract": "Qwen uses official center_crop(0.95); ESM uses raw full-FOV frame and explicit full-grid->crop coordinate warp" if sf else "same shared official DataModule",
        "policy_loss": "0.5*loss_mse + loss_freq; loss_l1/loss_score audit-only detached",
        "checkpoint_every_optimization_steps": 1000,
        "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
        "projector": {"architecture": "Linear(Dq,4096)->GELU->Linear(4096,2048)", "use_vlm_norm": bool(args.use_vlm_norm), "initialization": "xavier_uniform,bias_zero"},
        "qwen_lora_suffixes": list(QWEN_LORA_SUFFIXES), "dit_lora_suffixes": list(DIT_LORA_SUFFIXES),
        "lr": LR, "weight_decay": WEIGHT_DECAY, "optimizer": "torch.optim.AdamW", "scheduler": "official cosine warmup",
        "esm_contract": "frozen/eval/no_grad/not optimizer" if sf else "not constructed",
    }
    (out / "experiment_contract.json").write_text(json.dumps(payload, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True, choices=["SF_control", "SF_qwen_L24", "SF_qwen_L27", "SF_qwen_L32"])
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--model-ckpt", type=Path, required=True)
    p.add_argument("--processor", type=Path, required=True)
    p.add_argument("--esm-checkpoint", type=Path, required=True)
    p.add_argument("--falcon-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--devices", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--val-batches", type=int, default=16)
    p.add_argument("--target-gpu-gb", type=float, default=67.0)
    p.add_argument("--limit-gpu-gb", type=float, default=70.0)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--alignment-layer", type=int, default=None)
    p.add_argument("--no-gpu-prefetch", dest="gpu_prefetch", action="store_false")
    p.add_argument("--acceptance-only", action="store_true")
    p.add_argument("--calibration-only", action="store_true")
    p.add_argument("--calibration-steps", type=int, default=8)
    p.add_argument("--use-vlm-norm", action="store_true", help="Opt-in official-compatible VLM LayerNorm before projector")
    p.add_argument("--gradient-audit", action="store_true", help="Run no-DeepSpeed single-GPU gradient/update audit")
    p.add_argument("--alignment-overfit-test", action="store_true", help="Run 500-step projector-only fixed-batch test")
    p.add_argument("--overfit-steps", type=int, default=500)
    p.add_argument("--negative-controls", action="store_true", help="Print spatial/view/sample-shuffle controls on first train batch")
    args = p.parse_args()
    locked = {
        "SF_control": {"alignment_layer": None, "alpha": 0.0},
        "SF_qwen_L24": {"alignment_layer": 24, "alpha": 0.5},
        "SF_qwen_L27": {"alignment_layer": 27, "alpha": 0.5},
        "SF_qwen_L32": {"alignment_layer": 32, "alpha": 0.5},
    }[args.experiment]
    if args.alignment_layer is not None and args.alignment_layer != locked["alignment_layer"]:
        raise ValueError(
            f"{args.experiment} locks alignment_layer={locked['alignment_layer']}; "
            f"CLI supplied {args.alignment_layer}"
        )
    if args.alpha is not None and not math.isclose(args.alpha, locked["alpha"], rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{args.experiment} locks alpha={locked['alpha']}; CLI supplied {args.alpha}")
    args.alignment_layer = locked["alignment_layer"]
    args.alpha = locked["alpha"]
    seed_all(SEED)
    args.dataset_root = Path(args.dataset_root)
    common.DATA_ROOT = args.dataset_root
    common.MANIFEST = args.manifest
    common.META = args.manifest.with_suffix(".meta.json")
    common.MODEL_CKPT = args.model_ckpt
    common.PROCESSOR = args.processor
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_six_rows(args.manifest)
    stats_processor = common.build_action_stats_processor()
    processor = common.build_official_qwen_processor()
    sf = args.experiment != "SF_control"
    if args.gradient_audit:
        if not sf:
            raise ValueError("--gradient-audit requires an SF_qwen_L24/L27/L32 experiment")
        run_gradient_audit(args, processor, rows, stats_processor)
        return
    if args.alignment_overfit_test:
        if not sf:
            raise ValueError("--alignment-overfit-test requires an SF experiment")
        run_alignment_overfit_test(args, processor, rows, stats_processor)
        return
    write_contract(args.output_dir, args, rows, sf)
    print(json.dumps({"experiment": args.experiment, "tasks": list(TASKS), "train_rows": len(rows["train"]), "val_rows": len(rows["val"]), "per_gpu_raw_batch": args.batch_size, "global_raw_batch": args.batch_size * args.devices, "seed": SEED}), flush=True)
    data = SixTaskData(rows, processor, args.batch_size, args.workers, sf, args.falcon_root, args.gpu_prefetch, stats_processor)
    runner = ForcingRunner(args, processor, sf)
    from lightning import Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.strategies import DeepSpeedStrategy
    short_run = args.acceptance_only or args.calibration_only
    max_steps = 2 if args.acceptance_only else (args.calibration_steps if args.calibration_only else args.max_steps)
    val_interval = 999999 if short_run else 1000
    callbacks = [MemoryCallback()] if short_run else [MemoryCallback(), ModelCheckpoint(dirpath=str(args.output_dir / "checkpoints"), every_n_train_steps=1000, save_top_k=-1, save_last=True, enable_version_counter=False)]
    trainer = Trainer(accelerator="cuda", devices=args.devices, num_nodes=1, precision="bf16-mixed", strategy=DeepSpeedStrategy(), max_steps=max_steps, default_root_dir=str(args.output_dir), accumulate_grad_batches=1, gradient_clip_val=1.0, log_every_n_steps=1 if short_run else 10, val_check_interval=val_interval, check_val_every_n_epoch=None, limit_train_batches=2 if args.acceptance_only else None, limit_val_batches=args.val_batches, num_sanity_val_steps=0, callbacks=callbacks, enable_checkpointing=not short_run)
    trainer.fit(runner, datamodule=data)


if __name__ == "__main__":
    main()
