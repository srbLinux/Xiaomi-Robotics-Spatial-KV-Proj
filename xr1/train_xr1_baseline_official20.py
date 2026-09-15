#!/usr/bin/env python3
"""Official XR-1 post-training on the verified RoboCasa365 20-task split.

This file deliberately keeps the training mathematics in clean ``mibot``:
the only model call used for training is ``model(batch, return_loss=True)``.
The adapter below only turns the RoboCasa365 parquet/video contract into the
flat batch consumed by the official runner.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import types
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLImageProcessor, Qwen3VLProcessor, Qwen3VLVideoProcessor

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MAX_LENGTH", "40000")

from mibot.models.VLA.XR1 import xr1  # noqa: E402
from mibot.models.runner.base_runner import BaseRunner  # noqa: E402
import mibot.data  # noqa: F401,E402  (registers official modules)

DATA_ROOT = Path("/data-cfs/data3/datasets/robocasa365-datasets/pretrain")
MANIFEST = Path("/data-cfs/data3/shurenbin/Xiaomi-Robotics-Spatial/train/esm_kv_injection/xr1_base_spatial20_full_dit_20260913/manifests/spatial20_train_val_95_5_20260913.jsonl")
META = Path(str(MANIFEST) + ".meta.json")
MODEL_CKPT = Path("/data-cfs/data3/models/Xiaomi-Robotics-1-5B/model_states.pt")
PROCESSOR = Path("/data-cfs/data3/models/Xiaomi-Robotics-1-RoboCasa365")
TASKS = (
    "StackBowlsCabinet", "PrepareCoffee", "SearingMeat", "RinseSinkBasin",
    "SetUpCuttingStation", "StoreLeftoversInBowl", "LoadDishwasher",
    "PackIdenticalLunches", "CoffeeSetupMug", "PickPlaceCounterToStove",
    "AlignSilverware", "LineUpCondiments", "OrganizeMugsByHandle", "StackCans",
    "PlaceVegetablesEvenly", "SizeSorting", "MaximizeFreezerSpace",
    "FryingPanAdjustment", "PlaceBeveragesTogether", "MatchCupAndDrink",
)
LEFT = "observation.images.robot0_agentview_left"
RIGHT = "observation.images.robot0_agentview_right"
WRIST = "observation.images.robot0_eye_in_hand"
VIEWS = (LEFT, RIGHT, WRIST)
STATE_DIM, ACTION_DIM, STATE_HISTORY, STATE_INTERVAL, ACTION_HORIZON = 60, 60, 4, 2, 16
SEED = 42


def quat_axis(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float32)
    q = q / norm
    if q[3] < 0:
        q = -q
    s = np.linalg.norm(q[:3])
    if s < 1e-12:
        return np.zeros(3, dtype=np.float32)
    return (q[:3] / s * (2 * np.arctan2(s, np.clip(q[3], -1, 1)))).astype(np.float32)


def state_to_60(raw) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32).reshape(-1)
    x = np.concatenate((raw[7:10], quat_axis(raw[10:14]), raw[14:16], raw[:3], quat_axis(raw[3:7])))
    out = np.zeros(STATE_DIM, dtype=np.float32)
    out[: len(x)] = x
    return out


def action_to_60(raw) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32).reshape(-1)
    x = np.concatenate((raw[5:8], raw[8:11], raw[11:12], raw[:4], raw[4:5]))
    out = np.zeros(ACTION_DIM, dtype=np.float32)
    out[: len(x)] = x
    return out


def remap_path(path: str) -> str:
    old = "/data-cfs/data3/shurenbin/datasets/robocasa365-extracted"
    if path.startswith(old):
        path = str(DATA_ROOT) + path[len(old):]
    if not path.startswith(str(DATA_ROOT)):
        raise RuntimeError(f"dataset path outside accepted pretrain root: {path}")
    return path


def load_rows() -> dict[str, list[dict]]:
    meta = json.loads(META.read_text())
    if tuple(meta.get("tasks", ())) != TASKS or meta.get("task_count") != 20:
        raise RuntimeError("verified exact-20 manifest metadata mismatch")
    rows = {"train": [], "val": []}
    for line in MANIFEST.open():
        row = json.loads(line)
        if row.get("task") in TASKS and row.get("split") in rows:
            row["camera_paths"] = {k: remap_path(v) for k, v in row["camera_paths"].items()}
            rows[row["split"]].append(row)
    if any(not rows[k] for k in rows):
        raise RuntimeError("verified train/val split is empty")
    train_eps = {(r["task"], int(r["episode_id"])) for r in rows["train"]}
    val_eps = {(r["task"], int(r["episode_id"])) for r in rows["val"]}
    if train_eps & val_eps:
        raise RuntimeError("episode leakage detected")
    return rows


def center_crop(image: np.ndarray, ratio: float = 0.95) -> Image.Image:
    image = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    w, h = image.size
    cw, ch = max(1, int(w * ratio)), max(1, int(h * ratio))
    x, y = (w - cw) // 2, (h - ch) // 2
    # Preserve the verified RoboCasa crop contract while restoring the source
    # resolution so the official processor receives patch-aligned dimensions.
    return image.crop((x, y, x + cw, y + ch)).resize((w, h), Image.Resampling.BILINEAR)


class RoboCasa20(Dataset):
    def __init__(self, rows: list[dict]):
        from decord import VideoReader
        self.rows = rows
        self.VideoReader = VideoReader
        self.tables: dict[str, object] = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        right = row["camera_paths"][RIGHT]
        root = Path(right.split("/videos/", 1)[0])
        parquet = root / "data/chunk-000" / f"episode_{int(row['episode_id']):06d}.parquet"
        table = self.tables.setdefault(str(parquet), pq.read_table(str(parquet), columns=["observation.state", "action"]))
        frame = int(row["frame_index"])
        history = [max(0, frame - (STATE_HISTORY - 1 - i) * STATE_INTERVAL) for i in range(STATE_HISTORY)]
        states = np.stack([state_to_60(table["observation.state"][i].as_py()) for i in history])
        action = np.zeros((ACTION_HORIZON, ACTION_DIM), dtype=np.float32)
        valid = np.zeros(ACTION_HORIZON, dtype=np.float32)
        for j in range(ACTION_HORIZON):
            if frame + j < table.num_rows:
                action[j] = action_to_60(table["action"][frame + j].as_py())
                valid[j] = 1.0
        videos, fps = {}, {}
        for view in VIEWS:
            reader = self.VideoReader(row["camera_paths"][view], num_threads=1)
            if len(reader) <= max(history):
                raise RuntimeError(f"video shorter than history: {row['sample_id']}")
            videos[view] = [center_crop(x) for x in reader.get_batch(history).asnumpy()]
            fps[view] = float(reader.get_avg_fps())
        return {"sample_id": row["sample_id"], "task": row["task"], "instruction": row["instruction"], "frame": frame, "videos": videos, "fps": fps, "state": states, "action": action, "valid": valid}


def build_action_stats_processor():
    return AutoProcessor.from_pretrained(str(PROCESSOR), trust_remote_code=True, use_fast=False, extra_special_tokens={"score": "<score>", "state": "<state>", **{f"a_{j}": f"<a_{j}>" for j in range(60)}})


def build_official_qwen_processor():
    """Build the clean-collate Qwen processor entirely from local files."""
    special = {"score": "<score>", "state": "<state>", **{f"a_{j}": f"<a_{j}>" for j in range(60)}}
    tokenizer = AutoTokenizer.from_pretrained(str(PROCESSOR), use_fast=True, local_files_only=True, extra_special_tokens=special)
    image_processor = Qwen2VLImageProcessor.from_pretrained(str(PROCESSOR), local_files_only=True)
    video_processor = Qwen3VLVideoProcessor.from_pretrained(str(PROCESSOR), local_files_only=True)
    return Qwen3VLProcessor(image_processor, tokenizer, video_processor, chat_template=(PROCESSOR / "chat_template.jinja").read_text())


def official_messages(item):
    content = []
    for index, (label, view) in enumerate((("Left camera: ", LEFT), ("Right camera: ", RIGHT), ("Wrist camera: ", WRIST))):
        if index:
            content.append({"type": "text", "text": "\n"})
        content.extend(({"type": "text", "text": label}, {"type": "video", "video": item["videos"][view], "fps": item["fps"][view]}))
    content.append({"type": "text", "text": "\n\nGenerate robot actions for the task:\n" + str(item["instruction"]) + " /no_cot"})
    steps = max(1, int(item["valid"].sum()))
    return [{"role": "user", "content": content}, {"role": "user", "content": [{"type": "text", "text": "Robot state: <state>"}]}, {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>" + "".join(f"<a_{j}>" for j in range(steps)) + "<score>"}]}]


class OfficialSampleDataset(Dataset):
    """Thin adapter from raw RoboCasa rows to clean JsonDataset sample schema."""
    def __init__(self, raw_dataset, stats_processor):
        self.raw_dataset = raw_dataset
        cfg = stats_processor.action_config["robocasa365"]
        self.mean, self.std = cfg["mean"].squeeze(0).float(), cfg["std"].squeeze(0).float()
        if tuple(self.mean.shape) != (ACTION_HORIZON, ACTION_DIM):
            raise RuntimeError(f"processor action stats must be {(ACTION_HORIZON, ACTION_DIM)}, got {tuple(self.mean.shape)}")
        if not torch.equal(stats_processor.get_action_mask("robocasa365").squeeze(0).bool(), self.std > 1e-5):
            raise RuntimeError("processor action mask != std > 1e-5")
        self.action_mask = (self.std > 1e-5).float()

    def __len__(self): return len(self.raw_dataset)

    def __getitem__(self, index):
        raw = self.raw_dataset[index]
        target = torch.where(self.action_mask.bool(), (torch.from_numpy(raw["action"]) - self.mean) / self.std.clamp_min(1e-5), torch.zeros_like(torch.from_numpy(raw["action"])))
        steps = max(1, int(raw["valid"].sum()))
        valid_mask = self.action_mask * torch.from_numpy(raw["valid"])[:, None]
        return {"messages": official_messages(raw), "action": target, "action_mask": valid_mask, "state": torch.from_numpy(raw["state"]), "vlm_action_target": target[:steps], "vlm_action_mask": valid_mask[:steps], "vlm_action_actual_length": steps, "sample_id": raw["sample_id"], "task": raw["task"], **({"esm_image": raw["esm_image"]} if "esm_image" in raw else {})}


from mibot.data.collate.custom_collate import CustomCollate as _CleanCustomCollate


class LocalOfficialCollate(_CleanCustomCollate):
    """Use clean official ``__call__``; only inject local offline processor."""
    def __init__(self, processor):
        self.max_length = int(os.environ.get("MAX_LENGTH", 4096))
        self.processor = processor
        token_ids = self.processor.tokenizer.convert_tokens_to_ids(["<score>", "<state>", "<a_0>", "<a_59>"])
        if token_ids != [151669, 151670, 151671, 151730]:
            raise ValueError(f"unexpected action token ids: {token_ids}")


class CUDAPrefetcher:
    def __init__(self, loader):
        self.loader = loader
        self.device = None
        self.stream = None
        self.iterator = None
        self.next_batch = None

    @staticmethod
    def _move(value, device):
        if isinstance(value, dict): return {k: CUDAPrefetcher._move(v, device) for k, v in value.items()}
        if isinstance(value, (list, tuple)): return type(value)(CUDAPrefetcher._move(v, device) for v in value)
        return value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value

    def __iter__(self):
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.stream = torch.cuda.Stream(self.device)
        self.iterator = iter(self.loader)
        self._preload()
        return self

    def _preload(self):
        try: batch = next(self.iterator)
        except StopIteration: self.next_batch = None; return
        with torch.cuda.stream(self.stream): self.next_batch = self._move(batch, self.device)

    def __next__(self):
        if self.next_batch is None: raise StopIteration
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self.next_batch
        self._preload()
        return batch

    def __len__(self): return len(self.loader)

    @property
    def sampler(self): return self.loader.sampler

    @property
    def dataset(self): return self.loader.dataset

    def set_epoch(self, epoch):
        if hasattr(self.loader.sampler, "set_epoch"):
            self.loader.sampler.set_epoch(epoch)

    def __getattr__(self, name):
        # Keep Lightning's DataLoader/sampler introspection working while the
        # iterator itself performs the asynchronous host-to-device transfer.
        return getattr(self.loader, name)


class Official20Data(LightningDataModule):
    def __init__(self, rows, processor, batch_size, workers, gpu_prefetch=True, stats_processor=None):
        super().__init__()
        self.rows, self.processor, self.batch_size, self.workers = rows, processor, batch_size, workers
        self.gpu_prefetch = bool(gpu_prefetch)
        self.stats_processor = stats_processor or build_action_stats_processor()
        self.collate = LocalOfficialCollate(processor)
        self.train_set = None
        self.val_set = None

    def prepare_data(self):
        # All artifacts are local and are intentionally never downloaded here.
        return None

    def setup(self, stage=None):
        if stage in (None, "fit"):
            if self.train_set is None:
                self.train_set = OfficialSampleDataset(RoboCasa20(self.rows["train"]), self.stats_processor)
            if self.val_set is None:
                self.val_set = OfficialSampleDataset(RoboCasa20(self.rows["val"]), self.stats_processor)

    def _loader(self, dataset, sampler, shuffle=False):
        kwargs = dict(
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=self.workers,
            pin_memory=True,
            persistent_workers=self.workers > 0,
            collate_fn=self.collate,
        )
        if self.workers:
            kwargs["prefetch_factor"] = 2
        loader = DataLoader(dataset, **kwargs)
        return CUDAPrefetcher(loader) if torch.cuda.is_available() and self.gpu_prefetch else loader

    def train_dataloader(self):
        self.setup("fit")
        sampler = DistributedSampler(self.train_set, shuffle=True, seed=SEED)
        return self._loader(self.train_set, sampler)

    def val_dataloader(self):
        self.setup("fit")
        sampler = DistributedSampler(self.val_set, shuffle=False, seed=SEED)
        return self._loader(self.val_set, sampler)


def memory_policy(target_gb: float, limit_gb: float):
    if not torch.cuda.is_available(): return
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 2**30
    effective = min(float(limit_gb), total * 0.98)
    torch.cuda.set_per_process_memory_fraction(effective / total)
    print(json.dumps({"gpu_total_gib": total, "gpu_target_gib": target_gb, "gpu_allocator_limit_gib": effective}), flush=True)


def use_local_vlm_config():
    """Keep clean XR1 source, but resolve its released Qwen config offline."""
    from transformers import Qwen3VLConfig
    original = Qwen3VLConfig.from_pretrained.__func__

    def local(cls, name_or_path, *args, **kwargs):
        if str(name_or_path) == "Qwen/Qwen3-VL-4B-Instruct":
            payload = json.loads((PROCESSOR / "config.json").read_text())["vlm_config"]
            return cls.from_dict(payload)
        return original(cls, name_or_path, *args, **kwargs)

    Qwen3VLConfig.from_pretrained = classmethod(local)


class Official20Runner(BaseRunner):
    def __init__(self, pretrained, max_steps, target_gb, limit_gb):
        super().__init__({"pretrained": str(pretrained), "model": {"type": "xr1", "freq_coefficient": 1.0, "freq_excluded_dims": [17, 18, 19], "ffn_gradient_checkpointing": True, "async_train": True}, "optimizer": {"type": "deepspeed.ops.adam.FusedAdam", "params": {"lr": 1.0, "betas": [0.9, 0.95], "weight_decay": 0.1, "eps": 1e-8}}, "scheduler": {"type": "mibot.utils.cosine_warmup.get_cosine_schedule_with_warmup", "params": {"num_training_steps": max_steps, "num_warmup_steps": 500, "warmup_lr_start": 5e-7, "max_lr": 2e-5, "min_lr": 5e-6}}})
        self.max_steps, self.target_gb, self.limit_gb = max_steps, target_gb, limit_gb

    def configure_model(self):
        use_local_vlm_config()
        self.model = xr1(freq_coefficient=1.0, freq_excluded_dims=[17, 18, 19], ffn_gradient_checkpointing=True, async_train=True)
        # In clean XR1, only the last dimension is consumed by constructor-time
        # projectors; sequence lengths are read from each runtime batch.  Set
        # the RoboCasa contract after construction and before load/forward.
        self.model.state_shape = (STATE_HISTORY, STATE_DIM)
        self.model.action_shape = (ACTION_HORIZON, ACTION_DIM)
        if self.model.state_projector_choice.layers[0].in_features != STATE_DIM or self.model.action_projector.layers[0].in_features != ACTION_DIM:
            raise RuntimeError("clean XR1 projector input dimensions are not the expected 60")
        if self.model.state_shape != (STATE_HISTORY, STATE_DIM) or self.model.action_shape != (ACTION_HORIZON, ACTION_DIM):
            raise RuntimeError("RoboCasa state/action shape contract was not applied before load")
        ckpt = torch.load(str(MODEL_CKPT), map_location="cpu", mmap=True, weights_only=False)
        info = self.load_state_dict(ckpt["module"], strict=False)
        allowed_missing, allowed_unexpected = set(), set()
        missing = sorted(set(info.missing_keys) - allowed_missing)
        unexpected = sorted(set(info.unexpected_keys) - allowed_unexpected)
        print(json.dumps({"official_checkpoint": str(MODEL_CKPT), "missing_keys": info.missing_keys, "unexpected_keys": info.unexpected_keys, "allowed_missing_keys": sorted(allowed_missing), "allowed_unexpected_keys": sorted(allowed_unexpected), "shape_patch_stage": "post-construction-before-load"}), flush=True)
        if missing or unexpected:
            raise RuntimeError(f"official checkpoint mismatch outside strict whitelist: missing={missing[:8]}, unexpected={unexpected[:8]}")
        del ckpt

    def on_train_start(self):
        memory_policy(self.target_gb, self.limit_gb)
        if self.global_rank == 0:
            trainable = {"total": sum(p.numel() for p in self.model.parameters() if p.requires_grad), "input_embeddings_frozen": not self.model.vlm.model.get_input_embeddings().weight.requires_grad}
            print(json.dumps({"trainable_summary": trainable, "official_training_repeat": self.model.training_repeat, "official_async_train": self.model.async_train}), flush=True)

    def validation_step(self, batch, batch_idx):
        # Official flow objective is selected by the official module's training flag.
        was_training = self.model.training
        self.model.train()
        with torch.no_grad(): result = self.model(batch, return_loss=True)
        if not was_training: self.model.eval()
        for name, value in result.items(): self.log(f"val/{name}", value.detach(), prog_bar=name == "loss", sync_dist=True)
        return result["loss"]

    def on_after_backward(self):
        if self.global_step == 0 and self.global_rank == 0:
            names = ("visual", "language_model", "dit", "state_projector", "action_projector", "t_embedder", "action_output_layer")
            stats = {}
            for key in names:
                vals = [p.grad.detach().float().norm().item() for n, p in self.model.named_parameters() if key in n and p.grad is not None]
                stats[key] = max(vals, default=0.0)
            print(json.dumps({"gradient_smoke": stats}), flush=True)


def write_audit(out: Path, processor, rows, args):
    cfg = processor.action_config["robocasa365"]
    mean, std = cfg["mean"].squeeze(0), cfg["std"].squeeze(0)
    raw = torch.linspace(-0.01, 0.01, ACTION_DIM).view(1, ACTION_DIM).repeat(ACTION_HORIZON, 1)
    normalized = torch.where(std > 1e-5, (raw - mean) / std.clamp_min(1e-5), torch.zeros_like(raw))
    decoded = normalized * std + mean
    (out / "action_normalization_roundtrip.json").write_text(json.dumps({"processor": str(PROCESSOR), "mean_first12": mean[0, :12].tolist(), "std_first12": std[0, :12].tolist(), "active_indices": torch.where(std[0] > 1e-5)[0].tolist(), "max_abs_error": float((decoded[:, :12] - raw[:, :12]).abs().max())}, indent=2) + "\n")
    (out / "normalization_audit.json").write_text(json.dumps({"action": {"processor_source": str(PROCESSOR), "mean_first12": mean[0, :12].tolist(), "std_first12": std[0, :12].tolist(), "active_indices": torch.where(std[0] > 1e-5)[0].tolist(), "roundtrip_max_abs_error": float((decoded[:, :12] - raw[:, :12]).abs().max())}, "state": {"shape": [STATE_HISTORY, STATE_DIM], "extra_state_normalization": False}, "vlm_rgb": {"source": "official RoboCasa365 processor", "manual_rescale": False, "manual_normalize": False}, "manifest": {"path": str(MANIFEST), "train_rows": len(rows["train"]), "val_rows": len(rows["val"]), "tasks": list(TASKS), "episode_split": "verified metadata"}}, indent=2) + "\n")
    (out / "training_contract.json").write_text(json.dumps({"max_steps": int(args.max_steps), "val_check_interval_optimization_steps": 1000, "save_every_optimization_steps": 2000, "gradient_accumulation": 1, "per_gpu_batch_size": int(args.batch_size), "gpu_prefetch": bool(args.gpu_prefetch), "gpu_allocator_limit_gib": float(args.limit_gpu_gb)}, indent=2) + "\n")
    (out / "collate_contract.json").write_text(json.dumps({"reference": "clean official mibot/data/collate/custom_collate.py", "reuse": "inherited official CustomCollate.__call__", "same_sample_fields": ["messages", "input_ids", "position_ids", "cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k", "action_segments", "action_vlm_condition_segments", "pixel_values", "image_grid_thw", "state", "action", "action_mask", "vlm_action_target", "vlm_action_mask", "vlm_action_actual_length", "sample_id", "task"], "dataset_adapter": "RoboCasa20 raw item -> official JsonDataset sample schema", "processor_adapter": "local Qwen3VLProcessor built from local tokenizer/image/video files; clean collate constructor is not used because it hardcodes online Qwen path", "manual_collate_logic": False}, indent=2) + "\n")
    (out / "official_source_audit.md").write_text(f"# Official XR-1 source audit\n\n- clean commit: `0dd7aef8dc87296246aae812a1f59ccb708e5546`\n- model: `xr1/mibot/models/VLA/XR1.py`\n- runner: `xr1/mibot/models/runner/base_runner.py`\n- collate reference: `xr1/mibot/data/collate/custom_collate.py`\n- official call: `model(batch, return_loss=True)`\n- official losses: flow MSE, frequency, choice, score\n- official repeat/async: `training_repeat=4`, `async_train=true`\n- official optimizer: DeepSpeed FusedAdam, official no-decay grouping\n- official scheduler: cosine warmup, 500 warmup steps, 2e-5 max, 5e-6 min\n- RoboCasa contract: state `[B,4,60]`, action `[B,16,60]`, active dims from processor metadata\n- manifest: `{MANIFEST}`\n- GPU: 3 processes, per-GPU batch `{args.batch_size}`, GPU prefetch `{args.gpu_prefetch}`\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--devices", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=40000)
    p.add_argument("--val-batches", type=int, default=16)
    p.add_argument("--output-dir", type=Path, default=Path("/data-cfs/data3/shurenbin/Xiaomi-Robotics-Spatial/train/xr1_official20/baseline"))
    p.add_argument("--target-gpu-gb", type=float, default=70.0)
    p.add_argument("--limit-gpu-gb", type=float, default=70.0)
    p.add_argument("--no-gpu-prefetch", dest="gpu_prefetch", action="store_false")
    p.set_defaults(gpu_prefetch=True)
    p.add_argument("--audit-only", action="store_true")
    args = p.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_processor = build_action_stats_processor()
    processor = build_official_qwen_processor()
    rows = load_rows()
    write_audit(args.output_dir, stats_processor, rows, args)
    print(json.dumps({"mode": "baseline", "tasks": list(TASKS), "train_rows": len(rows["train"]), "val_rows": len(rows["val"]), "state_shape": ["B", 4, 60], "action_shape": ["B", 16, 60], "per_gpu_batch_size": args.batch_size, "gpu_prefetch": args.gpu_prefetch}), flush=True)
    if args.audit_only: return
    data = Official20Data(rows, processor, args.batch_size, args.workers, args.gpu_prefetch, stats_processor)
    runner = Official20Runner(MODEL_CKPT, args.max_steps, args.target_gpu_gb, args.limit_gpu_gb)
    from lightning import Trainer
    from lightning.pytorch.strategies import DeepSpeedStrategy
    from lightning.pytorch.callbacks import ModelCheckpoint
    checkpoint = ModelCheckpoint(dirpath=str(args.output_dir / "checkpoints"), every_n_train_steps=2000, save_top_k=-1, save_last=True, enable_version_counter=False)
    trainer = Trainer(accelerator="cuda", devices=args.devices, num_nodes=1, precision="bf16-mixed", strategy=DeepSpeedStrategy(), max_steps=args.max_steps, default_root_dir=str(args.output_dir), accumulate_grad_batches=1, gradient_clip_val=1.0, log_every_n_steps=10, val_check_interval=1000, check_val_every_n_epoch=None, limit_val_batches=args.val_batches, callbacks=[checkpoint], enable_checkpointing=True)
    trainer.fit(runner, datamodule=data)


if __name__ == "__main__":
    main()
