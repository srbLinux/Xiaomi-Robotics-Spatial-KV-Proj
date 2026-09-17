#!/usr/bin/env python3
"""Build the fixed, episode-disjoint Stage1 six-task manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

DATA_ROOT = Path("/data-cfs/data3/datasets/robocasa365-datasets/pretrain")
TASKS = (
    "CloseBlenderLid", "CoffeeSetupMug", "TurnOffStove", "TurnOnMicrowave",
    "SlideDishwasherRack", "PickPlaceCounterToCabinet",
)
VIEWS = (
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
)


def read_instruction(meta_root: Path, task: str) -> str:
    records = [json.loads(line) for line in (meta_root / "tasks.jsonl").open()]
    for record in records:
        if record.get("task_index") == 0:
            return str(record["task"])
    for record in records:
        if record.get("task") == task:
            return str(record["task"])
    raise RuntimeError(f"no task instruction found for {task} in {meta_root}")


def complete_episodes(task: str) -> list[dict]:
    datasets = sorted((DATA_ROOT / "atomic" / task).glob("*/lerobot"))
    if not datasets:
        raise FileNotFoundError(f"no RoboCasa365 dataset found for {task}")
    result = []
    for root in datasets:
        episodes_path = root / "meta" / "episodes.jsonl"
        if not episodes_path.is_file():
            raise FileNotFoundError(episodes_path)
        instruction = read_instruction(root / "meta", task)
        episodes = sorted((json.loads(line) for line in episodes_path.open()), key=lambda x: int(x["episode_index"]))
        for episode in episodes:
            episode_id = int(episode["episode_index"])
            length = int(episode["length"])
            parquet = root / "data" / "chunk-000" / f"episode_{episode_id:06d}.parquet"
            camera_paths = {
                key: root / "videos" / "chunk-000" / key / f"episode_{episode_id:06d}.mp4"
                for key in VIEWS
            }
            if any(not path.is_file() for path in [parquet, *camera_paths.values()]):
                continue
            table = pq.read_table(str(parquet), columns=["frame_index", "episode_index"])
            frame_ids = [int(x) for x in table["frame_index"].to_pylist()]
            episode_ids = {int(x) for x in table["episode_index"].to_pylist()}
            if table.num_rows != length or frame_ids != list(range(length)) or episode_ids != {episode_id}:
                continue
            result.append({
                "task": task,
                "dataset_date": root.parent.name,
                "episode_id": episode_id,
                "episode_key": f"{root.parent.name}:{episode_id}",
                "length": length,
                "instruction": instruction,
                "camera_paths": {name: str(path) for name, path in camera_paths.items()},
            })
    result.sort(key=lambda x: (x["dataset_date"], x["episode_id"]))
    if len(result) < 3:
        raise RuntimeError(f"{task} has only {len(result)} complete episodes; at least 3 are required")
    return result


def build(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "spatial6_stage1_train_val.jsonl"
    meta_path = output_dir / "spatial6_stage1_train_val.meta.json"
    all_rows: list[dict] = []
    per_task: dict[str, dict] = {}
    for task in TASKS:
        episodes = complete_episodes(task)
        val_keys = {episode["episode_key"] for episode in episodes[-2:]}
        per_task[task] = {
            "total_episode_count": len(episodes),
            "train_episode_ids": [episode["episode_id"] for episode in episodes[:-2]],
            "val_episode_ids": [episode["episode_id"] for episode in episodes[-2:]],
            "train_episode_keys": [episode["episode_key"] for episode in episodes[:-2]],
            "val_episode_keys": [episode["episode_key"] for episode in episodes[-2:]],
            "train_row_count": 0,
            "val_row_count": 0,
        }
        for episode in episodes:
            split = "val" if episode["episode_key"] in val_keys else "train"
            for frame in range(episode["length"]):
                all_rows.append({
                    "sample_id": f"{task}_{episode['dataset_date']}_ep{episode['episode_id']:06d}_f{frame:06d}",
                    "task": task,
                    "instruction": episode["instruction"],
                    "episode_id": episode["episode_id"],
                    "episode_key": episode["episode_key"],
                    "frame_index": frame,
                    "split": split,
                    "camera_paths": episode["camera_paths"],
                })
                per_task[task][f"{split}_row_count"] += 1
    all_rows.sort(key=lambda x: (TASKS.index(x["task"]), x["episode_key"], x["frame_index"]))
    with manifest.open("w") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    train_keys = {(row["task"], row["episode_key"]) for row in all_rows if row["split"] == "train"}
    val_keys = {(row["task"], row["episode_key"]) for row in all_rows if row["split"] == "val"}
    if train_keys & val_keys:
        raise RuntimeError("episode leakage detected")
    metadata = {
        "task_names": list(TASKS),
        "tasks": list(TASKS),
        "task_count": len(TASKS),
        "per_task": per_task,
        "train_row_count": sum(row["split"] == "train" for row in all_rows),
        "val_row_count": sum(row["split"] == "val" for row in all_rows),
        "split_rule": "last_2_sorted_episode_ids_per_task",
        "manifest_sha256": digest,
        "data_root": str(DATA_ROOT),
        "complete_episode_definition": "parquet and all three camera videos exist; parquet frame_index is 0..length-1 and episode_index is constant",
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"manifest": str(manifest), "meta": str(meta_path), "sha256": digest, "train_rows": metadata["train_row_count"], "val_rows": metadata["val_row_count"], "per_task": per_task}, ensure_ascii=False, indent=2))
    return manifest, meta_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("manifests"))
    args = parser.parse_args()
    build(args.output_dir)


if __name__ == "__main__":
    main()
