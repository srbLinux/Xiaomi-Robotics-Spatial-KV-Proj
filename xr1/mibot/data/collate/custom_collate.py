# Copyright (C) 2026 Xiaomi Corporation.
import itertools
import os
from typing import Any, Dict, List

import torch
from torch.utils.data.dataloader import default_collate
from transformers import AutoProcessor

SCORE_ID = 151669
STATE_ID = 151670
IM_START_ID = 151644
IMAGE_ID = 151655
VIDEO_ID = 151656


class CustomCollate:
    def __init__(self) -> None:
        self.max_length = int(os.environ.get("MAX_LENGTH", 4096))
        special_tokens = {"score": "<score>", "state": "<state>"}
        special_tokens.update({f"a_{index}": f"<a_{index}>" for index in range(60)})
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-4B-Instruct", use_fast=True, extra_special_tokens=special_tokens
        )
        token_ids = self.processor.tokenizer.convert_tokens_to_ids(
            ["<score>", "<state>", "<a_0>", "<a_59>"]
        )
        if token_ids != [SCORE_ID, STATE_ID, STATE_ID + 1, STATE_ID + 60]:
            raise ValueError(f"unexpected action token ids: {token_ids}")

    def _position_ids(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs["input_ids"]
        image_grids = inputs.get("image_grid_thw")
        video_grids = inputs.get("video_grid_thw")
        if image_grids is None and video_grids is None:
            return torch.arange(input_ids.shape[1]).view(1, 1, -1).expand(3, 1, -1)
        video_position_grids = None
        if video_grids is not None:
            # Qwen3-VL renders one timestamp/vision span per video frame, so
            # RoPE consumes one t=1 grid for each frame, while the model still
            # receives the original video_grid_thw.
            video_position_grids = torch.repeat_interleave(video_grids, video_grids[:, 0], dim=0).clone()
            video_position_grids[:, 0] = 1

        tokens = input_ids[0].tolist()
        media = []
        image_index = video_index = 0
        token_index = 0
        while token_index < len(tokens):
            token_id = tokens[token_index]
            if token_id == IMAGE_ID:
                if image_grids is None or image_index >= len(image_grids):
                    raise ValueError("image token/grid count mismatch")
                grid = image_grids[image_index]
                image_index += 1
                media.append((token_index, grid))
                _, height, width = (int(value) for value in grid)
                merge = self.processor.image_processor.merge_size
                token_index += (height // merge) * (width // merge)
            elif token_id == VIDEO_ID:
                if video_position_grids is None or video_index >= len(video_position_grids):
                    raise ValueError("video token/grid count mismatch")
                grid = video_position_grids[video_index]
                video_index += 1
                media.append((token_index, grid))
                time, height, width = (int(value) for value in grid)
                merge = self.processor.image_processor.merge_size
                token_index += time * (height // merge) * (width // merge)
            else:
                token_index += 1
        if image_grids is not None and image_index != len(image_grids):
            raise ValueError("image grid/token count mismatch")
        if video_position_grids is not None and video_index != len(video_position_grids):
            raise ValueError("video grid/token count mismatch")

        positions, start = [], 0
        for media_start, grid in media:
            text_length = media_start - start
            base = positions[-1].max() + 1 if positions else 0
            positions.append(torch.arange(text_length).view(1, -1).expand(3, -1) + base)
            time, height, width = (int(value) for value in grid)
            height //= self.processor.image_processor.merge_size
            width //= self.processor.image_processor.merge_size
            temporal = torch.arange(time).view(-1, 1).expand(-1, height * width).flatten()
            rows = torch.arange(height).view(1, -1, 1).expand(time, -1, width).flatten()
            columns = torch.arange(width).view(1, 1, -1).expand(time, height, -1).flatten()
            positions.append(torch.stack([temporal, rows, columns]) + text_length + base)
            start = media_start + time * height * width

        if start < len(tokens):
            base = positions[-1].max() + 1
            positions.append(torch.arange(len(tokens) - start).view(1, -1).expand(3, -1) + base)
        return torch.cat(positions, dim=1).view(3, 1, -1)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        selected = []
        total_length = 0
        for item in batch:
            inputs = self.processor.apply_chat_template(
                item["messages"],
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                do_resize=False,
            )
            length = inputs["input_ids"].shape[1]
            if total_length + length > self.max_length:
                continue

            state = (inputs["input_ids"][0] == STATE_ID).nonzero(as_tuple=False)
            im_starts = (inputs["input_ids"][0] == IM_START_ID).nonzero(as_tuple=False).flatten()
            im_starts = im_starts[im_starts < state[0, 0]] if state.numel() else im_starts[:0]
            if im_starts.numel() == 0:
                raise ValueError("cannot locate the action-conditioning turn")
            inputs["position_ids"] = self._position_ids(inputs)
            payload = {key: value for key, value in item.items() if key != "messages"}
            selected.append((inputs, payload, int(im_starts[-1])))
            total_length += length

        if not selected:
            raise ValueError(f"no training sample fits within MAX_LENGTH={self.max_length}")

        lengths = [inputs["input_ids"].shape[1] for inputs, _, _ in selected]
        offsets = [0] + list(itertools.accumulate(lengths))
        result = {
            "input_ids": torch.cat([inputs["input_ids"] for inputs, _, _ in selected], dim=1),
            "position_ids": torch.cat([inputs["position_ids"] for inputs, _, _ in selected], dim=2),
            "cu_seq_lens_q": torch.tensor(offsets, dtype=torch.int32),
            "cu_seq_lens_k": torch.tensor(offsets, dtype=torch.int32),
            "max_length_q": max(lengths),
            "max_length_k": max(lengths),
            "action_segments": torch.tensor(list(zip(offsets[:-1], offsets[1:])), dtype=torch.long),
            "action_vlm_condition_segments": torch.tensor(
                [[offsets[index], offsets[index] + end] for index, (_, _, end) in enumerate(selected)],
                dtype=torch.long,
            ),
        }
        for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            values = [inputs[key] for inputs, _, _ in selected if inputs.get(key) is not None]
            if values:
                result[key] = torch.cat(values)

        payloads = [payload for _, payload, _ in selected]
        targets = [payload.pop("vlm_action_target") for payload in payloads]
        masks = [payload.pop("vlm_action_mask") for payload in payloads]
        result.update(default_collate(payloads))
        result["vlm_action_target"] = torch.cat(targets)
        result["vlm_action_mask"] = torch.cat(masks)
        return result
