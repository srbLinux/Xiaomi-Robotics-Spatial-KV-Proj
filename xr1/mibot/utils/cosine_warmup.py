# Copyright (C) 2026 Xiaomi Corporation.
import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    max_lr: float = 4e-4,
    warmup_lr_start: float = 1e-6,
    min_lr: float = 8e-5,
    last_epoch: int = -1,
) -> LambdaLR:
    # LambdaLR multiplies each optimizer group's initial LR by the returned
    # value.  The public arguments below are absolute target LRs, so convert
    # them to a multiplier here.  This preserves the official runner's
    # base-lr=1.0 contract and also makes 2e-5 base-LR Spatial Forcing runs
    # receive an actual 2e-5 peak LR instead of 4e-10.
    base_lr = float(optimizer.param_groups[0]["lr"])
    if base_lr <= 0:
        raise ValueError(f"optimizer base LR must be positive, got {base_lr}")

    def lr_lambda(current_step: int) -> float:
        warmup_lr_start_ = warmup_lr_start
        if current_step < num_warmup_steps:
            if warmup_lr_start_ < 0:
                warmup_lr_start_ = max_lr
            lr = min(
                max_lr,
                warmup_lr_start_ + (max_lr - warmup_lr_start_) * current_step / max(num_warmup_steps, 1),
            )
            return lr / base_lr
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        lr = (max_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)) + min_lr
        return lr / base_lr

    return LambdaLR(optimizer, lr_lambda, last_epoch)
