from collections import defaultdict

import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from .base_trainer import BaseTrainer


def masked_label_smoothing_loss(pred, target, mask, smoothing=0.1):
    """
    Masked label smoothing loss using PyTorch's built-in CrossEntropyLoss with label_smoothing.

    Args:
        pred (Tensor): (B, L, C) model logits
        target (Tensor): (B, L) ground-truth class indices
        mask (Tensor): (B, L), 0 = valid token, 1 = masked/padded token
        smoothing (float): label smoothing factor between 0 and 1
    """
    B, L, C = pred.shape
    valid_mask = (mask == 0).float()  # (B, L), 1 for valid tokens
    valid_sum = valid_mask.sum()

    # Built-in CrossEntropyLoss supports label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=smoothing, reduction="none")

    # CrossEntropyLoss expects input shape (N, C, *), so transpose
    loss = criterion(pred.transpose(1, 2), target)  # (B, L)

    # Apply mask and normalize
    loss = (loss * valid_mask).sum() / (valid_sum + 1e-8)

    # # Optional monitoring
    # probs = F.softmax(pred, dim=-1)  # (B, L, C)
    # correct_prob = probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (B, L)
    # mean_prob = correct_prob[mask == 0].mean().item()

    # print(f"loss={loss.item():.3f}, mean_prob={mean_prob:.3f}")

    return loss, valid_sum


def multi_task_loss(preds, targets, masks, weights=None):
    total_loss = None  # not total_loss = 0.0 or the computational graph will be lost
    losses = {}
    valid_nums = {}

    for task_name, pred in preds.items():
        target = targets[task_name]
        mask = masks[task_name]
        loss_i, valid_num = masked_label_smoothing_loss(pred, target, mask)
        w = 1.0 if weights is None else weights.get(task_name, 1.0)

        # accumulate loss
        total_loss = loss_i * w if total_loss is None else total_loss + loss_i * w

        # save each loss and valid sample num
        losses[task_name] = loss_i.detach().item()
        valid_nums[task_name] = valid_num

    return total_loss, losses, valid_nums


class Pretrainer(BaseTrainer):
    def __init__(
        self,
        model,
        optimizer,
        device,
        scheduler=None,
        metrics=None,
        use_logger=True,
        use_tensorboard=True,
        log_dir: str | None = None,
        log_file="training.log",
        tb_logdir="tb_logs",
        save_dir: str | None = None,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
            metrics=metrics,
            use_logger=use_logger,
            use_tensorboard=use_tensorboard,
            log_dir=log_dir,
            log_file=log_file,
            tb_logdir=tb_logdir,
            save_dir=save_dir,
        )

    def train_one_epoch(self, dataloader, epoch):
        self.model.train()
        running_loss = defaultdict(float)
        total_samples = defaultdict(int)

        n_step = 0

        for _ids, inputs, targets, target_masks in tqdm(
            dataloader, ncols=80, desc="Training", leave=False
        ):
            n_step += 1
            for task, target in targets.items():
                targets[task] = target.to(self.device)
            for task, target_mask in target_masks.items():
                target_masks[task] = target_mask.to(self.device)

            preds = self.model(inputs)

            total_loss, loss_dict, valid_dict = multi_task_loss(preds, targets, target_masks)

            self.optimizer.zero_grad()
            total_loss.backward()  # type: ignore

            max_norm = 1.0
            total_norm = clip_grad_norm_(self.model.parameters(), max_norm=max_norm)

            if self.writer:
                self.writer.add_scalar(f"grad_norm/epoch_{epoch}", total_norm, n_step)

            self.optimizer.step()

            for task in targets:
                running_loss[task] += loss_dict[task] * valid_dict[task]
                total_samples[task] += valid_dict[task]

        if self.scheduler:
            self.scheduler.step()

        avg_loss = {task: running_loss[task] / total_samples[task] for task in targets}

        return avg_loss

    def run(self, train_loader, n_epoch):
        for epoch in range(1, n_epoch + 1):
            avg_loss = self.train_one_epoch(train_loader, epoch)

            loss_str = " | ".join(f"{task} Loss: {avg_loss[task]}" for task in avg_loss)

            if self.logger:
                self.logger.info(f"Epoch {epoch}/{n_epoch} - " + loss_str)

            # write to TensorBoard
            if self.writer:
                for task in avg_loss:
                    self.writer.add_scalar(f"Loss/{task}_Loss", avg_loss[task], epoch)

        if self.writer:
            self.writer.close()
