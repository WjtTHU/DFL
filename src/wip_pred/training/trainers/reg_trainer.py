import os
from copy import deepcopy

import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from .base_trainer import BaseTrainer


class RegTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device,
        target_scaler: tuple[float, float],
        scheduler=None,
        metrics=None,
        use_logger=True,
        use_tensorboard=True,
        log_dir: str | None = None,
        log_file="training.log",
        tb_logdir="tb_logs",
        save_dir: str | None = None,
        early_stopping=None,
    ):
        super().__init__(
            model,
            optimizer,
            device,
            criterion,
            scheduler=scheduler,
            metrics=metrics,
            use_logger=use_logger,
            use_tensorboard=use_tensorboard,
            log_dir=log_dir,
            log_file=log_file,
            tb_logdir=tb_logdir,
            save_dir=save_dir,
        )

        self.min_target = target_scaler[0]
        self.max_target = target_scaler[1]
        self.best_model_weights = None

        # --------- History ---------
        self.history = {"train_loss": [], "val_loss": []}
        if metrics:
            for name in metrics:
                self.history[f"train_{name}"] = []
                self.history[f"val_{name}"] = []

        if early_stopping:
            if early_stopping.monitor not in self.history:
                raise ValueError("'monitor' of EarlyStopping is not recorded in history!")

        self.early_stopping = early_stopping

        # --------- Attribute for storing current epoch predictions ---------
        self.current_epoch_predictions = {"outputs": [], "targets": [], "ids": []}

    def _collect_details(self, ids, targets, outputs, has_label):
        has_label = has_label.squeeze(1)
        self.current_epoch_predictions["ids"].extend(ids[has_label.cpu().numpy()].tolist())
        self.current_epoch_predictions["targets"].append(targets[has_label].squeeze(1))
        self.current_epoch_predictions["outputs"].append(outputs[has_label].squeeze(1))

    def collect_all_details_in_an_epoch(self, metric_results):
        self.current_epoch_predictions["ids"] = pd.Series(self.current_epoch_predictions["ids"])  # type: ignore
        self.current_epoch_predictions["targets"] = (
            torch.cat(self.current_epoch_predictions["targets"]).detach().cpu().numpy()
            * (self.max_target - self.min_target)
            + self.min_target
        )  # type: ignore
        self.current_epoch_predictions["outputs"] = (
            torch.cat(self.current_epoch_predictions["outputs"]).detach().cpu().numpy()
            * (self.max_target - self.min_target)
            + self.min_target
        )  # type: ignore
        for name, fn in self.metrics.items():
            metric_results[name] = fn(
                self.current_epoch_predictions["targets"], self.current_epoch_predictions["outputs"]
            )

    def train_one_epoch(self, dataloader, collect_details=False, epoch=0):
        self.model.train()
        running_loss = 0.0
        total_samples = 0

        self.current_epoch_predictions["outputs"] = []
        self.current_epoch_predictions["targets"] = []
        self.current_epoch_predictions["ids"] = []

        n_step = 0

        for ids, inputs, targets, target_mask in tqdm(
            dataloader, ncols=80, desc="Training", leave=False
        ):
            n_step += 1
            targets = targets.to(self.device).view(-1, 1)

            outputs = self.model(inputs).view(-1, 1)

            has_label = target_mask.to(self.device).view(-1, 1)

            loss = self.criterion(outputs, targets)  # type: ignore
            masked_loss = (loss * has_label).sum() / (has_label.sum() + 1e-8)

            self.optimizer.zero_grad()
            masked_loss.backward()

            max_norm = 1.0
            total_norm = clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
            if self.writer:
                self.writer.add_scalar(f"grad_norm/epoch_{epoch}", total_norm, n_step)
            # if total_norm > max_norm:
            #     print(f"Gradient Clipped")
            #     print(f"   norm before clipping: {total_norm.item():.4f} > max norm: {max_norm}")

            self.optimizer.step()

            sample_num = has_label.sum().item()
            running_loss += masked_loss.item() * sample_num
            total_samples += sample_num

            if collect_details:
                self._collect_details(ids, targets, outputs, has_label)

        metric_results = {}

        if collect_details:
            self.collect_all_details_in_an_epoch(metric_results)

        if self.scheduler:
            self.scheduler.step()

        avg_loss = running_loss / total_samples

        return avg_loss, metric_results

    def validate(self, dataloader, ids_to_pred=None, collect_details=False):
        self.model.eval()
        running_loss = 0.0
        total_samples = 0

        self.current_epoch_predictions["outputs"] = []
        self.current_epoch_predictions["targets"] = []
        self.current_epoch_predictions["ids"] = []

        with torch.no_grad():
            for ids, inputs, targets, target_mask in tqdm(
                dataloader, ncols=80, desc="Validating", leave=False
            ):
                targets = targets.to(self.device).view(-1, 1)

                outputs = self.model(inputs).view(-1, 1)
                # has label or not
                if ids_to_pred is None:
                    has_label = target_mask.to(self.device).view(-1, 1)
                else:
                    has_label = torch.tensor(
                        ids.isin(ids_to_pred), dtype=torch.bool, device=self.device
                    ).view(-1, 1)

                loss = self.criterion(outputs, targets)  # type: ignore
                masked_loss = (loss * has_label).sum()

                running_loss += masked_loss.item()
                total_samples += has_label.sum().item()

                if collect_details:
                    self._collect_details(ids, targets, outputs, has_label)

        metric_results = {}

        if collect_details:
            self.collect_all_details_in_an_epoch(metric_results)

        avg_loss = running_loss / total_samples

        return avg_loss, metric_results

    @staticmethod
    def get_metric_str(metric_results, prefix=""):
        if metric_results:
            metric_str = " | ".join(f"{prefix}{k}: {v:.4f}" for k, v in metric_results.items())
            metric_str = f" | {metric_str}"
        else:
            metric_str = ""

        return metric_str

    def run(
        self,
        train_loader,
        val_loader,
        n_epoch,
        ids_to_pred_in_val=None,
        val_interval=1,
        collect_train_details=False,
        collect_val_details=False,
    ):
        best_val_loss = float("inf")

        for epoch in range(1, n_epoch + 1):
            train_loss, train_metrics = self.train_one_epoch(
                train_loader, collect_train_details, epoch=epoch
            )
            for metrics, value in train_metrics.items():
                self.history[f"train_{metrics}"].append(value)
            train_metric_str = self.get_metric_str(train_metrics, prefix="Train ")

            self.history["train_loss"].append(train_loss)

            if epoch % val_interval == 0:
                val_loss, val_metrics = self.validate(
                    val_loader, ids_to_pred_in_val, collect_val_details
                )
                for metrics, value in val_metrics.items():
                    self.history[f"val_{metrics}"].append(value)
                val_metric_str = self.get_metric_str(val_metrics, prefix="Val ")
                self.history["val_loss"].append(val_loss)

                if self.logger:
                    self.logger.info(
                        f"Epoch {epoch}/{n_epoch} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
                        + train_metric_str
                        + val_metric_str
                    )

                if self.early_stopping:
                    self.early_stopping(self.history[self.early_stopping.monitor][-1], self.model)
                    if self.early_stopping.early_stop:
                        print("Early stopping triggered!")
                        break

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_model_weights = deepcopy(self.model.state_dict())
                    if not self.early_stopping and self.save_dir:
                        save_path = os.path.join(self.save_dir or ".", "best_model.pth")
                        torch.save(self.model.state_dict(), save_path)
                        if self.logger:
                            self.logger.info(
                                f"Best model saved (epoch={epoch}, val_loss={best_val_loss:.4f})."
                            )

            else:
                if self.logger:
                    self.logger.info(
                        f"Epoch {epoch}/{n_epoch} - Train Loss: {train_loss:.4f}" + train_metric_str
                    )

            # write to TensorBoard
            if self.writer:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                if epoch % val_interval == 0:
                    self.writer.add_scalar("Loss/val", val_loss, epoch)

        if self.writer:
            self.writer.close()
