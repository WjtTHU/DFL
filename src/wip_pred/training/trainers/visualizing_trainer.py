import os
from copy import deepcopy

import torch

from wip_pred.utils.visualize import plot_true_vs_pred

from .reg_trainer import RegTrainer


class VisualizingTrainer(RegTrainer):
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device,
        target_scaler,
        log_dir,
        scheduler=None,
        metrics=None,
        use_logger=True,
        use_tensorboard=True,
        log_file="training.log",
        tb_logdir="tb_logs",
        save_dir: str | None = None,
        early_stopping=None,
    ):
        super().__init__(
            model,
            criterion,
            optimizer,
            device,
            target_scaler,
            scheduler=scheduler,
            metrics=metrics,
            use_logger=use_logger,
            use_tensorboard=use_tensorboard,
            log_dir=log_dir,
            log_file=log_file,
            tb_logdir=tb_logdir,
            save_dir=save_dir,
            early_stopping=early_stopping,
        )

    def run(
        self,
        train_loader,
        val_loader,
        n_epoch,
        ids_to_pred_in_val=None,
        id_groups: dict | None = None,
        plot_interval: int = 5,
    ):
        best_val_loss = float("inf")

        # For now, only samples in val_loader can be plotted
        for epoch in range(1, n_epoch + 1):
            train_loss, train_metrics = self.train_one_epoch(
                train_loader, epoch=epoch, collect_details=True
            )
            for metrics, value in train_metrics.items():
                self.history[f"train_{metrics}"].append(value)
            train_metric_str = self.get_metric_str(train_metrics, prefix="Train ")
            self.history["train_loss"].append(train_loss)

            val_loss, val_metrics = self.validate(val_loader, ids_to_pred_in_val, True)
            for metrics, value in val_metrics.items():
                self.history[f"val_{metrics}"].append(value)
            val_metric_str = self.get_metric_str(val_metrics, prefix="Val ")
            self.history["val_loss"].append(val_loss)

            # log
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

            # write to TensorBoard
            if self.writer:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)

            # plot
            if epoch % plot_interval == 0 or best_val_loss == val_loss:
                if epoch % plot_interval == 0:
                    fig_path = f"val_pred_epoch{epoch}.png"
                else:
                    fig_path = "val_best_pred.png"
                all_cfg = {
                    "x": self.current_epoch_predictions["targets"],
                    "y": self.current_epoch_predictions["outputs"],
                    "alpha": 0,
                    "label": "All",
                }
                cfgs = [all_cfg]
                if id_groups:
                    for name, ids in id_groups.items():
                        mask = self.current_epoch_predictions["ids"].isin(ids).values  # type: ignore
                        new_cfg = {
                            "x": self.current_epoch_predictions["targets"][mask],
                            "y": self.current_epoch_predictions["outputs"][mask],
                            "alpha": 0.5,
                            "label": name,
                        }
                        cfgs.append(new_cfg)
                else:
                    cfgs[0]["alpha"] = 1

                plot_true_vs_pred(
                    self.current_epoch_predictions["targets"],
                    cfgs,
                    title=f"Validation Predictions (Epoch {epoch})",
                    save_path=os.path.join(self.log_dir, fig_path),
                )  # type: ignore

        if self.writer:
            self.writer.close()
