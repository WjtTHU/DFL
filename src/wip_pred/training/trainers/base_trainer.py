import logging
import os
from pathlib import Path

from torch.utils.tensorboard.writer import SummaryWriter


class BaseTrainer:
    def __init__(
        self,
        model,
        optimizer,
        device,
        criterion=None,
        scheduler=None,
        metrics=None,
        use_logger=True,
        use_tensorboard=True,
        log_dir: str | None = None,
        log_file="training.log",
        tb_logdir="tb_logs",
        save_dir: str | None = None,
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.log_dir = log_dir
        self.save_dir = save_dir
        self.metrics = metrics or {}  # dict，e.g. {'MAE': mean_absolute_error, 'R2': r2_score}

        # --------- Create Path ---------
        if log_dir:
            if Path(log_dir).is_dir():  # ensure the path exists
                log_file = os.path.join(log_dir, log_file)
                tb_logdir = os.path.join(log_dir, tb_logdir)
            else:
                raise FileNotFoundError(f"Log dir does not exist: {log_dir}")

        if save_dir:
            if not Path(save_dir).is_dir():
                raise FileNotFoundError(f"Save dir does not exist: {save_dir}")

        # --------- Logger setting ---------
        self.logger = None
        if use_logger:
            self.logger = logging.getLogger(f"Trainer_{id(self)}")  # unique name per instance
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False

            # clear handlers
            if self.logger.hasHandlers():
                self.logger.handlers.clear()

            # console output
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            )
            self.logger.addHandler(console_handler)

            # file output
            if log_dir is not None:
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(
                    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
                )
                self.logger.addHandler(file_handler)

            self.logger.info(f"Log dir: {self.log_dir}")
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.logger.info(f"Trainable Params: {trainable_params}")

        # --------- TensorBoard ---------
        self.writer = SummaryWriter(tb_logdir) if use_tensorboard and log_dir else None

    def train_one_epoch(self, dataloader):
        pass

    def validate(self, dataloader):
        pass

    def run(self, train_loader, val_loader, n_epoch):
        pass
