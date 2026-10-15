import torch


class EarlyStopping:
    def __init__(
        self, monitor="val_loss", patience=5, min_delta=0, mode="min", path: str | None = None
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.path = path

        self.counter = 0
        self.early_stop = False

        if self.mode == "min":
            self.best_score = float("inf")
        else:  # mode == 'max'
            self.best_score = -float("inf")

    def __call__(self, val_score, model):
        if self.mode == "min":
            improvement = self.best_score - val_score
        else:
            improvement = val_score - self.best_score

        if improvement > self.min_delta:
            self.best_score = val_score
            self.counter = 0
            if self.path:
                self.save_checkpoint(val_score, model)
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_score, model):
        print(f"Best model saved ({self.monitor}={val_score:.4f}).")
        torch.save(model.state_dict(), self.path)  # type: ignore
