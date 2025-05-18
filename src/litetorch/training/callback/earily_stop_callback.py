from .base import Callback


class EarlyStopCallback(Callback):
    """
    Early stopping callback.
    """
    def __init__(self, patience: int = 10, monitor: str = 'val_loss', mode: str = 'min'):
        """
        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            monitor (str): Metric to monitor. Default is 'val_loss'.
            mode (str): One of {'min', 'max'}. In 'min' mode, training will stop when the quantity monitored has stopped decreasing.
                        In 'max' mode, it will stop when the quantity monitored has stopped increasing.
        """
        super().__init__()
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        
        if self.mode not in ['min', 'max']:
            raise ValueError("mode should be one of {'min', 'max'}")
        
    def on_epoch_end(self, trainer):
        current = getattr(trainer, self.monitor)
        improved = (current < self.best_score) if self.mode == "min" else (current > self.best_score)
        
        if self.best_score is None or improved:
            self.best_score = current
            self.epochs_no_improvement = 0
        else:
            self.epochs_no_improvement += 1
            if self.epochs_no_improvement >= self.patience:
                self.early_stop = True
                print(f"Early stopping triggered after {self.epochs_no_improvement} epochs with no improvement.")
                
    def on_train_end(self, trainer):
        if self.early_stop:
            print(f"Training stopped at epoch {trainer.epoch} due to early stopping.")
        else:
            print("Training completed without early stopping.")
