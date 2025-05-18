from typing import Dict
from tensorboardX import SummaryWriter
from .base import Callback


class TensorboardLoggerCallback(Callback):
    """
    Tensorboard logger callback.
    """
    def __init__(self, metrics: Dict, log_dir: str = "runs", flush_secs: int = 30):
        """
        Args:
            metrics (List[str]): List of metrics to log.
            log_dir (str): Directory to save the logs. Default is "runs".
            flush_secs (int): How often to flush the logs. Default is 30 seconds.
        """
        super().__init__()
        self.metrics = metrics
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs)
        
    def on_epoch_end(self, trainer):
        epoch = trainer.epoch + 1
        for name, fn in self.metrics.items():
            try:
                value = fn(trainer)
                self.writer.add_scalar(name, value, epoch)
            except Exception as e:
                print(f"[TensorBoardLogger] Failed to log {name}: {e}")

    def on_train_end(self, trainer):
        self.writer.close()
        print(f"[TensorBoardLogger] Saved logs to {self.log_dir}")
