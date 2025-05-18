from typing import List
from tensorboardX import SummaryWriter
from .base import Callback


class TensorboardLoggerCallback(Callback):
    """
    Tensorboard logger callback.
    """
    def __init__(self, metrics: List[str], log_dir: str = "runs", flush_secs: int = 30):
        """
        Args:
            metrics (List[str]): List of metrics to log.
            log_dir (str): Directory to save the logs. Default is "runs".
            flush_secs (int): How often to flush the logs. Default is 30 seconds.
        """
        super().__init__()
        self.metrics = metrics
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs)
        
    def on_epoch_end(self, trainer):
        """
        Logs the metrics at the end of each epoch.
        """
        for metric in self.metrics:
            if hasattr(trainer, metric):
                value = getattr(trainer, metric)[-1]
                self.writer.add_scalar(metric, value, trainer.epoch)
            else:
                print(f"Metric '{metric}' not found in trainer.")
        self.writer.flush()
        
    def on_train_end(self, trainer):
        """
        Closes the writer at the end of training.
        """
        self.writer.close()
        print("Tensorboard logs saved.")
