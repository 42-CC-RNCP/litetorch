from typing import Dict
from tensorboardX import SummaryWriter
from litetorch.metrics.base import ScalarMetric, FigureMetric
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
        grouped_metrics = {}

        for name, fn in self.metrics.items():
            try:
                if isinstance(fn, ScalarMetric):
                    value = fn(trainer)
                    if "/" in name:
                        tag, sub = name.split("/")
                        if tag not in grouped_metrics:
                            grouped_metrics[tag] = {}
                        grouped_metrics[tag][sub] = value
                    else:
                        self.writer.add_scalar(name, value, epoch)
                elif isinstance(fn, FigureMetric):
                    figure = fn(trainer)
                    self.writer.add_figure(name, figure, epoch)
            except Exception as e:
                print(f"[TensorBoardLogger] Failed to log {name}: {e}")

        # Now log grouped scalars
        for tag, sub_metrics in grouped_metrics.items():
            self.writer.add_scalars(tag, sub_metrics, epoch)

    def on_train_end(self, trainer):
        self.writer.close()
        print(f"[TensorBoardLogger] Saved logs to {self.log_dir}")
