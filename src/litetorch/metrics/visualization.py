import seaborn as sns
import numpy as np
import matplotlib
from typing import Optional, List
from matplotlib import pyplot as plt
from sklite.metrics import ConfusionMatrix
from litetorch.core.tensor import Tensor
from .base import FigureMetric


class ConfusionMatrixImage(FigureMetric):
    """
    Confusion Matrix metric for visualizing the performance of a classification model.
    """
    
    def __init__(self, class_names=Optional[List[str]]):
        super().__init__()
        self.class_names = ["Negative", "Positive"]  # if label 0 = Negative, 1 = Positive
        if class_names is not None:
            self.class_names = class_names

    def __call__(self, trainer) -> "matplotlib.figure.Figure":
        """
        Calculate the confusion matrix and return it as a matplotlib figure.

        Parameters
        ----------
        trainer : Trainer
            The trainer object containing the model and data loaders.

        Returns
        -------
        matplotlib.figure.Figure
            The confusion matrix figure.
        """
        y_true, y_pred = np.array([]), np.array([])
        for X_batch, y_batch in trainer.val_loader:
            X_batch: Tensor
            y_batch: Tensor
            probs = trainer.model(X_batch).data.ravel()
            preds = (probs >= 0.5).astype(int)
            y_true = np.concatenate((y_true, y_batch.data.ravel()))
            y_pred = np.concatenate((y_pred, preds))

        cm = ConfusionMatrix()(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")

        plt.tight_layout()
        return fig
