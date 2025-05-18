import seaborn as sns
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklite.metrics import ConfusionMatrix
from litetorch.core.tensor import Tensor
from .base import FigureMetric


class ConfusionMatrixImage(FigureMetric):
    """
    Confusion Matrix metric for visualizing the performance of a classification model.
    """
    
    def __init__(self):
        super().__init__()
        # self.class_names = ["True Negative", "False Positive",
        #                     "False Negative", "True Positive"]
        self.class_names = [["Positive", "Negative"],
                            ["Positive", "Negative"]]

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
                    xticklabels=self.class_names[0],
                    yticklabels=self.class_names[1],
                    ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")

        plt.tight_layout()
        return fig
