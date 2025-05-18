import matplotlib
from abc import ABC, abstractmethod


class Metric(ABC):
    """
    Abstract base class for all metrics.
    """

    @abstractmethod
    def __call__(self, trainer):
        pass
    
    def __str__(self):
        return self.__class__.__name__


class ScalarMetric(Metric):
    @abstractmethod
    def __call__(self, trainer) -> float:
        pass


class FigureMetric(Metric):
    @abstractmethod
    def __call__(self, trainer) -> "matplotlib.figure.Figure":
        pass
