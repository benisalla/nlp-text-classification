"""Emotion classification of short English texts with RNN / LSTM / GRU in PyTorch."""

from .config import CLASSES, Config, DataConfig, ModelConfig, TrainConfig
from .model import TextClassifier

__all__ = ["CLASSES", "Config", "DataConfig", "ModelConfig", "TextClassifier", "TrainConfig"]
__version__ = "0.1.0"
