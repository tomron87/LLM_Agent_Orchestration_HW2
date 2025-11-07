"""
Training Module
===============

This module handles:
- Training configuration
- Training loops for both L=1 and L>1 models
- Optimizer and learning rate scheduling
- Early stopping and checkpointing
- Training history tracking
"""

from .config import TrainingConfig
from .trainer import Trainer, train_model

__all__ = ['TrainingConfig', 'Trainer', 'train_model']
