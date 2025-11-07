"""
Data Generation and Management Module
======================================

This module handles:
- Synthetic signal generation with random noise
- Dataset creation for training and testing
- PyTorch Dataset and DataLoader utilities
"""

from .signal_generator import SignalGenerator, generate_dataset
from .dataset import FrequencyExtractionDataset, SequenceDataset
from .data_loader import create_dataloader, create_train_val_loaders, create_test_loader

__all__ = [
    'SignalGenerator',
    'generate_dataset',
    'FrequencyExtractionDataset',
    'SequenceDataset',
    'create_dataloader',
    'create_train_val_loaders',
    'create_test_loader'
]
