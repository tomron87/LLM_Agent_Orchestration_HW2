"""
Pytest Configuration and Fixtures
==================================

Shared fixtures for all tests.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import SignalGenerator, generate_dataset
from src.models import StatefulLSTM, SequenceLSTM
from src.training import TrainingConfig


@pytest.fixture
def frequencies():
    """Standard frequency list."""
    return [1.0, 3.0, 5.0, 7.0]


@pytest.fixture
def sampling_rate():
    """Standard sampling rate."""
    return 1000


@pytest.fixture
def duration():
    """Standard signal duration."""
    return 10.0


@pytest.fixture
def signal_generator(frequencies, sampling_rate, duration):
    """Create a SignalGenerator instance."""
    return SignalGenerator(
        frequencies=frequencies,
        fs=sampling_rate,
        duration=duration,
        seed=42
    )


@pytest.fixture
def sample_dataset(signal_generator):
    """Generate a small sample dataset."""
    return signal_generator.generate_dataset()


@pytest.fixture
def stateful_model():
    """Create a StatefulLSTM model for testing."""
    return StatefulLSTM(
        input_size=5,
        hidden_size=32,  # Smaller for faster tests
        num_layers=1
    )


@pytest.fixture
def sequence_model():
    """Create a SequenceLSTM model for testing."""
    return SequenceLSTM(
        input_size=5,
        hidden_size=32,
        num_layers=2,
        sequence_length=10
    )


@pytest.fixture
def sample_input():
    """Create a sample input tensor."""
    return torch.randn(4, 5)  # Batch of 4, input size 5


@pytest.fixture
def sample_sequence_input():
    """Create a sample sequence input tensor."""
    return torch.randn(4, 10, 5)  # Batch of 4, sequence length 10, input size 5


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def training_config(temp_dir):
    """Create a test training configuration."""
    return TrainingConfig(
        model_type='stateful',
        hidden_size=32,
        num_layers=1,
        batch_size=16,
        num_epochs=2,  # Very few epochs for fast testing
        learning_rate=0.01,
        patience=1,
        checkpoint_dir=temp_dir,
        verbose=False
    )


@pytest.fixture
def device():
    """Get available device (cuda, mps, or cpu)."""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducibility in tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
