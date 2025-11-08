"""
Tests for Training Pipeline
============================

Tests for Trainer class and training utilities.
"""

import pytest
import torch
import os
from pathlib import Path

from src.training import Trainer, TrainingConfig
from src.models import StatefulLSTM, SequenceLSTM
from src.data import create_train_val_loaders


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()

        assert config.model_type == 'stateful'
        assert config.hidden_size == 64
        assert config.num_layers == 1
        assert config.batch_size == 32
        assert config.num_epochs == 50

    def test_custom_config(self):
        """Test creating custom configuration."""
        config = TrainingConfig(
            model_type='sequence',
            hidden_size=128,
            num_epochs=10
        )

        assert config.model_type == 'sequence'
        assert config.hidden_size == 128
        assert config.num_epochs == 10

    def test_checkpoint_dir_creation(self, temp_dir):
        """Test that checkpoint directory is created."""
        checkpoint_path = os.path.join(temp_dir, 'checkpoints')
        config = TrainingConfig(checkpoint_dir=checkpoint_path)

        assert config.checkpoint_dir == checkpoint_path


class TestTrainer:
    """Tests for Trainer class."""

    def test_trainer_initialization_stateful(self, stateful_model, training_config):
        """Test trainer initialization for stateful model."""
        trainer = Trainer(
            model=stateful_model,
            config=training_config
        )

        assert trainer.model == stateful_model
        assert trainer.config == training_config
        assert str(trainer.device) in ['cuda', 'cpu', 'mps']

    def test_trainer_initialization_sequence(self, sequence_model):
        """Test trainer initialization for sequence model."""
        config = TrainingConfig(
            model_type='sequence',
            hidden_size=32,
            num_layers=2,
            sequence_length=10
        )

        trainer = Trainer(
            model=sequence_model,
            config=config
        )

        assert trainer.model == sequence_model

    def test_model_device_placement(self, stateful_model, training_config, device):
        """Test that model is moved to correct device."""
        trainer = Trainer(
            model=stateful_model,
            config=training_config
        )

        # Check model is on device
        assert next(trainer.model.parameters()).device.type == device


class TestTrainingUtilities:
    """Tests for training utility functions."""

    def test_optimizer_creation(self, stateful_model, training_config):
        """Test that optimizer is created correctly."""
        trainer = Trainer(
            model=stateful_model,
            config=training_config
        )

        assert trainer.optimizer is not None
        assert isinstance(trainer.optimizer, torch.optim.Adam)

    def test_criterion_setup(self, stateful_model, training_config):
        """Test that loss criterion is set up correctly."""
        trainer = Trainer(
            model=stateful_model,
            config=training_config
        )

        assert trainer.criterion is not None
        assert isinstance(trainer.criterion, torch.nn.MSELoss)
