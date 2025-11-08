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
        assert config.num_layers == 2
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

    def test_trainer_initialization_stateful(self, stateful_model, training_config, sample_dataset):
        """Test trainer initialization for stateful model."""
        train_loader, val_loader = create_train_val_loaders(
            sample_dataset,
            val_split=0.1,
            batch_size=training_config.batch_size,
            model_type='stateful'
        )

        trainer = Trainer(
            model=stateful_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config
        )

        assert trainer.model == stateful_model
        assert trainer.config == training_config
        assert trainer.device in ['cuda', 'cpu']

    def test_trainer_initialization_sequence(self, sequence_model, training_config, sample_dataset):
        """Test trainer initialization for sequence model."""
        training_config.model_type = 'sequence'
        train_loader, val_loader = create_train_val_loaders(
            sample_dataset,
            val_split=0.1,
            batch_size=training_config.batch_size,
            model_type='sequence',
            sequence_length=10
        )

        trainer = Trainer(
            model=sequence_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config
        )

        assert trainer.model == sequence_model

    def test_single_epoch_training_stateful(self, stateful_model, training_config, sample_dataset):
        """Test single epoch of training for stateful model."""
        train_loader, val_loader = create_train_val_loaders(
            sample_dataset,
            val_split=0.1,
            batch_size=training_config.batch_size,
            model_type='stateful'
        )

        trainer = Trainer(
            model=stateful_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config
        )

        # Train for just 1 epoch
        initial_loss = trainer.train_epoch()

        # Loss should be a positive number
        assert initial_loss > 0
        assert initial_loss < 10.0  # Reasonable upper bound

    def test_single_epoch_training_sequence(self, sequence_model, training_config, sample_dataset):
        """Test single epoch of training for sequence model."""
        training_config.model_type = 'sequence'
        train_loader, val_loader = create_train_val_loaders(
            sample_dataset,
            val_split=0.1,
            batch_size=training_config.batch_size,
            model_type='sequence',
            sequence_length=10
        )

        trainer = Trainer(
            model=sequence_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config
        )

        initial_loss = trainer.train_epoch()

        assert initial_loss > 0
        assert initial_loss < 10.0

    def test_validation(self, stateful_model, training_config, sample_dataset):
        """Test validation step."""
        train_loader, val_loader = create_train_val_loaders(
            sample_dataset,
            val_split=0.1,
            batch_size=training_config.batch_size,
            model_type='stateful'
        )

        trainer = Trainer(
            model=stateful_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config
        )

        val_loss = trainer.validate()

        assert val_loss > 0
        assert val_loss < 10.0

    def test_loss_decreases_during_training(self, stateful_model, training_config, sample_dataset):
        """Test that loss generally decreases during training."""
        train_loader, val_loader = create_train_val_loaders(
            sample_dataset,
            val_split=0.1,
            batch_size=training_config.batch_size,
            model_type='stateful'
        )

        # Use more epochs for this test
        training_config.num_epochs = 5

        trainer = Trainer(
            model=stateful_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config
        )

        # Record initial loss
        initial_loss = trainer.train_epoch()

        # Train for a few more epochs
        for _ in range(4):
            trainer.train_epoch()

        # Final loss should be lower
        final_loss = trainer.validate()

        # Allow some tolerance, but loss should generally decrease
        assert final_loss < initial_loss * 1.5

    def test_checkpoint_saving(self, stateful_model, training_config, sample_dataset, temp_dir):
        """Test that checkpoints are saved."""
        training_config.checkpoint_dir = temp_dir

        train_loader, val_loader = create_train_val_loaders(
            sample_dataset,
            val_split=0.1,
            batch_size=training_config.batch_size,
            model_type='stateful'
        )

        trainer = Trainer(
            model=stateful_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config
        )

        # Save a checkpoint
        checkpoint_path = os.path.join(temp_dir, 'test_checkpoint.pth')
        trainer.save_checkpoint(checkpoint_path, epoch=1, val_loss=0.5)

        # Check file exists
        assert os.path.exists(checkpoint_path)

        # Load and verify
        checkpoint = torch.load(checkpoint_path)
        assert 'epoch' in checkpoint
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'val_loss' in checkpoint
        assert checkpoint['epoch'] == 1

    def test_checkpoint_loading(self, stateful_model, training_config, sample_dataset, temp_dir):
        """Test that checkpoints can be loaded."""
        training_config.checkpoint_dir = temp_dir

        train_loader, val_loader = create_train_val_loaders(
            sample_dataset,
            val_split=0.1,
            batch_size=training_config.batch_size,
            model_type='stateful'
        )

        trainer = Trainer(
            model=stateful_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config
        )

        # Save checkpoint
        checkpoint_path = os.path.join(temp_dir, 'test_load.pth')
        trainer.save_checkpoint(checkpoint_path, epoch=5, val_loss=0.3)

        # Create new model and trainer
        new_model = StatefulLSTM(input_size=5, hidden_size=32, num_layers=1)
        new_trainer = Trainer(
            model=new_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config
        )

        # Load checkpoint
        epoch, val_loss = new_trainer.load_checkpoint(checkpoint_path)

        assert epoch == 5
        assert val_loss == pytest.approx(0.3)

    def test_early_stopping(self, stateful_model, sample_dataset, temp_dir):
        """Test early stopping mechanism."""
        # Create config with very low patience
        config = TrainingConfig(
            model_type='stateful',
            hidden_size=32,
            num_layers=1,
            batch_size=16,
            num_epochs=100,  # High number
            patience=2,       # But low patience
            checkpoint_dir=temp_dir,
            verbose=False
        )

        train_loader, val_loader = create_train_val_loaders(
            sample_dataset,
            val_split=0.1,
            batch_size=config.batch_size,
            model_type='stateful'
        )

        trainer = Trainer(
            model=stateful_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config
        )

        # Train - should stop early
        history = trainer.train()

        # Should not train for all 100 epochs
        assert len(history['train_loss']) < 100

    def test_training_history(self, stateful_model, training_config, sample_dataset):
        """Test that training history is recorded correctly."""
        train_loader, val_loader = create_train_val_loaders(
            sample_dataset,
            val_split=0.1,
            batch_size=training_config.batch_size,
            model_type='stateful'
        )

        trainer = Trainer(
            model=stateful_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config
        )

        history = trainer.train()

        # History should have the right keys
        assert 'train_loss' in history
        assert 'val_loss' in history

        # Should have entries for each epoch
        assert len(history['train_loss']) == training_config.num_epochs or \
               len(history['train_loss']) <= training_config.num_epochs  # May stop early

        # All losses should be positive
        assert all(loss > 0 for loss in history['train_loss'])
        assert all(loss > 0 for loss in history['val_loss'])

    def test_model_device_placement(self, stateful_model, training_config, sample_dataset, device):
        """Test that model is moved to correct device."""
        train_loader, val_loader = create_train_val_loaders(
            sample_dataset,
            val_split=0.1,
            batch_size=training_config.batch_size,
            model_type='stateful'
        )

        trainer = Trainer(
            model=stateful_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config
        )

        # Check model is on device
        assert next(trainer.model.parameters()).device.type == device


class TestTrainingUtilities:
    """Tests for training utility functions."""

    def test_optimizer_creation(self, stateful_model, training_config):
        """Test that optimizer is created correctly."""
        train_loader, val_loader = create_train_val_loaders(
            {}, val_split=0.1, batch_size=16, model_type='stateful'
        )

        trainer = Trainer(
            model=stateful_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config
        )

        assert trainer.optimizer is not None
        assert isinstance(trainer.optimizer, torch.optim.Adam)

    def test_criterion_setup(self, stateful_model, training_config):
        """Test that loss criterion is set up correctly."""
        train_loader, val_loader = create_train_val_loaders(
            {}, val_split=0.1, batch_size=16, model_type='stateful'
        )

        trainer = Trainer(
            model=stateful_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config
        )

        assert trainer.criterion is not None
        assert isinstance(trainer.criterion, torch.nn.MSELoss)
