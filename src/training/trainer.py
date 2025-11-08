"""
Trainer Module
==============

Main training loop implementation for both L=1 stateful and L>1 sequence models.

The Trainer class handles:
- Training and validation loops
- State management for L=1 models
- Loss computation and backpropagation
- Gradient clipping
- Early stopping
- Model checkpointing
- Training history tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import os
import time
from pathlib import Path

from .config import TrainingConfig


class EarlyStopping:
    """
    Early stopping handler to stop training when validation loss stops improving.

    Args:
        patience: Number of epochs with no improvement before stopping
        min_delta: Minimum change to qualify as improvement
        verbose: Print messages
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Current validation loss

        Returns:
            should_stop: True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            # Improvement
            if self.verbose:
                print(f"   Validation loss improved: {self.best_loss:.6f} → {val_loss:.6f}")
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f"   No improvement for {self.counter}/{self.patience} epochs")

            if self.counter >= self.patience:
                if self.verbose:
                    print(f"   Early stopping triggered!")
                self.early_stop = True
                return True

        return False


class Trainer:
    """
    Main trainer class for LSTM models.

    Handles training for both stateful (L=1) and sequence (L>1) models
    with proper state management and tracking.

    Attributes:
        model: LSTM model to train
        config: Training configuration
        device: Device for training (cuda/cpu/mps)
        optimizer: Optimizer instance
        criterion: Loss function
        scheduler: Learning rate scheduler (optional)
        early_stopping: Early stopping handler
        history: Training history dictionary
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig
    ):
        """
        Initialize trainer.

        Args:
            model: LSTM model (StatefulLSTM or SequenceLSTM)
            config: Training configuration
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.device)

        # Move model to device
        self.model.to(self.device)

        # Initialize optimizer
        if config.optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer_type}")

        # Loss function
        self.criterion = nn.MSELoss()

        # Learning rate scheduler (ReduceLROnPlateau)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        # Early stopping
        if config.use_early_stopping:
            self.early_stopping = EarlyStopping(
                patience=config.patience,
                min_delta=config.min_delta,
                verbose=config.verbose
            )
        else:
            self.early_stopping = None

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_times': []
        }

        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def train_epoch_stateful(
        self,
        train_loader: DataLoader
    ) -> float:
        """
        Train one epoch for stateful L=1 model.

        CRITICAL: Properly manages LSTM internal state between samples.

        Args:
            train_loader: Training data loader

        Returns:
            avg_loss: Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        current_freq_idx = None

        for batch_idx, batch in enumerate(train_loader):
            inputs = batch['input'].to(self.device)  # (batch, 5)
            targets = batch['target'].to(self.device)  # (batch, 1)
            freq_idx = batch['freq_idx']  # (batch,)
            sample_idx = batch['sample_idx']  # (batch,)

            # Check if we need to reset state (new frequency or start of sequence)
            # For L=1 stateful, reset when:
            # 1. Frequency changes
            # 2. Sample index is 0 (start of frequency sequence)
            reset_state = False
            if current_freq_idx is None or freq_idx[0].item() != current_freq_idx:
                reset_state = True
                current_freq_idx = freq_idx[0].item()
            elif sample_idx[0].item() == 0:
                reset_state = True

            # Forward pass
            outputs = self.model(inputs, reset_state=reset_state)

            # Compute loss
            loss = self.criterion(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Detach states after backward to truncate BPTT and free computational graph
            # Gradients have already flowed through, so this doesn't affect learning
            # but prevents "backward through graph a second time" errors
            if hasattr(self.model, 'hidden_state') and self.model.hidden_state is not None:
                self.model.hidden_state = self.model.hidden_state.detach()
                self.model.cell_state = self.model.cell_state.detach()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm
            )

            # Optimizer step
            self.optimizer.step()

            # Track loss
            total_loss += loss.item()
            num_batches += 1

            # Logging
            if self.config.verbose and (batch_idx + 1) % self.config.log_interval == 0:
                print(f"      Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.6f}")

        avg_loss = total_loss / num_batches
        return avg_loss

    def train_epoch_sequence(
        self,
        train_loader: DataLoader
    ) -> float:
        """
        Train one epoch for sequence L>1 model.

        Args:
            train_loader: Training data loader

        Returns:
            avg_loss: Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            inputs = batch['input'].to(self.device)  # (batch, L, 5)
            targets = batch['target'].to(self.device)  # (batch, L, 1)

            # Forward pass
            outputs = self.model(inputs)  # (batch, L, 1)

            # Compute loss over all timesteps
            loss = self.criterion(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm
            )

            # Optimizer step
            self.optimizer.step()

            # Track loss
            total_loss += loss.item()
            num_batches += 1

            # Logging
            if self.config.verbose and (batch_idx + 1) % self.config.log_interval == 0:
                print(f"      Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.6f}")

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def validate_stateful(
        self,
        val_loader: DataLoader
    ) -> float:
        """
        Validate stateful L=1 model.

        Args:
            val_loader: Validation data loader

        Returns:
            avg_loss: Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        current_freq_idx = None

        for batch in val_loader:
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            freq_idx = batch['freq_idx']
            sample_idx = batch['sample_idx']

            # Reset state when needed
            reset_state = False
            if current_freq_idx is None or freq_idx[0].item() != current_freq_idx:
                reset_state = True
                current_freq_idx = freq_idx[0].item()
            elif sample_idx[0].item() == 0:
                reset_state = True

            # Forward pass
            outputs = self.model(inputs, reset_state=reset_state)

            # Compute loss
            loss = self.criterion(outputs, targets)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def validate_sequence(
        self,
        val_loader: DataLoader
    ) -> float:
        """
        Validate sequence L>1 model.

        Args:
            val_loader: Validation data loader

        Returns:
            avg_loss: Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)

            # Forward pass
            outputs = self.model(inputs)

            # Compute loss
            loss = self.criterion(outputs, targets)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict:
        """
        Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader

        Returns:
            history: Training history dictionary
        """
        print(f"\nStarting training on {self.device}...")
        print(f"Model type: {self.config.model_type}")
        print(f"Total epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}\n")

        # Select appropriate training/validation functions
        if self.config.model_type == 'stateful':
            train_fn = self.train_epoch_stateful
            val_fn = self.validate_stateful
        else:
            train_fn = self.train_epoch_sequence
            val_fn = self.validate_sequence

        # Training loop
        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()

            print(f"Epoch {epoch+1}/{self.config.num_epochs}")

            # Training
            print(f"  Training...")
            train_loss = train_fn(train_loader)

            # Validation
            print(f"  Validating...")
            val_loss = val_fn(val_loader)

            # Timing
            epoch_time = time.time() - epoch_start_time

            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            self.history['epoch_times'].append(epoch_time)

            # Print epoch summary
            print(f"  Results: Train Loss = {train_loss:.6f}, "
                  f"Val Loss = {val_loss:.6f}, "
                  f"LR = {current_lr:.6f}, "
                  f"Time = {epoch_time:.1f}s")

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                if self.config.save_best_only:
                    self.save_checkpoint(epoch, is_best=True)
                    print(f"  ✓ Best model saved (val_loss: {val_loss:.6f})")

            # Early stopping check
            if self.early_stopping is not None:
                if self.early_stopping(val_loss):
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    print(f"Best epoch: {self.best_epoch+1} "
                          f"(val_loss: {self.best_val_loss:.6f})")
                    break

            print()  # Empty line between epochs

        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.6f} at epoch {self.best_epoch+1}")

        return self.history

    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False
    ) -> None:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        # Create checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Prepare checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config.to_dict()
        }

        # Save checkpoint
        if is_best:
            path = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
        else:
            path = os.path.join(self.config.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')

        torch.save(checkpoint, path)

    def load_checkpoint(
        self,
        checkpoint_path: str
    ) -> None:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']

        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Epoch: {checkpoint['epoch']+1}, Best val loss: {self.best_val_loss:.6f}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig
) -> Tuple[nn.Module, Dict]:
    """
    Convenience function to train a model.

    Args:
        model: LSTM model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration

    Returns:
        trained_model: Trained model
        history: Training history
    """
    trainer = Trainer(model, config)
    history = trainer.train(train_loader, val_loader)

    # Load best model weights
    best_model_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        trainer.load_checkpoint(best_model_path)

    return trainer.model, history


if __name__ == '__main__':
    """Test trainer with dummy data."""

    print("Testing Trainer...")

    # This would require full data pipeline, so we'll just test initialization
    from ..models import StatefulLSTM
    from .config import get_fast_test_config

    print("\n1. Creating model and config...")
    model = StatefulLSTM(input_size=5, hidden_size=32, num_layers=1)
    config = get_fast_test_config()

    print("\n2. Initializing trainer...")
    trainer = Trainer(model, config)
    print(f"   Device: {trainer.device}")
    print(f"   Optimizer: {trainer.optimizer.__class__.__name__}")
    print(f"   Loss function: {trainer.criterion.__class__.__name__}")

    print("\n3. Testing early stopping...")
    early_stop = EarlyStopping(patience=3, min_delta=0.01)

    losses = [1.0, 0.9, 0.85, 0.84, 0.83, 0.83, 0.83, 0.83]
    for i, loss in enumerate(losses):
        print(f"   Epoch {i+1}: loss={loss:.2f}")
        should_stop = early_stop(loss)
        if should_stop:
            print(f"   Stopped at epoch {i+1}")
            break

    print("\n✓ Trainer tests passed!")
