"""
Training Configuration
======================

Configuration dataclass for training hyperparameters and settings.
"""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class TrainingConfig:
    """
    Configuration for LSTM training.

    This dataclass contains all hyperparameters and settings needed
    for training both L=1 stateful and L>1 sequence models.

    Model Settings:
        model_type: 'stateful' for L=1 or 'sequence' for L>1
        hidden_size: LSTM hidden dimension
        num_layers: Number of LSTM layers
        sequence_length: Sequence length (only for L>1 model)
        dropout: Dropout probability (recommended for L>1)

    Training Settings:
        batch_size: Training batch size
        learning_rate: Initial learning rate
        num_epochs: Maximum number of training epochs
        optimizer_type: 'adam' or 'sgd'
        weight_decay: L2 regularization coefficient
        gradient_clip_norm: Maximum gradient norm (for clipping)

    Data Settings:
        val_split: Fraction of training data for validation
        num_workers: DataLoader worker processes

    Early Stopping:
        patience: Number of epochs with no improvement before stopping
        min_delta: Minimum change to qualify as improvement

    Checkpointing:
        save_best_only: Only save model when validation improves
        checkpoint_dir: Directory for saving model checkpoints

    Device and Logging:
        device: 'cuda', 'cpu', or 'mps' (Apple Silicon)
        verbose: Print training progress
        log_interval: Print frequency (batches)
    """

    # Model settings
    model_type: str = 'stateful'  # 'stateful' or 'sequence'
    hidden_size: int = 64
    num_layers: int = 1
    sequence_length: int = 1  # Only for L>1
    dropout: float = 0.0

    # Training settings
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 50
    optimizer_type: str = 'adam'  # 'adam' or 'sgd'
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0

    # Data settings
    val_split: float = 0.1
    num_workers: int = 0

    # Early stopping
    use_early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4

    # Checkpointing
    save_best_only: bool = True
    checkpoint_dir: str = 'outputs/models'

    # Device and logging
    device: str = field(default_factory=lambda: (
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    ))
    verbose: bool = True
    log_interval: int = 100  # Print every N batches

    # Random seed
    seed: int = 42

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate model_type
        assert self.model_type in ['stateful', 'sequence'], \
            f"model_type must be 'stateful' or 'sequence', got '{self.model_type}'"

        # Validate optimizer
        assert self.optimizer_type in ['adam', 'sgd'], \
            f"optimizer_type must be 'adam' or 'sgd', got '{self.optimizer_type}'"

        # Validate device
        valid_devices = ['cuda', 'cpu', 'mps']
        assert any(self.device.startswith(d) for d in valid_devices), \
            f"device must be one of {valid_devices}, got '{self.device}'"

        # Validate ranges
        assert 0 < self.val_split < 1, "val_split must be between 0 and 1"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert self.hidden_size > 0, "hidden_size must be positive"
        assert self.num_layers > 0, "num_layers must be positive"

        # Warnings
        if self.model_type == 'stateful' and self.dropout > 0:
            print(f"WARNING: dropout={self.dropout} for stateful model. "
                  "Usually not needed for L=1.")

        if self.model_type == 'sequence' and self.sequence_length == 1:
            print("WARNING: sequence_length=1 for sequence model. "
                  "Consider using L>1 (e.g., 10 or 50).")

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'model_type': self.model_type,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'sequence_length': self.sequence_length,
            'dropout': self.dropout,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'optimizer_type': self.optimizer_type,
            'weight_decay': self.weight_decay,
            'gradient_clip_norm': self.gradient_clip_norm,
            'val_split': self.val_split,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'seed': self.seed
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TrainingConfig(\n"
            f"  Model: {self.model_type}, hidden={self.hidden_size}, "
            f"layers={self.num_layers}, L={self.sequence_length}\n"
            f"  Training: lr={self.learning_rate}, batch={self.batch_size}, "
            f"epochs={self.num_epochs}\n"
            f"  Device: {self.device}\n"
            f")"
        )


# Predefined configurations
def get_default_stateful_config() -> TrainingConfig:
    """Get default configuration for L=1 stateful model."""
    return TrainingConfig(
        model_type='stateful',
        hidden_size=64,
        num_layers=1,
        sequence_length=1,
        dropout=0.0,
        batch_size=32,
        learning_rate=0.001,
        num_epochs=50
    )


def get_default_sequence_config(sequence_length: int = 10) -> TrainingConfig:
    """Get default configuration for L>1 sequence model."""
    return TrainingConfig(
        model_type='sequence',
        hidden_size=64,
        num_layers=2,
        sequence_length=sequence_length,
        dropout=0.2,
        batch_size=32,
        learning_rate=0.001,
        num_epochs=50
    )


def get_fast_test_config() -> TrainingConfig:
    """Get fast configuration for testing (few epochs, small model)."""
    return TrainingConfig(
        model_type='stateful',
        hidden_size=32,
        num_layers=1,
        batch_size=64,
        learning_rate=0.01,
        num_epochs=5,
        patience=3,
        verbose=True
    )


if __name__ == '__main__':
    """Test training configuration."""

    print("Testing TrainingConfig...")

    # Test default configs
    print("\n1. Default stateful config:")
    config_stateful = get_default_stateful_config()
    print(config_stateful)

    print("\n2. Default sequence config:")
    config_sequence = get_default_sequence_config(sequence_length=10)
    print(config_sequence)

    print("\n3. Fast test config:")
    config_test = get_fast_test_config()
    print(config_test)

    # Test to_dict
    print("\n4. Config as dictionary:")
    config_dict = config_stateful.to_dict()
    for key, value in config_dict.items():
        print(f"   {key}: {value}")

    print("\n✓ All configuration tests passed!")
