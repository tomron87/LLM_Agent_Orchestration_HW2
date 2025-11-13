# Extensibility & Extension Guide

**Version:** 1.0
**Last Updated:** November 13, 2025

This document provides comprehensive guidance on extending the LSTM Frequency Extraction System with new functionality. The system is designed with modularity and extensibility in mind.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Extension Points](#extension-points)
3. [Adding New Models](#adding-new-models)
4. [Adding New Data Sources](#adding-new-data-sources)
5. [Custom Training Strategies](#custom-training-strategies)
6. [Custom Evaluation Metrics](#custom-evaluation-metrics)
7. [Custom Visualizations](#custom-visualizations)
8. [Plugin System Guidelines](#plugin-system-guidelines)
9. [Testing Extensions](#testing-extensions)
10. [Best Practices](#best-practices)

---

## Architecture Overview

The system follows a modular architecture with clear separation of concerns:

```
src/
├── data/           # Data generation and loading
├── models/         # Neural network architectures
├── training/       # Training logic and configuration
├── evaluation/     # Metrics and visualization
└── utils/          # Shared utilities (logger, config)
```

**Design Principles:**
- **Separation of Concerns**: Each module has a single responsibility
- **Interface-Based**: Base classes define clear contracts
- **Configuration-Driven**: Behavior controlled via YAML files
- **Dependency Injection**: Components receive dependencies explicitly

---

## Extension Points

### 1. Models (`src/models/`)

**Base Interface:** All models should inherit from `nn.Module` and follow the pattern:

```python
import torch.nn as nn
from typing import Optional, Tuple

class CustomModel(nn.Module):
    """Your custom model implementation."""

    def __init__(self, input_size: int, hidden_size: int, **kwargs):
        super().__init__()
        # Initialize layers
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor
            **kwargs: Additional arguments (e.g., reset_state for stateful models)

        Returns:
            Output tensor
        """
        # Implement forward logic
        pass
```

**Integration:**
- Add your model file to `src/models/`
- Import in `src/models/__init__.py`
- Update `main.py` to recognize new model type

### 2. Data Sources (`src/data/`)

**Base Interface:** Extend `torch.utils.data.Dataset`:

```python
from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):
    """Your custom dataset implementation."""

    def __init__(self, data_source, **kwargs):
        """Initialize with your data source."""
        self.data = self._load_data(data_source)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item at index.

        Returns:
            Tuple of (input, target)
        """
        return self.data[idx]

    def _load_data(self, source):
        """Load data from source."""
        pass
```

### 3. Training Strategies (`src/training/`)

**Extension Pattern:**

```python
from src.training.trainer import Trainer
from typing import Dict

class CustomTrainer(Trainer):
    """Custom training strategy."""

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Override to implement custom training logic.

        Args:
            epoch: Current epoch number

        Returns:
            Dict of metrics
        """
        # Your custom training loop
        pass
```

### 4. Evaluation Metrics (`src/evaluation/`)

**Adding New Metrics:**

```python
import torch

def custom_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute your custom metric.

    Args:
        predictions: Model predictions
        targets: Ground truth values

    Returns:
        Metric value (float)
    """
    # Implement your metric
    pass
```

Add to `src/evaluation/metrics.py` and use in evaluation loop.

---

## Adding New Models

### Step-by-Step Guide

#### Step 1: Create Model File

Create `src/models/your_model.py`:

```python
"""
Your Model
==========

Description of your model architecture.
"""

import torch
import torch.nn as nn
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class YourModel(nn.Module):
    """
    Your custom LSTM variant or alternative architecture.

    Args:
        input_size: Input feature dimension
        hidden_size: Hidden layer size
        **kwargs: Additional model-specific parameters

    Example:
        >>> model = YourModel(input_size=5, hidden_size=128)
        >>> output = model(input_tensor)
    """

    def __init__(self, input_size: int = 5, hidden_size: int = 64, **kwargs):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Define your layers
        self.encoder = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.decoder = nn.Linear(hidden_size, 1)

        logger.info(f"YourModel initialized: input={input_size}, hidden={hidden_size}")

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass."""
        x = self.encoder(x)
        x, _ = self.lstm(x)
        output = self.decoder(x)
        return output

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test function
if __name__ == '__main__':
    model = YourModel(input_size=5, hidden_size=128)
    test_input = torch.randn(16, 10, 5)  # batch=16, seq=10, features=5
    output = model(test_input)
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {model.count_parameters():,}")
```

#### Step 2: Register Model

Update `src/models/__init__.py`:

```python
from .lstm_stateful import StatefulLSTM
from .lstm_sequence import SequenceLSTM
from .your_model import YourModel  # Add this line

__all__ = ['StatefulLSTM', 'SequenceLSTM', 'YourModel']
```

#### Step 3: Update Configuration

Add to `config.yaml`:

```yaml
model:
  type: "your_model"  # New model type
  hidden_size: 128
  # Add your model-specific parameters here
```

#### Step 4: Update Main Script

Modify `main.py` to instantiate your model:

```python
from src.models import YourModel

# In create_model() function:
elif args.model == 'your_model':
    model = YourModel(
        input_size=5,
        hidden_size=args.hidden_size,
        # Add your parameters
    )
```

#### Step 5: Add Tests

Create `tests/test_your_model.py`:

```python
import pytest
import torch
from src.models.your_model import YourModel


class TestYourModel:
    """Test suite for YourModel."""

    @pytest.fixture
    def model(self):
        """Create model instance."""
        return YourModel(input_size=5, hidden_size=32)

    def test_initialization(self, model):
        """Test model initializes correctly."""
        assert model.input_size == 5
        assert model.hidden_size == 32

    def test_forward_pass(self, model):
        """Test forward pass produces correct output shape."""
        batch_size, seq_len = 4, 10
        x = torch.randn(batch_size, seq_len, 5)
        output = model(x)
        assert output.shape == (batch_size, seq_len, 1)

    def test_parameter_count(self, model):
        """Test parameter counting."""
        params = model.count_parameters()
        assert params > 0
```

---

## Adding New Data Sources

### Example: Real Audio Data

```python
"""
Custom Audio Dataset
====================

Load real audio signals instead of synthetic data.
"""

from torch.utils.data import Dataset
import torch
import librosa  # Audio processing library
from pathlib import Path
from typing import Tuple


class AudioDataset(Dataset):
    """
    Dataset for real audio signals.

    Args:
        audio_dir: Directory containing audio files
        sample_rate: Target sample rate
        duration: Duration of each clip in seconds
    """

    def __init__(self, audio_dir: str, sample_rate: int = 1000, duration: float = 10.0):
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.duration = duration
        self.audio_files = list(self.audio_dir.glob("*.wav"))

        if not self.audio_files:
            raise ValueError(f"No audio files found in {audio_dir}")

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and process audio file."""
        audio_path = self.audio_files[idx]

        # Load audio
        signal, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)

        # Extract features (e.g., MFCC, spectrogram)
        features = self._extract_features(signal)

        # Load corresponding target (if available)
        target = self._load_target(audio_path)

        return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

    def _extract_features(self, signal):
        """Extract features from audio signal."""
        # Implement your feature extraction
        pass

    def _load_target(self, audio_path):
        """Load target for audio file."""
        # Implement target loading logic
        pass
```

---

## Custom Training Strategies

### Example: Curriculum Learning

```python
"""
Curriculum Learning Trainer
===========================

Gradually increase training difficulty.
"""

from src.training.trainer import Trainer
import torch


class CurriculumTrainer(Trainer):
    """
    Trainer with curriculum learning strategy.

    Starts with easy examples and gradually increases difficulty.
    """

    def __init__(self, model, config, difficulty_schedule):
        super().__init__(model, config)
        self.difficulty_schedule = difficulty_schedule
        self.current_difficulty = 0

    def train_epoch(self, epoch: int):
        """
        Train one epoch with curriculum learning.

        Difficulty increases based on schedule.
        """
        # Update difficulty
        self.current_difficulty = self.difficulty_schedule(epoch)

        # Filter dataset by difficulty
        filtered_data = self._filter_by_difficulty(self.train_loader, self.current_difficulty)

        # Train on filtered data
        return super().train_epoch_on_data(filtered_data)

    def _filter_by_difficulty(self, dataloader, difficulty):
        """Select examples based on current difficulty level."""
        # Implement filtering logic
        pass
```

---

## Custom Evaluation Metrics

### Example: Frequency-Specific Accuracy

```python
"""
Custom Metrics
==============

Additional evaluation metrics beyond MSE.
"""

import torch
import numpy as np
from scipy import signal as sp_signal


def frequency_domain_similarity(predictions: torch.Tensor, targets: torch.Tensor, fs: int = 1000) -> float:
    """
    Compute similarity in frequency domain using FFT.

    Args:
        predictions: Predicted signals
        targets: Target signals
        fs: Sampling frequency

    Returns:
        Cosine similarity in frequency domain
    """
    # Convert to numpy
    pred_np = predictions.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()

    # Compute FFT
    pred_fft = np.fft.fft(pred_np)
    target_fft = np.fft.fft(target_np)

    # Compute cosine similarity
    similarity = np.dot(pred_fft.flatten(), target_fft.flatten().conj())
    similarity /= (np.linalg.norm(pred_fft) * np.linalg.norm(target_fft))

    return float(np.abs(similarity))


def phase_coherence(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Measure phase alignment between predictions and targets.

    Args:
        predictions: Predicted signals
        targets: Target signals

    Returns:
        Phase coherence value [0, 1]
    """
    # Extract instantaneous phase
    pred_analytic = sp_signal.hilbert(predictions.detach().cpu().numpy().flatten())
    target_analytic = sp_signal.hilbert(targets.detach().cpu().numpy().flatten())

    pred_phase = np.angle(pred_analytic)
    target_phase = np.angle(target_analytic)

    # Compute phase difference
    phase_diff = np.abs(pred_phase - target_phase)
    coherence = 1 - np.mean(phase_diff) / np.pi

    return float(coherence)
```

---

## Custom Visualizations

### Example: Interactive Plots

```python
"""
Interactive Visualization
=========================

Create interactive plots using Plotly.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


class InteractiveVisualizer:
    """Create interactive visualizations with Plotly."""

    def plot_training_curves_interactive(self, history: dict, output_path: str):
        """
        Create interactive training curve plot.

        Args:
            history: Training history dict
            output_path: Path to save HTML file
        """
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Loss", "Learning Rate"))

        # Loss curves
        fig.add_trace(
            go.Scatter(y=history['train_loss'], name='Train Loss', mode='lines'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(y=history['val_loss'], name='Val Loss', mode='lines'),
            row=1, col=1
        )

        # Learning rate
        if 'learning_rate' in history:
            fig.add_trace(
                go.Scatter(y=history['learning_rate'], name='LR', mode='lines'),
                row=2, col=1
            )

        fig.update_layout(height=800, showlegend=True, hovermode='x unified')
        fig.write_html(output_path)
```

---

## Plugin System Guidelines

### Creating a Plugin

Plugins should follow this structure:

```
plugins/
└── my_plugin/
    ├── __init__.py
    ├── plugin.py      # Main plugin logic
    ├── config.yaml     # Plugin configuration
    └── README.md       # Plugin documentation
```

**Plugin Template:**

```python
"""
Plugin Template
===============

Template for creating system plugins.
"""

from typing import Any, Dict


class Plugin:
    """
    Base plugin class.

    All plugins should inherit from this class and implement
    the required methods.
    """

    name = "base_plugin"
    version = "1.0.0"
    description = "Base plugin template"

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize plugin with configuration.

        Args:
            config: Plugin configuration dictionary
        """
        self.config = config
        self.enabled = config.get('enabled', True)

    def initialize(self):
        """Called when plugin is loaded."""
        pass

    def on_train_start(self, trainer):
        """Called at start of training."""
        pass

    def on_epoch_end(self, epoch, metrics):
        """Called at end of each epoch."""
        pass

    def on_train_end(self, final_metrics):
        """Called at end of training."""
        pass

    def cleanup(self):
        """Called when plugin is unloaded."""
        pass
```

---

## Testing Extensions

### Testing Checklist

- [ ] Unit tests for all new functions
- [ ] Integration tests with existing system
- [ ] Test with different configurations
- [ ] Test error handling and edge cases
- [ ] Document test coverage
- [ ] Add to CI/CD pipeline

### Example Test Structure

```python
import pytest
from your_extension import YourExtension


class TestYourExtension:
    """Comprehensive tests for your extension."""

    @pytest.fixture
    def extension(self):
        """Create extension instance."""
        return YourExtension(config={})

    def test_initialization(self, extension):
        """Test proper initialization."""
        assert extension is not None

    def test_integration(self, extension):
        """Test integration with main system."""
        # Test that extension works with existing components
        pass

    def test_error_handling(self, extension):
        """Test error cases."""
        with pytest.raises(ValueError):
            extension.invalid_operation()
```

---

## Best Practices

### Code Quality

1. **Type Hints**: Use type annotations for all function signatures
2. **Docstrings**: Document all public APIs with examples
3. **Error Handling**: Provide clear error messages
4. **Logging**: Add logging statements at key points
5. **Testing**: Achieve ≥80% code coverage

### Integration

1. **Configuration**: Make extensions configurable via YAML
2. **Backward Compatibility**: Don't break existing functionality
3. **Documentation**: Update relevant documentation
4. **Examples**: Provide usage examples
5. **Version Control**: Use semantic versioning

### Performance

1. **Profiling**: Profile your extension for bottlenecks
2. **Memory**: Monitor memory usage for large datasets
3. **GPU Utilization**: Ensure efficient GPU usage if applicable
4. **Benchmarking**: Compare performance against baseline

---

## Getting Help

- **Documentation**: Check existing documentation in `documentation/`
- **Code Examples**: Review existing implementations in `src/models/`, `src/data/`
- **Tests**: Look at `tests/` for usage patterns
- **Issues**: Report issues on GitHub
- **Community**: Join project discussions

---

## Contribution Guidelines

When contributing extensions:

1. **Follow Style Guide**: Match existing code style (PEP 8)
2. **Add Tests**: Include comprehensive tests
3. **Update Docs**: Document your changes
4. **Provide Examples**: Include usage examples
5. **Review**: Submit for code review before merging

---

**Happy Extending! 🚀**
