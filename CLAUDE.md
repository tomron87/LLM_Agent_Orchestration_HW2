# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an M.Sc. Data Science homework assignment implementing LSTM models for frequency extraction from noisy mixed signals. The assignment explores how LSTM memory window length (L) affects the ability to separate frequency components from a signal containing 4 sinusoids (1, 3, 5, 7 Hz) with per-sample random amplitude and phase noise.

**Critical Context**: The assignment requirements (L2-homework.pdf) specify per-sample randomization of amplitude A_i(t) and phase φ_i(t), which makes the task theoretically challenging. The correlation between input signal S(t) and target frequencies is near zero, resulting in MSE convergence around 0.5 (the variance of sinusoidal targets).

## Common Commands

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run tests
pytest tests/                           # All tests
pytest tests/test_models.py            # Model tests only
pytest tests/test_data.py -v           # Data generation tests with verbose output

# Train models
python src/training/train.py           # Default training (L=1 stateful)
python examples/train_sequence.py      # Sequence model (L>1)

# Module testing (each module is self-contained)
python -m src.data.signal_generator    # Test signal generation
python -m src.models.lstm_stateful     # Test stateful LSTM
python -m src.training.trainer         # Test trainer initialization

# Jupyter demo
jupyter notebook Demonstration.ipynb   # Interactive demonstration
```

## Architecture Highlights

### Two Model Paradigms

The codebase implements two distinct approaches to LSTM-based frequency extraction:

1. **L=1 Stateful LSTM** (`src/models/lstm_stateful.py`)
   - Processes ONE sample at a time (input: 5 features → output: 1 value)
   - Explicitly manages hidden/cell states between forward passes
   - **Critical**: State MUST be reset when switching frequencies (tracked via `freq_idx` and `sample_idx` in dataset)
   - State detachment after backward pass prevents "backward through graph twice" errors (trainer.py:223-225)
   - Used when you want to emphasize recurrent memory across many timesteps

2. **L>1 Sequence LSTM** (`src/models/lstm_sequence.py`)
   - Processes sequences of L samples at once (input: [batch, L, 5] → output: [batch, L, 1])
   - Standard PyTorch LSTM usage with no explicit state management
   - Assignment recommends L=50 or L=100 (Section 4.3 of PDF)
   - Simpler to train but different computational characteristics

### State Management Pattern (Critical for L=1)

The stateful model requires careful coordination between dataset, model, and trainer:

- **Dataset** (`src/data/dataset.py`): Returns `freq_idx` and `sample_idx` with each batch
- **DataLoader**: Uses custom collate function to preserve batch structure
- **Trainer** (`src/training/trainer.py:199-208`): Resets state when:
  1. Frequency changes (new `freq_idx`)
  2. Sample index is 0 (start of frequency sequence)

```python
# Pattern used throughout trainer.py
reset_state = False
if current_freq_idx is None or freq_idx[0].item() != current_freq_idx:
    reset_state = True
    current_freq_idx = freq_idx[0].item()
elif sample_idx[0].item() == 0:
    reset_state = True

outputs = self.model(inputs, reset_state=reset_state)
```

### Data Generation Pipeline

Signal generation follows assignment specifications (Section 2.2 of PDF):

1. **Raw signal** (`signal_generator.py`): Creates S(t) = (1/4) * Σ Sinus_i^noisy(t)
   - Each frequency i has: A_i(t) ~ U(0.8, 1.2) and φ_i(t) ~ U(0, 2π) **per sample**
   - Ground truth: Target_i(t) = sin(2π * f_i * t) (pure sinusoid)

2. **Feature extraction** (`feature_extractor.py`): Converts S(t) → 5 features per sample
   - Temporal: [S(t-2), S(t-1), S(t), S(t+1), S(t+2)]
   - Provides local context window around each sample

3. **Dataset classes** (`dataset.py`):
   - `StatefulDataset`: For L=1, yields (5,) → (1,) with metadata
   - `SequenceDataset`: For L>1, yields (L, 5) → (L, 1) sequences

4. **Splitter** (`splitter.py`): Separates by frequency to prevent data leakage
   - Training/validation split respects frequency boundaries
   - Different frequencies can have different noise patterns

## Configuration System

All training controlled via `TrainingConfig` dataclass (`src/training/config.py`):

```python
config = TrainingConfig(
    model_type='stateful',      # or 'sequence'
    hidden_size=64,
    sequence_length=1,          # L value (1 for stateful, 50-100 for sequence)
    learning_rate=0.001,
    batch_size=32,
    num_epochs=50,
    device='cuda'               # or 'cpu', 'mps' for Apple Silicon
)
```

Predefined configs available:
- `get_default_stateful_config()`: L=1, no dropout
- `get_default_sequence_config(L)`: L>1, dropout=0.2, 2 layers
- `get_fast_test_config()`: Quick testing, 5 epochs

## Known Issues and Constraints

1. **Expected MSE ≈ 0.5**: Due to per-sample randomization, correlation between S(t) and targets is near zero. The theoretical minimum MSE equals the variance of sinusoidal targets (~0.5). This is not a bug.

2. **Apple Silicon (MPS)**: Device detection currently defaults to 'cuda' → 'cpu'. Manual override needed for MPS:
   ```python
   config.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
   ```

3. **State detachment required** (stateful only): After `loss.backward()`, states must be detached to prevent computational graph errors on next forward pass (trainer.py:223-225).

4. **Batch size = 1 recommended for L=1**: Stateful model works best with batch_size=1 to ensure proper sequential processing of frequency data.

## Testing Philosophy

The test suite (1,198 lines across 5 files) is comprehensive:

- `test_signal_generator.py`: Validates signal statistics, frequency content, reproducibility
- `test_dataset.py`: Checks shapes, data flow, train/val splitting
- `test_models.py`: Forward pass, state management, gradient flow, output ranges
- `test_training.py`: End-to-end training, loss computation, checkpointing
- `test_integration.py`: Full pipeline from data generation → training → evaluation

Each module also has `if __name__ == '__main__':` blocks for standalone testing.

## File Organization

```
src/
├── data/
│   ├── signal_generator.py    # Core: creates S(t) and targets
│   ├── feature_extractor.py   # 5-feature temporal window
│   ├── dataset.py              # StatefulDataset, SequenceDataset
│   └── splitter.py             # Train/val split by frequency
├── models/
│   ├── lstm_stateful.py        # L=1 with explicit state management
│   ├── lstm_sequence.py        # L>1 standard LSTM
│   └── lstm_film.py            # Experimental: FiLM conditioning
├── training/
│   ├── config.py               # TrainingConfig dataclass
│   ├── trainer.py              # Main training loop (handles both L=1 and L>1)
│   ├── evaluator.py            # MSE, correlation metrics
│   └── train.py                # Entry point script
└── utils/
    ├── helpers.py              # Seed setting, device helpers
    └── visualization.py        # Plotting functions

tests/                          # Mirror structure of src/
examples/
├── train_stateful.py           # L=1 training example
├── train_sequence.py           # L>1 training example
└── train_film.py               # FiLM-conditioned training

Demonstration.ipynb             # Full walkthrough with visualizations
```

## Important Notes

- The `seed` parameter is critical for reproducibility. Training uses seed=1, test uses seed=2.
- Early stopping is enabled by default (patience=10, min_delta=1e-4).
- Best model is saved automatically when validation loss improves.
- Gradient clipping (max_norm=1.0) is always applied to prevent exploding gradients.
- The FiLM-conditioned model (`lstm_film.py`) is experimental and allows conditioning predictions on which frequency to extract.
