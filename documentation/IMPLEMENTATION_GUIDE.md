# Implementation Guide
## LSTM Frequency Extraction System

**Version:** 1.0
**Date:** November 2025
**Audience:** Developers, Students, Researchers

---

## Table of Contents

1. [Introduction](#introduction)
2. [System Overview](#system-overview)
3. [Step-by-Step Implementation](#step-by-step-implementation)
4. [Code Walkthrough](#code-walkthrough)
5. [Training Guide](#training-guide)
6. [Evaluation Guide](#evaluation-guide)
7. [Troubleshooting](#troubleshooting)
8. [Extension Ideas](#extension-ideas)

---

## 1. Introduction

This guide walks you through the complete implementation of the LSTM frequency extraction system, explaining design decisions, code structure, and how to run experiments.

### Learning Objectives

After following this guide, you will understand:
- How to generate synthetic signal data for deep learning
- LSTM state management (L=1 vs L>1 approaches)
- PyTorch dataset and dataloader creation
- Training pipeline implementation
- Model evaluation and visualization

---

## 2. System Overview

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Main Pipeline                         │
└─────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Data         │   │ Model        │   │ Evaluation   │
│ Generation   │──▶│ Training     │──▶│ & Viz        │
└──────────────┘   └──────────────┘   └──────────────┘
        │                  │                  │
        ▼                  ▼                  ▼
  Dataset Files      Model Weights      Figures & Metrics
```

### Component Breakdown

| Component | Purpose | Key Files |
|-----------|---------|-----------|
| Data Generation | Create synthetic signals | `src/data/signal_generator.py` |
| Dataset | PyTorch data handling | `src/data/dataset.py` |
| Models | LSTM architectures | `src/models/lstm_stateful.py`, `lstm_sequence.py` |
| Training | Training loops | `src/training/trainer.py` |
| Evaluation | Metrics & plots | `src/evaluation/metrics.py`, `visualization.py` |

---

## 3. Step-by-Step Implementation

### Step 1: Environment Setup

**Create virtual environment:**
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Verify installation:**
```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### Step 2: Generate Datasets

**Understanding the data:**
- **Mixed Signal S(t)**: Average of 4 noisy sinusoids
- **Noise**: Random amplitude and phase per sample
- **Targets**: Pure sinusoids (ground truth)

**Generate train and test sets:**
```bash
python3 -c "
from src.data import generate_dataset

# Training set (seed=1)
train_data = generate_dataset(seed=1, save_path='outputs/train_data.pkl')
print(f'Train samples: {len(train_data[\"S\"])}')

# Test set (seed=2)
test_data = generate_dataset(seed=2, save_path='outputs/test_data.pkl')
print(f'Test samples: {len(test_data[\"S\"])}')
"
```

**Inspect the data:**
```python
import pickle
import numpy as np

# Load data
with open('outputs/train_data.pkl', 'rb') as f:
    data = pickle.load(f)

print("Data keys:", data.keys())
print("Mixed signal S shape:", data['S'].shape)
print("Targets shape:", data['targets'].shape)
print("Frequencies:", data['frequencies'])
```

### Step 3: Choose Model Type

**Decision matrix:**

| Criterion | L=1 Stateful | L>1 Sequence |
|-----------|--------------|--------------|
| Learning Goal | Understand LSTM internals | Standard PyTorch workflow |
| Performance | Good | Slightly better |
| Training Speed | Slower | Faster |
| Implementation | More complex | Simpler |

### Step 4: Run Training

**Option A: L=1 Stateful Model**
```bash
python3 main.py \
    --model stateful \
    --hidden-size 64 \
    --num-layers 1 \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001
```

**Option B: L>1 Sequence Model**
```bash
python3 main.py \
    --model sequence \
    --sequence-length 10 \
    --hidden-size 64 \
    --num-layers 2 \
    --dropout 0.2 \
    --epochs 50
```

### Step 5: Monitor Training

**Training output shows:**
```
Epoch 1/50
  Training...
      Batch 100/1125, Loss: 0.245631
  Validating...
  Results: Train Loss = 0.234567, Val Loss = 0.245123, LR = 0.001000
  ✓ Best model saved (val_loss: 0.245123)

Epoch 2/50
  ...
```

**Key metrics to watch:**
- Train loss should decrease steadily
- Val loss should track train loss (no large gap = good generalization)
- Early stopping will trigger if val loss plateaus

### Step 6: Evaluate Results

**Automatic evaluation:**
The system automatically evaluates on both train and test sets, printing:
```
==============================================================
                    EVALUATION RESULTS
==============================================================

Overall Performance:
  Training MSE:   0.048234
  Test MSE:       0.056789

Generalization Check:
  Test/Train Ratio: 1.177
  Generalizes Well: ✓ YES

Per-Frequency Performance:
Freq     Hz       Train MSE    Test MSE     Correlation
--------------------------------------------------------------
1        1.0      0.052341     0.061234     0.9876
2        3.0      0.048123     0.056789     0.9892
3        5.0      0.046789     0.054321     0.9901
4        7.0      0.045678     0.052109     0.9915
==============================================================
```

### Step 7: Examine Visualizations

**Generated figures in `outputs/figures/`:**

1. **graph1_single_frequency.png**
   - Target (green line)
   - LSTM output (blue dots/line)
   - Mixed signal (gray background)

2. **graph2_all_frequencies.png**
   - 2×2 grid showing all 4 frequencies

3. **training_curves.png**
   - Loss over epochs
   - Learning rate schedule

4. **fft_analysis.png**
   - Frequency domain analysis
   - Spectral content comparison

5. **error_distribution.png**
   - Error histogram
   - Predictions vs targets scatter
   - Error over time

---

## 4. Code Walkthrough

### 4.1 Signal Generation (`src/data/signal_generator.py`)

**Key function: `generate_noisy_component()`**

```python
def generate_noisy_component(self, freq_idx: int, t: np.ndarray) -> np.ndarray:
    """
    Generate one noisy sinusoidal component.

    For each sample, amplitude and phase are randomly generated:
    - A_i(t) ~ Uniform(0.8, 1.2)
    - φ_i(t) ~ Uniform(0, 2π)
    """
    freq = self.frequencies[freq_idx]

    # CRITICAL: Random values for EACH sample
    A = self.rng.uniform(0.8, 1.2, size=len(t))
    phi = self.rng.uniform(0, 2 * np.pi, size=len(t))

    # Noisy sinusoid
    noisy_signal = A * np.sin(2 * np.pi * freq * t + phi)

    return noisy_signal
```

**Why this matters:**
- Each sample has different noise → challenging for the model
- Forces LSTM to learn underlying frequency pattern, not memorize noise
- Tests generalization capability

### 4.2 PyTorch Dataset (`src/data/dataset.py`)

**Key class: `FrequencyExtractionDataset`**

```python
def __getitem__(self, idx: int) -> Dict:
    """
    Map global index to (freq_idx, sample_idx):
    - freq_idx = idx // num_samples  (which frequency: 0-3)
    - sample_idx = idx % num_samples (which time point: 0-9999)
    """
    freq_idx = idx // self.num_samples
    sample_idx = idx % self.num_samples

    # Create one-hot encoding
    C = np.zeros(self.num_frequencies)
    C[freq_idx] = 1.0

    # Input: [S[t], C1, C2, C3, C4]
    input_vec = np.concatenate([[self.S[sample_idx]], C])

    # Target
    target_val = self.targets[freq_idx, sample_idx]

    return {
        'input': torch.FloatTensor(input_vec),
        'target': torch.FloatTensor([target_val]),
        'freq_idx': freq_idx,
        'sample_idx': sample_idx,
        't': self.t[sample_idx]
    }
```

**Dataset organization:**
```
Rows 0-9999:      Frequency 1 (1 Hz), all time points
Rows 10000-19999: Frequency 2 (3 Hz), all time points
Rows 20000-29999: Frequency 3 (5 Hz), all time points
Rows 30000-39999: Frequency 4 (7 Hz), all time points
```

**Why sequential order matters:**
- For L=1 stateful model, consecutive samples must maintain temporal continuity
- Shuffling would break the temporal dependency
- State must reset only when switching frequencies

### 4.3 Stateful LSTM Model (`src/models/lstm_stateful.py`)

**Critical: State Management**

```python
def forward(self, x: torch.Tensor, reset_state: bool = False) -> torch.Tensor:
    """
    Forward pass with state preservation.

    Args:
        x: Input (batch, 5) or (batch, 1, 5)
        reset_state: If True, reset internal state to zeros
    """
    batch_size = x.size(0)

    # Add sequence dimension if needed
    if x.dim() == 2:
        x = x.unsqueeze(1)  # (batch, 1, 5)

    # Initialize or reset state if needed
    if reset_state or self.hidden_state is None:
        self.init_hidden(batch_size, x.device)

    # LSTM forward pass
    lstm_out, (h_new, c_new) = self.lstm(x, (self.hidden_state, self.cell_state))

    # CRITICAL: Update and detach state for next forward pass
    self.hidden_state = h_new.detach()
    self.cell_state = c_new.detach()

    # Output layer
    output = self.fc(lstm_out[:, -1, :])

    return output
```

**State management rules:**
1. **Reset state** when:
   - Starting a new frequency sequence
   - First sample (sample_idx == 0)
   - Frequency changes

2. **Preserve state** when:
   - Processing consecutive samples within same frequency

### 4.4 Training Loop (`src/training/trainer.py`)

**L=1 Stateful Training:**

```python
def train_epoch_stateful(self, train_loader: DataLoader) -> float:
    """Train one epoch with proper state management."""
    self.model.train()
    total_loss = 0.0
    current_freq_idx = None

    for batch in train_loader:
        inputs = batch['input'].to(self.device)
        targets = batch['target'].to(self.device)
        freq_idx = batch['freq_idx']
        sample_idx = batch['sample_idx']

        # Determine if we need to reset state
        reset_state = False
        if current_freq_idx is None or freq_idx[0].item() != current_freq_idx:
            reset_state = True
            current_freq_idx = freq_idx[0].item()
        elif sample_idx[0].item() == 0:
            reset_state = True

        # Forward pass
        outputs = self.model(inputs, reset_state=reset_state)

        # Backward pass
        loss = self.criterion(outputs, targets)
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.gradient_clip_norm
        )

        self.optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)
```

**Why gradient clipping:**
- LSTMs can suffer from exploding gradients
- Clipping stabilizes training
- Max norm of 1.0 is standard practice

---

## 5. Training Guide

### 5.1 Hyperparameter Tuning

**Recommended starting points:**

| Hyperparameter | L=1 Model | L>1 Model | Notes |
|----------------|-----------|-----------|-------|
| hidden_size | 64 | 64-128 | Larger for L>1 |
| num_layers | 1 | 2-3 | Depth helps with sequences |
| dropout | 0.0 | 0.2-0.3 | Regularization for L>1 |
| batch_size | 32 | 32-64 | Larger possible with sequences |
| learning_rate | 0.001 | 0.001 | Adam default works well |
| sequence_length | 1 | 10-50 | Experiment with L |

**Tuning strategy:**

1. **Start with defaults**
2. **Increase hidden_size** if underfitting (high train loss)
3. **Add dropout** if overfitting (train << val loss)
4. **Adjust learning rate** if not converging
5. **Try different L values** for sequence model

### 5.2 Common Training Issues

**Issue: Loss not decreasing**
```
Solution:
1. Check learning rate (try 0.01 or 0.0001)
2. Verify data is normalized properly
3. Check for bugs in state management (L=1)
4. Try simpler model (hidden_size=32)
```

**Issue: Overfitting (val_loss >> train_loss)**
```
Solution:
1. Add dropout (0.2-0.3)
2. Reduce model size
3. Add more regularization (weight_decay)
4. Check if you're shuffling data (don't shuffle for L=1!)
```

**Issue: Training very slow**
```
Solution:
1. Increase batch_size
2. Use GPU if available
3. Reduce num_epochs for testing
4. Use L>1 model (faster than L=1)
```

### 5.3 Monitoring Training

**What to log:**
```python
# Every epoch
- train_loss: Should decrease
- val_loss: Should track train_loss
- learning_rate: May decrease with scheduler
- epoch_time: Monitor performance

# End of training
- best_epoch: When best val_loss occurred
- best_val_loss: Best validation performance
```

**Good training curve:**
```
Epoch  Train Loss  Val Loss   LR
1      0.450       0.465      0.001
5      0.125       0.138      0.001
10     0.075       0.082      0.001
15     0.055       0.061      0.001
20     0.048       0.056      0.001  ← Converged
```

---

## 6. Evaluation Guide

### 6.1 Success Criteria

**From assignment specifications:**

✅ **MSE < 0.1** on both train and test sets
✅ **Generalization**: test_mse / train_mse ≈ 1.0-1.3
✅ **Visual Quality**: Extracted signals match targets
✅ **Consistency**: All 4 frequencies extracted well

### 6.2 Interpreting Results

**MSE Values:**
```
< 0.05: Excellent
0.05-0.1: Good
0.1-0.2: Acceptable
> 0.2: Needs improvement
```

**Generalization Ratio:**
```
1.0-1.2: Excellent generalization
1.2-1.5: Good generalization
1.5-2.0: Some overfitting
> 2.0: Significant overfitting
```

### 6.3 Debugging Poor Performance

**If MSE is high:**

1. **Check data generation:**
   ```python
   # Verify signal properties
   print(f"S min/max: {data['S'].min():.2f}, {data['S'].max():.2f}")
   print(f"Target min/max: {data['targets'].min():.2f}, {data['targets'].max():.2f}")

   # Plot a sample
   import matplotlib.pyplot as plt
   plt.plot(data['t'][:1000], data['S'][:1000])
   plt.show()
   ```

2. **Check model predictions:**
   ```python
   # Get predictions
   model.eval()
   with torch.no_grad():
       sample_input = torch.FloatTensor([[S[0], 1, 0, 0, 0]])
       pred = model(sample_input, reset_state=True)
       print(f"Prediction: {pred.item():.4f}")
       print(f"Target: {targets[0,0]:.4f}")
   ```

3. **Visualize activations:**
   ```python
   # Check if model is learning anything
   for name, param in model.named_parameters():
       print(f"{name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")
   ```

---

## 7. Troubleshooting

### 7.1 Import Errors

**Error:** `ModuleNotFoundError: No module named 'src'`

**Solution:**
```bash
# Ensure you're in project root
cd /path/to/HW2

# Check PYTHONPATH
echo $PYTHONPATH

# Run from correct directory
python3 main.py
```

### 7.2 CUDA/GPU Issues

**Error:** `CUDA out of memory`

**Solution:**
```bash
# Reduce batch size
python3 main.py --batch-size 16

# Or use CPU
python3 main.py --device cpu
```

### 7.3 Training Crashes

**Error:** `RuntimeError: Expected hidden[0] size (1, 32, 64), got (1, 16, 64)`

**Cause:** Batch size mismatch in LSTM state

**Solution:**
- This happens with L=1 model when batch size changes
- Model should handle this automatically via `init_hidden()`
- If persists, check state management code

### 7.4 Visualization Issues

**Error:** `Backend is not installed` or plots don't show

**Solution:**
```bash
# macOS/Linux
export MPLBACKEND=TkAgg

# Windows
set MPLBACKEND=TkAgg

# Or use non-interactive backend
export MPLBACKEND=Agg  # Saves files but doesn't display
```

---

## 8. Extension Ideas

### 8.1 Model Improvements

**Add attention mechanism:**
```python
class LSTMWithAttention(nn.Module):
    def __init__(self, ...):
        # ... LSTM layers ...

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        # Compute attention weights
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)

        # Apply attention
        context = torch.sum(attn_weights * lstm_out, dim=1)
        output = self.fc(context)

        return output
```

### 8.2 Data Augmentation

**Add more noise types:**
```python
def add_gaussian_noise(signal, snr_db=20):
    """Add Gaussian noise at specified SNR."""
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise
```

### 8.3 Multi-Task Learning

**Extract all frequencies simultaneously:**
```python
class MultiTaskLSTM(nn.Module):
    def __init__(self, ...):
        self.lstm = nn.LSTM(...)

        # Separate output heads for each frequency
        self.fc1 = nn.Linear(hidden_size, 1)  # 1 Hz
        self.fc2 = nn.Linear(hidden_size, 1)  # 3 Hz
        self.fc3 = nn.Linear(hidden_size, 1)  # 5 Hz
        self.fc4 = nn.Linear(hidden_size, 1)  # 7 Hz

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        # Predict all frequencies at once
        f1 = self.fc1(lstm_out)
        f2 = self.fc2(lstm_out)
        f3 = self.fc3(lstm_out)
        f4 = self.fc4(lstm_out)

        return torch.cat([f1, f2, f3, f4], dim=-1)
```

### 8.4 Real-World Application

**Apply to real audio signals:**
```python
import librosa

# Load audio file
audio, sr = librosa.load('audio_file.wav')

# Resample to 1000 Hz
audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=1000)

# Use trained model to extract frequency components
model.eval()
with torch.no_grad():
    for t in range(len(audio_resampled)):
        input_vec = torch.FloatTensor([[audio_resampled[t], 1, 0, 0, 0]])
        prediction = model(input_vec)
        # ... store predictions ...
```

---

## Conclusion

This implementation guide provides everything needed to understand, run, and extend the LSTM frequency extraction system. The modular code structure makes it easy to experiment with different approaches and hyperparameters.

**Key Takeaways:**
1. **L=1 approach** teaches LSTM state management fundamentals
2. **L>1 approach** provides better performance and is more practical
3. **Proper data handling** (no shuffling for L=1) is critical
4. **Comprehensive evaluation** ensures robust performance

For further assistance, refer to:
- `README.md` for usage instructions
- `TECHNICAL_SPECIFICATION.md` for algorithmic details
- `PRD.md` for requirements and specifications

**Happy coding!** 🎵📊

---

**Document Status:** ✅ Complete
**Last Updated:** November 2025
