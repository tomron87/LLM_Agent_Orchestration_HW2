# Technical Specification
## LSTM Frequency Extraction System

**Version:** 1.0
**Date:** November 2025
**Authors:** Igor Nazarenko, Tom Ron, Roie Gilad
**Status:** Design Phase

---

## Table of Contents
1. [System Architecture](#1-system-architecture)
2. [Data Generation Pipeline](#2-data-generation-pipeline)
3. [Model Architecture](#3-model-architecture)
4. [Training Pipeline](#4-training-pipeline)
5. [Evaluation System](#5-evaluation-system)
6. [Implementation Details](#6-implementation-details)
7. [Algorithm Specifications](#7-algorithm-specifications)

---

## 1. System Architecture

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     LSTM Frequency Extraction System         │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌──────────────┐
│ Data          │     │ Model         │     │ Evaluation   │
│ Generation    │────▶│ Training      │────▶│ & Viz        │
└───────────────┘     └───────────────┘     └──────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
   Datasets            Model Weights          Metrics & Graphs
```

### 1.2 Module Structure

```
project/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── signal_generator.py      # Synthetic signal creation
│   │   ├── dataset.py                # PyTorch Dataset classes
│   │   └── data_loader.py            # Data loading utilities
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm_stateful.py          # L=1 implementation
│   │   ├── lstm_sequence.py          # L>1 implementation
│   │   └── base_model.py             # Base model interface
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                # Training loop
│   │   ├── config.py                 # Training configuration
│   │   └── callbacks.py              # Early stopping, checkpoints
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py                # MSE, generalization metrics
│       └── visualization.py          # Plotting functions
│
├── outputs/
│   ├── figures/                      # Generated plots
│   ├── models/                       # Saved model weights
│   └── logs/                         # Training logs
│
├── documentation/                    # All markdown docs
├── notebooks/                        # Jupyter demonstrations
└── main.py                           # Main execution script
```

---

## 2. Data Generation Pipeline

### 2.1 Signal Generator Class

```python
class SignalGenerator:
    """
    Generates synthetic mixed signals with noisy frequency components.

    Attributes:
        frequencies: List of Hz values [1, 3, 5, 7]
        fs: Sampling rate in Hz (1000)
        duration: Signal duration in seconds (10)
        seed: Random seed for reproducibility
    """

    def __init__(self, frequencies, fs, duration, seed):
        self.frequencies = frequencies  # [1, 3, 5, 7]
        self.fs = fs                    # 1000 Hz
        self.duration = duration        # 10 seconds
        self.num_samples = int(fs * duration)  # 10,000
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def generate_noisy_component(self, freq_idx, t):
        """
        Generate one noisy sinusoidal component.

        Args:
            freq_idx: Index of frequency (0-3)
            t: Time array

        Returns:
            noisy_signal: Array of noisy sinusoid values
        """
        freq = self.frequencies[freq_idx]

        # Random amplitude for EACH sample
        A = self.rng.uniform(0.8, 1.2, size=len(t))

        # Random phase for EACH sample
        phi = self.rng.uniform(0, 2*np.pi, size=len(t))

        # Noisy sinusoid
        signal = A * np.sin(2 * np.pi * freq * t + phi)

        return signal

    def generate_mixed_signal(self):
        """
        Generate mixed signal S(t) = average of 4 noisy components.

        Returns:
            S: Mixed signal array (10,000 samples)
            t: Time array
        """
        t = np.arange(0, self.duration, 1/self.fs)

        # Generate all 4 noisy components
        components = []
        for i in range(4):
            comp = self.generate_noisy_component(i, t)
            components.append(comp)

        # Mix: average of all components
        S = np.mean(components, axis=0)

        return S, t

    def generate_ground_truth(self, freq_idx, t):
        """
        Generate pure sinusoid (ground truth target).

        Args:
            freq_idx: Index of frequency (0-3)
            t: Time array

        Returns:
            target: Pure sinusoid array
        """
        freq = self.frequencies[freq_idx]
        target = np.sin(2 * np.pi * freq * t)
        return target
```

### 2.2 Dataset Structure

#### Training Dataset Format
```python
{
    'S': np.array([10000]),           # Mixed signal
    't': np.array([10000]),           # Time points
    'targets': np.array([4, 10000]),  # Ground truth for each frequency
    'seed': 1
}
```

#### PyTorch Dataset Class
```python
class FrequencyExtractionDataset(Dataset):
    """
    PyTorch Dataset for LSTM frequency extraction.

    Each sample contains:
        - S[t]: Scalar mixed signal value
        - C: One-hot frequency selector [4]
        - target: Ground truth pure sinusoid value
        - t: Time value (for reference)
    """

    def __init__(self, S, targets, frequencies=[1,3,5,7]):
        """
        Args:
            S: Mixed signal array (10000,)
            targets: Ground truth array (4, 10000)
            frequencies: List of frequency values
        """
        self.S = S
        self.targets = targets
        self.num_frequencies = len(frequencies)
        self.num_samples = len(S)

    def __len__(self):
        return self.num_frequencies * self.num_samples  # 40,000

    def __getitem__(self, idx):
        """
        Returns:
            input: [S[t], C1, C2, C3, C4] shape (5,)
            target: scalar ground truth value
            freq_idx: frequency index (for state management)
            sample_idx: sample index within frequency
        """
        freq_idx = idx // self.num_samples
        sample_idx = idx % self.num_samples

        # One-hot encoding
        C = np.zeros(self.num_frequencies)
        C[freq_idx] = 1

        # Input: concatenate S[t] and C
        input_vec = np.concatenate([
            [self.S[sample_idx]],
            C
        ])

        # Target
        target = self.targets[freq_idx, sample_idx]

        return {
            'input': torch.FloatTensor(input_vec),
            'target': torch.FloatTensor([target]),
            'freq_idx': freq_idx,
            'sample_idx': sample_idx
        }
```

### 2.3 Data Generation Algorithm

```
Algorithm: Generate_Training_Data(seed=1)

1. Initialize SignalGenerator with seed
2. Generate time array: t = [0, 0.001, 0.002, ..., 9.999]
3. Generate mixed signal S(t):
   For each frequency i in [1, 3, 5, 7]:
       For each sample t:
           A_i(t) ~ Uniform(0.8, 1.2)
           phi_i(t) ~ Uniform(0, 2π)
           Sinus_i^noisy(t) = A_i(t) * sin(2π * f_i * t + phi_i(t))
   S(t) = (1/4) * Σ Sinus_i^noisy(t)

4. Generate ground truth targets:
   For each frequency i:
       Target_i(t) = sin(2π * f_i * t)

5. Create dataset structure with 40,000 rows:
   Rows 1-10000:    [S[0:10000], C=[1,0,0,0], Target_1[0:10000]]
   Rows 10001-20000: [S[0:10000], C=[0,1,0,0], Target_2[0:10000]]
   Rows 20001-30000: [S[0:10000], C=[0,0,1,0], Target_3[0:10000]]
   Rows 30001-40000: [S[0:10000], C=[0,0,0,1], Target_4[0:10000]]

6. Save to disk and return dataset

Algorithm: Generate_Test_Data(seed=2)
   Same as above but with seed=2
```

---

## 3. Model Architecture

### 3.1 Implementation 1: Stateful LSTM (L=1)

#### 3.1.1 Architecture Diagram
```
Input: [S[t], C1, C2, C3, C4]  (5,)
         │
         ▼
    ┌─────────┐
    │ LSTM    │  Hidden Size: H
    │ Cell    │
    └─────────┘
         │
    (h_t, c_t) ← State preserved between samples
         │
         ▼
    ┌─────────┐
    │ Linear  │  H → 1
    └─────────┘
         │
         ▼
    Output: scalar (pure frequency value)
```

#### 3.1.2 Class Definition
```python
class StatefulLSTM(nn.Module):
    """
    LSTM with L=1: Processes one sample at a time with manual state management.

    Critical: Internal state (h_t, c_t) is preserved between consecutive samples
    within the same frequency sequence.
    """

    def __init__(self, input_size=5, hidden_size=64, num_layers=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Output layer
        self.fc = nn.Linear(hidden_size, 1)

        # State storage
        self.hidden_state = None
        self.cell_state = None

    def init_hidden(self, batch_size=1, device='cpu'):
        """Initialize hidden and cell states to zeros."""
        self.hidden_state = torch.zeros(
            self.num_layers, batch_size, self.hidden_size
        ).to(device)
        self.cell_state = torch.zeros(
            self.num_layers, batch_size, self.hidden_size
        ).to(device)

    def forward(self, x, reset_state=False):
        """
        Args:
            x: Input tensor (batch, 1, 5) - note sequence length = 1
            reset_state: If True, reset internal state to zeros

        Returns:
            output: Predicted value (batch, 1)
        """
        batch_size = x.size(0)

        # Reset state if requested (e.g., new frequency sequence)
        if reset_state or self.hidden_state is None:
            self.init_hidden(batch_size, x.device)

        # LSTM forward pass
        lstm_out, (h_new, c_new) = self.lstm(
            x,
            (self.hidden_state, self.cell_state)
        )

        # Update internal state (CRITICAL!)
        self.hidden_state = h_new.detach()
        self.cell_state = c_new.detach()

        # Output layer
        output = self.fc(lstm_out[:, -1, :])  # Use last (only) timestep

        return output
```

#### 3.1.3 State Management Protocol

**Training Loop for L=1:**
```python
def train_stateful_lstm(model, dataloader, optimizer, criterion):
    model.train()
    current_freq = None

    for batch in dataloader:
        inputs = batch['input']           # (batch, 5)
        targets = batch['target']         # (batch, 1)
        freq_idx = batch['freq_idx']      # (batch,)
        sample_idx = batch['sample_idx']  # (batch,)

        # Check if we've moved to a new frequency sequence
        if freq_idx[0] != current_freq or sample_idx[0] == 0:
            model.forward(inputs.unsqueeze(1), reset_state=True)
            current_freq = freq_idx[0]

        # Forward pass (state is preserved automatically)
        inputs = inputs.unsqueeze(1)  # Add sequence dimension: (batch, 1, 5)
        outputs = model(inputs, reset_state=False)

        # Compute loss and backprop
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 3.2 Implementation 2: Sequence LSTM (L>1)

#### 3.2.1 Architecture Diagram
```
Input Sequence: [[S[t], C], [S[t+1], C], ..., [S[t+L-1], C]]
                           (batch, L, 5)
                                │
                                ▼
                           ┌─────────┐
                           │ LSTM    │  Hidden Size: H
                           │ Layers  │
                           └─────────┘
                                │
                        Hidden states for all L steps
                                │
                                ▼
                           ┌─────────┐
                           │ Linear  │  H → 1
                           └─────────┘
                                │
                                ▼
                    Output Sequence: L predictions
                           (batch, L, 1)
```

#### 3.2.2 Class Definition
```python
class SequenceLSTM(nn.Module):
    """
    LSTM with L>1: Processes sequences of length L.

    Standard sequence-to-sequence approach leveraging LSTM's
    built-in temporal processing.
    """

    def __init__(self, input_size=5, hidden_size=64, num_layers=2,
                 sequence_length=10, dropout=0.2):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length

        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Args:
            x: Input tensor (batch, L, 5)

        Returns:
            output: Predictions for all L timesteps (batch, L, 1)
        """
        # LSTM forward (state initialized internally)
        lstm_out, _ = self.lstm(x)  # (batch, L, hidden_size)

        # Apply output layer to all timesteps
        output = self.fc(lstm_out)  # (batch, L, 1)

        return output
```

#### 3.2.3 Sequence Creation
```python
def create_sequences(S, targets, C, L=10):
    """
    Create sliding window sequences for L>1 approach.

    Args:
        S: Mixed signal (10000,)
        targets: Ground truth (10000,)
        C: One-hot vector (4,)
        L: Sequence length

    Returns:
        X: Input sequences (N, L, 5)
        Y: Target sequences (N, L, 1)
    """
    N = len(S) - L + 1  # Number of sequences
    X = np.zeros((N, L, 5))
    Y = np.zeros((N, L, 1))

    for i in range(N):
        for j in range(L):
            X[i, j, 0] = S[i + j]
            X[i, j, 1:5] = C
            Y[i, j, 0] = targets[i + j]

    return X, Y
```

### 3.3 Model Hyperparameters

| Hyperparameter | L=1 Model | L>1 Model | Notes |
|----------------|-----------|-----------|-------|
| Input Size | 5 | 5 | S[t] + 4-dim one-hot |
| Hidden Size | 64-128 | 64-128 | To be tuned |
| Num Layers | 1-2 | 2-3 | L>1 benefits from depth |
| Dropout | 0.0 | 0.2-0.3 | L>1 needs regularization |
| Sequence Length | 1 | 10-50 | To be determined |
| Output Size | 1 | 1 | Scalar prediction |

---

## 4. Training Pipeline

### 4.1 Training Configuration

```python
class TrainingConfig:
    """Configuration for training pipeline."""

    # Model settings
    model_type: str = 'stateful'  # or 'sequence'
    hidden_size: int = 64
    num_layers: int = 1
    sequence_length: int = 1

    # Training settings
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 50
    optimizer: str = 'adam'

    # Loss and metrics
    loss_function: str = 'mse'

    # Regularization
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    dropout: float = 0.0

    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4

    # Checkpointing
    save_best_only: bool = True
    checkpoint_dir: str = 'outputs/models'

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Logging
    log_interval: int = 100
    verbose: bool = True
```

### 4.2 Training Algorithm

```
Algorithm: Train_LSTM(model, train_loader, val_loader, config)

1. Initialize:
   - optimizer = Adam(model.parameters(), lr=config.learning_rate)
   - criterion = MSELoss()
   - best_val_loss = infinity
   - patience_counter = 0

2. For epoch in range(num_epochs):

   a. Training Phase:
      model.train()
      train_loss = 0

      For batch in train_loader:
          # Handle state management (if L=1)
          if model_type == 'stateful':
              check and reset state if new frequency

          # Forward pass
          outputs = model(inputs)
          loss = criterion(outputs, targets)

          # Backward pass
          optimizer.zero_grad()
          loss.backward()
          clip_grad_norm_(model.parameters(), max_norm=1.0)
          optimizer.step()

          train_loss += loss.item()

      avg_train_loss = train_loss / len(train_loader)

   b. Validation Phase:
      model.eval()
      val_loss = 0

      with torch.no_grad():
          For batch in val_loader:
              outputs = model(inputs)
              loss = criterion(outputs, targets)
              val_loss += loss.item()

      avg_val_loss = val_loss / len(val_loader)

   c. Logging:
      print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f},
                           Val Loss = {avg_val_loss:.6f}")

   d. Early Stopping Check:
      if avg_val_loss < best_val_loss - min_delta:
          best_val_loss = avg_val_loss
          save_checkpoint(model, epoch)
          patience_counter = 0
      else:
          patience_counter += 1
          if patience_counter >= patience:
              print("Early stopping triggered")
              break

   e. Learning Rate Scheduling (optional):
      scheduler.step(avg_val_loss)

3. Load best model weights
4. Return trained model and training history
```

### 4.3 Loss Function

**Mean Squared Error (MSE):**
```python
def compute_mse(predictions, targets):
    """
    Compute MSE between predictions and ground truth.

    Args:
        predictions: Model outputs (N,)
        targets: Ground truth values (N,)

    Returns:
        mse: Mean squared error
    """
    return torch.mean((predictions - targets) ** 2)
```

---

## 5. Evaluation System

### 5.1 Metrics Calculation

```python
class Evaluator:
    """Evaluation and metrics calculation."""

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device

    def compute_mse(self, dataloader):
        """
        Compute MSE over entire dataset.

        Returns:
            mse: Mean squared error
            predictions: All model outputs
            targets: All ground truth values
        """
        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)

                outputs = self.model(inputs)

                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)

        mse = np.mean((predictions - targets) ** 2)

        return mse, predictions, targets

    def check_generalization(self, train_mse, test_mse, threshold=0.2):
        """
        Check if model generalizes well.

        Args:
            train_mse: MSE on training set
            test_mse: MSE on test set
            threshold: Maximum allowed relative difference

        Returns:
            generalizes: True if test_mse ≈ train_mse
            ratio: test_mse / train_mse
        """
        ratio = test_mse / train_mse
        generalizes = abs(ratio - 1.0) < threshold

        return generalizes, ratio
```

### 5.2 Visualization System

```python
class Visualizer:
    """Generate all required visualizations."""

    def plot_single_frequency_comparison(
        self, t, target, lstm_output, mixed_signal, freq_hz,
        save_path=None
    ):
        """
        Graph 1: Detailed comparison for one frequency.

        Shows:
        - Target (pure sinusoid) as solid line
        - LSTM output as scatter/line
        - Mixed noisy signal as background
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot mixed signal (background)
        ax.plot(t, mixed_signal, 'gray', alpha=0.3,
                label='Mixed Signal S(t)', linewidth=0.5)

        # Plot target (solid line)
        ax.plot(t, target, 'g-', label='Target (Pure)', linewidth=2)

        # Plot LSTM output (dots or line)
        ax.plot(t, lstm_output, 'b.', label='LSTM Output',
                markersize=3)

        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.set_title(f'Frequency Extraction: {freq_hz} Hz', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_all_frequencies(
        self, t, targets, lstm_outputs, frequencies, save_path=None
    ):
        """
        Graph 2: Four subplots showing all frequency extractions.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        for i, (ax, freq) in enumerate(zip(axes, frequencies)):
            ax.plot(t, targets[i], 'g-', label='Target', linewidth=2)
            ax.plot(t, lstm_outputs[i], 'b-', label='LSTM',
                   linewidth=1.5, alpha=0.7)
            ax.set_title(f'Frequency: {freq} Hz', fontsize=12)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_training_curves(self, history, save_path=None):
        """Plot training and validation loss curves."""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(history['train_loss'], label='Training Loss', linewidth=2)
        ax.plot(history['val_loss'], label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('MSE Loss', fontsize=12)
        ax.set_title('Training Progress', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Log scale for better visualization

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_fft_analysis(
        self, signal, fs, label, save_path=None
    ):
        """
        Frequency domain analysis using FFT.

        Visualize spectral content of signals.
        """
        N = len(signal)
        fft_vals = np.fft.fft(signal)
        fft_freq = np.fft.fftfreq(N, 1/fs)

        # Only positive frequencies
        pos_mask = fft_freq > 0
        fft_freq = fft_freq[pos_mask]
        fft_magnitude = np.abs(fft_vals[pos_mask])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(fft_freq[:100], fft_magnitude[:100], linewidth=2)
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('Magnitude', fontsize=12)
        ax.set_title(f'FFT Analysis: {label}', fontsize=14)
        ax.grid(True, alpha=0.3)

        # Mark expected frequencies
        for f in [1, 3, 5, 7]:
            ax.axvline(x=f, color='r', linestyle='--', alpha=0.5)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
```

---

## 6. Implementation Details

### 6.1 Project Entry Point

```python
# main.py

def main():
    """Main execution pipeline."""

    # 1. Configuration
    config = TrainingConfig()

    # 2. Data Generation
    print("Generating datasets...")
    train_data = generate_dataset(seed=1)
    test_data = generate_dataset(seed=2)

    # 3. Create DataLoaders
    train_loader = create_dataloader(train_data, config)
    test_loader = create_dataloader(test_data, config)

    # 4. Initialize Model
    print(f"Initializing {config.model_type} model...")
    model = create_model(config)

    # 5. Train
    print("Training model...")
    trained_model, history = train(model, train_loader, config)

    # 6. Evaluate
    print("Evaluating model...")
    evaluator = Evaluator(trained_model)
    train_mse, _, _ = evaluator.compute_mse(train_loader)
    test_mse, predictions, targets = evaluator.compute_mse(test_loader)

    print(f"Training MSE: {train_mse:.6f}")
    print(f"Test MSE: {test_mse:.6f}")

    generalizes, ratio = evaluator.check_generalization(train_mse, test_mse)
    print(f"Generalization: {generalizes} (ratio: {ratio:.3f})")

    # 7. Visualize
    print("Generating visualizations...")
    visualizer = Visualizer()
    visualizer.plot_all_results(predictions, targets, history)

    print("Complete!")

if __name__ == '__main__':
    main()
```

### 6.2 Command-Line Interface

```bash
# Train L=1 model
python main.py --model stateful --hidden-size 64 --epochs 50

# Train L>1 model
python main.py --model sequence --sequence-length 10 --hidden-size 128 --epochs 50

# Evaluate saved model
python main.py --mode evaluate --checkpoint outputs/models/best_model.pth

# Generate visualizations only
python main.py --mode visualize --checkpoint outputs/models/best_model.pth
```

---

## 7. Algorithm Specifications

### 7.1 LSTM Forward Pass (L=1)

```
Input: x_t = [S[t], C₁, C₂, C₃, C₄]  ∈ ℝ⁵
State: h_{t-1} ∈ ℝᴴ, c_{t-1} ∈ ℝᴴ

Forget Gate:
    f_t = σ(W_f · [h_{t-1}, x_t] + b_f)

Input Gate:
    i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
    c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)

Cell State Update:
    c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t

Output Gate:
    o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
    h_t = o_t ⊙ tanh(c_t)

Output Layer:
    ŷ_t = W_out · h_t + b_out

Return: ŷ_t, (h_t, c_t)
```

### 7.2 Training Optimization

```
Loss Function:
    L = (1/N) · Σᵢ (ŷᵢ - yᵢ)²

Gradient Descent (Adam):
    θ_{t+1} = θ_t - α · m̂_t / (√v̂_t + ε)

Where:
    m_t = β₁ · m_{t-1} + (1 - β₁) · g_t
    v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²
    m̂_t = m_t / (1 - β₁ᵗ)
    v̂_t = v_t / (1 - β₂ᵗ)

Hyperparameters:
    α = 0.001 (learning rate)
    β₁ = 0.9
    β₂ = 0.999
    ε = 10⁻⁸
```

### 7.3 Gradient Clipping

```
Algorithm: Clip_Gradient_Norm(parameters, max_norm=1.0)

1. Compute total gradient norm:
   total_norm = √(Σ ||θᵢ.grad||²)

2. If total_norm > max_norm:
   clip_coef = max_norm / (total_norm + 1e-6)
   For each parameter θᵢ:
       θᵢ.grad = θᵢ.grad · clip_coef

3. Return clipped gradients
```

---

## 8. Testing & Validation

### 8.1 Unit Tests

```python
def test_signal_generation():
    """Verify signal generation produces correct shape and properties."""
    gen = SignalGenerator([1,3,5,7], fs=1000, duration=10, seed=1)
    S, t = gen.generate_mixed_signal()

    assert len(S) == 10000
    assert len(t) == 10000
    assert S.shape == t.shape
    assert -2 < S.min() < 2  # Reasonable amplitude range

def test_dataset_length():
    """Verify dataset produces correct number of samples."""
    dataset = FrequencyExtractionDataset(S, targets)
    assert len(dataset) == 40000

def test_state_preservation():
    """Verify LSTM preserves state between samples."""
    model = StatefulLSTM()
    x = torch.randn(1, 1, 5)

    # First forward pass
    model.init_hidden()
    _ = model(x, reset_state=False)
    state1 = model.hidden_state.clone()

    # Second forward pass (state should change)
    _ = model(x, reset_state=False)
    state2 = model.hidden_state

    assert not torch.equal(state1, state2)  # State changed
```

### 8.2 Integration Tests

```python
def test_full_pipeline():
    """Test complete pipeline from data generation to evaluation."""
    # Generate data
    train_data = generate_dataset(seed=1)

    # Create model
    model = StatefulLSTM()

    # Train for 1 epoch
    train(model, train_data, epochs=1)

    # Evaluate
    mse, _, _ = evaluate(model, train_data)

    assert mse < 1.0  # Should learn something in 1 epoch
```

---

## 9. Performance Optimization

### 9.1 Computational Complexity

| Operation | L=1 | L>1 (L=10) | Notes |
|-----------|-----|------------|-------|
| Forward Pass | O(H²) | O(L·H²) | Per sample |
| Memory | O(H) | O(L·H) | State storage |
| Training Time | Slow (sequential) | Faster (batched) | L>1 parallelizes better |

### 9.2 Optimization Strategies

1. **Batch Processing:** Group samples when possible
2. **GPU Acceleration:** Use CUDA if available
3. **Mixed Precision:** Use FP16 for faster training
4. **Data Loading:** Parallelize data loading with multiple workers
5. **Checkpoint Frequency:** Balance between safety and speed

---

## 10. Error Handling

### 10.1 Common Issues

```python
class StateManagementError(Exception):
    """Raised when LSTM state is not properly managed."""
    pass

class DataGenerationError(Exception):
    """Raised when data generation fails."""
    pass

def validate_state_continuity(model, dataloader):
    """
    Verify that state is preserved correctly during training.
    """
    prev_freq = None
    for batch in dataloader:
        freq_idx = batch['freq_idx'][0]

        if prev_freq is not None and freq_idx != prev_freq:
            # Should have reset state
            if not model.state_was_reset:
                raise StateManagementError(
                    f"State not reset when moving from freq {prev_freq} to {freq_idx}"
                )

        prev_freq = freq_idx
```

---

## 11. Documentation Standards

### 11.1 Code Documentation

```python
def function_name(param1: type, param2: type) -> return_type:
    """
    Brief description of function.

    Longer description with implementation details if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ExceptionType: When this exception occurs

    Example:
        >>> result = function_name(value1, value2)
        >>> print(result)
        expected_output
    """
    # Implementation
    pass
```

### 11.2 Type Hints

All functions should include complete type hints:
- Input parameters
- Return values
- Class attributes

---

**Document Status:** ✅ Complete
**Next Document:** Implementation Guide

