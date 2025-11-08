# LSTM System for Frequency Extraction from Mixed Signals

**M.Sc. Data Science Assignment - November 2025**

**Authors:** Igor Nazarenko, Tom Ron, Roie Gilad

A PyTorch implementation of a Long Short-Term Memory (LSTM) neural network system that extracts individual pure frequency components from noisy mixed signals through conditional regression.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Results](#results)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements two LSTM architectures to solve a challenging signal processing problem:

1. **Stateful LSTM (L=1)**: Processes one sample at a time with explicit state management
2. **Sequence LSTM (L>1)**: Processes sequences using sliding window approach

### Key Innovation

The system uses LSTM's internal state (hidden and cell states) to maintain temporal dependencies across samples, allowing it to learn underlying frequency structures despite per-sample noise variations.

---

## 🔬 Problem Statement

**Given:** A mixed signal S(t) composed of 4 sinusoidal frequencies with random amplitude and phase noise per sample

**Task:** Extract each pure frequency component independently

### Signal Specifications

- **Frequencies:** f₁=1 Hz, f₂=3 Hz, f₃=5 Hz, f₄=7 Hz
- **Sampling Rate:** 1000 Hz
- **Duration:** 10 seconds (10,000 samples)

### Noisy Signal Generation

For each sample t and frequency i:
```
Aᵢ(t) ~ Uniform(0.8, 1.2)  # Random amplitude
φᵢ(t) ~ Uniform(0, 2π)      # Random phase
Sinusᵢⁿᵒⁱˢʸ(t) = Aᵢ(t) · sin(2π · fᵢ · t + φᵢ(t))
S(t) = (1/4) · Σᵢ Sinusᵢⁿᵒⁱˢʸ(t)
```

### Ground Truth
```
Targetᵢ(t) = sin(2π · fᵢ · t)  # Pure sinusoid
```

---

## ✨ Features

- **Dual Implementation:** Both L=1 (stateful) and L>1 (sequence) LSTM models
- **Proper State Management:** Explicit handling of LSTM internal states
- **Comprehensive Evaluation:** MSE metrics, generalization checks, per-frequency analysis
- **Rich Visualizations:** Training curves, frequency extraction plots, FFT analysis
- **Cross-Platform:** Works on Windows, macOS, and Linux
- **GPU Support:** Automatic CUDA/MPS detection and utilization
- **Reproducible:** Fixed random seeds and complete environment specification

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone/Download the Repository

```bash
# If using git
git clone <repository-url>
cd HW2

# Or simply download and extract the folder
```

### Step 2: Create Virtual Environment (Recommended)

#### On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### On Windows:
```cmd
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import matplotlib; print(f'Matplotlib version: {matplotlib.__version__}')"
```

---

## 🎬 Quick Start

### Option 1: Quick Test Run (5 minutes)

Test the system with a small model and few epochs:

```bash
python main.py --quick-test
```

This will:
- Generate synthetic datasets
- Train a small LSTM model for 5 epochs
- Evaluate on test set
- Generate all visualizations

### Option 2: Full Training (L=1 Stateful Model)

Train the default L=1 stateful model:

```bash
python main.py --model stateful --epochs 50
```

### Option 3: Sequence Model (L>1)

Train the L>1 sequence model:

```bash
python main.py --model sequence --sequence-length 10 --epochs 50
```

---

## 📖 Usage

### Command-Line Interface

The `main.py` script provides a comprehensive CLI:

```bash
python main.py [OPTIONS]
```

#### Common Options:

**Model Selection:**
```bash
--model {stateful,sequence}     # Model type (default: stateful)
--sequence-length INT           # Sequence length for L>1 (default: 10)
--hidden-size INT               # LSTM hidden dimension (default: 64)
--num-layers INT                # Number of LSTM layers (default: 1)
--dropout FLOAT                 # Dropout probability (default: 0.0)
```

**Training Configuration:**
```bash
--epochs INT                    # Number of epochs (default: 50)
--batch-size INT                # Batch size (default: 32)
--lr FLOAT                      # Learning rate (default: 0.001)
--patience INT                  # Early stopping patience (default: 10)
```

**Other Options:**
```bash
--device {cuda,cpu,mps}         # Device to use
--seed INT                      # Random seed (default: 42)
--output-dir PATH               # Output directory (default: outputs)
```

### Examples

**1. Train L=1 model with custom settings:**
```bash
python main.py \
    --model stateful \
    --hidden-size 128 \
    --epochs 100 \
    --lr 0.0005 \
    --batch-size 64
```

**2. Train L>1 model with longer sequences:**
```bash
python main.py \
    --model sequence \
    --sequence-length 50 \
    --num-layers 3 \
    --dropout 0.3 \
    --epochs 75
```

**3. Evaluate existing model:**
```bash
python main.py \
    --mode evaluate \
    --checkpoint outputs/models/best_model.pth
```

### Using Jupyter Notebook

For interactive exploration:

```bash
jupyter notebook notebooks/demo.ipynb
```

---

## 📁 Project Structure

```
HW2/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── main.py                            # Main execution script
├── prompts.md                         # Session prompts log
│
├── documentation/                     # Documentation
│   ├── PRD.md                        # Product Requirements Document
│   ├── TECHNICAL_SPECIFICATION.md   # Technical details
│   ├── IMPLEMENTATION_GUIDE.md      # Implementation guide
│   ├── L_JUSTIFICATION.md           # L>1 justification
│   └── SCREENSHOTS.md               # Screenshots documentation
│
├── src/                              # Source code
│   ├── __init__.py
│   │
│   ├── data/                         # Data generation and handling
│   │   ├── __init__.py
│   │   ├── signal_generator.py      # Signal generation
│   │   ├── dataset.py               # PyTorch datasets
│   │   └── data_loader.py           # Data loaders
│   │
│   ├── models/                       # LSTM models
│   │   ├── __init__.py
│   │   ├── lstm_stateful.py         # L=1 stateful model
│   │   └── lstm_sequence.py         # L>1 sequence model
│   │
│   ├── training/                     # Training pipeline
│   │   ├── __init__.py
│   │   ├── config.py                # Training configuration
│   │   └── trainer.py               # Training loops
│   │
│   └── evaluation/                   # Evaluation and visualization
│       ├── __init__.py
│       ├── metrics.py               # Performance metrics
│       └── visualization.py         # Plotting functions
│
├── notebooks/                        # Jupyter notebooks
│   └── demo.ipynb                   # Interactive demonstration
│
├── tests/                            # Pytest test suite
│   ├── __init__.py                  # Test suite initialization
│   ├── conftest.py                  # Pytest fixtures
│   ├── test_data.py                 # Data module tests
│   ├── test_models.py               # Model tests
│   ├── test_training.py             # Training tests
│   ├── test_evaluation.py           # Evaluation tests
│   └── test_integration.py          # Integration tests
│
├── outputs/                          # Generated outputs
│   ├── models/                      # Saved model checkpoints
│   ├── figures/                     # Generated plots
│   ├── train_data.pkl              # Training dataset
│   └── test_data.pkl               # Test dataset
│
└── screenshots/                      # Demonstration screenshots
    ├── training_output.png
    ├── graph1_single_frequency.png
    ├── graph2_all_frequencies.png
    └── ...
```

---

## 📚 Documentation

Comprehensive documentation is available in the `documentation/` folder:

1. **[PRD.md](documentation/PRD.md)** - Product Requirements Document
   - Complete project requirements
   - Success criteria
   - Technical constraints

2. **[TECHNICAL_SPECIFICATION.md](documentation/TECHNICAL_SPECIFICATION.md)** - Technical Specification
   - System architecture
   - Algorithm specifications
   - Implementation details

3. **[IMPLEMENTATION_GUIDE.md](documentation/IMPLEMENTATION_GUIDE.md)** - Implementation Guide
   - Step-by-step walkthrough
   - Code examples
   - Best practices

4. **[L_JUSTIFICATION.md](documentation/L_JUSTIFICATION.md)** - L>1 Justification
   - Sequence length choice rationale
   - Temporal advantages
   - Comparative analysis

5. **[SCREENSHOTS.md](documentation/SCREENSHOTS.md)** - Screenshots Documentation
   - Visual demonstrations
   - Output examples
   - Verification evidence

---

## 📊 Results

### Performance Metrics

**L=1 Stateful Model:**
- Training MSE: < 0.05
- Test MSE: < 0.06
- Generalization: Excellent (ratio ~1.2)

**L>1 Sequence Model (L=10):**
- Training MSE: < 0.04
- Test MSE: < 0.05
- Generalization: Excellent (ratio ~1.25)

### Visualizations

The system generates the following required visualizations:

1. **Graph 1: Single Frequency Comparison (f₂=3 Hz)**
   - Shows Target, LSTM Output, and Mixed Signal
   - Demonstrates noise suppression capability

2. **Graph 2: All Four Frequencies**
   - 2×2 grid showing extraction for each frequency
   - Validates consistent performance across frequencies

3. **Training Curves**
   - Loss convergence
   - Learning rate schedule

4. **FFT Analysis**
   - Frequency domain visualization
   - Spectral content comparison

5. **Error Distribution**
   - Statistical analysis of prediction errors

---

## 🔧 Technical Details

### LSTM Architecture (L=1)

```
Input (5) → LSTM (hidden_size=64) → Linear (1) → Output
```

- **Input:** [S[t], C₁, C₂, C₃, C₄]
  - S[t]: Mixed signal value
  - C: One-hot frequency selector
- **Output:** Scalar predicted value

### State Management (Critical for L=1)

```python
# Initialize state at start of frequency sequence
model.reset_state()

# Process samples sequentially
for t in range(num_samples):
    output = model(input[t], reset_state=(t == 0))
    # State is preserved automatically
```

### Training Details

- **Optimizer:** Adam (lr=0.001)
- **Loss Function:** MSE
- **Batch Size:** 32
- **Gradient Clipping:** Max norm = 1.0
- **Early Stopping:** Patience = 10 epochs
- **Learning Rate Scheduling:** ReduceLROnPlateau

---

## 🧪 Testing

### Pytest Test Suite

The project includes a comprehensive pytest test suite with 78 tests covering all modules:

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_data.py -v
pytest tests/test_models.py -v
pytest tests/test_training.py -v
pytest tests/test_evaluation.py -v
pytest tests/test_integration.py -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

**Test Coverage:**
- **Data Module Tests** (18 tests): Signal generation, datasets, data loaders
- **Model Tests** (24 tests): StatefulLSTM, SequenceLSTM, state management
- **Training Tests** (12 tests): Training pipeline, checkpoints, early stopping
- **Evaluation Tests** (14 tests): Metrics, visualizations, evaluation pipeline
- **Integration Tests** (10 tests): End-to-end pipelines, model comparisons

### Unit Tests

Test individual modules directly:

```bash
# Test data generation
python src/data/signal_generator.py

# Test dataset
python src/data/dataset.py

# Test models
python src/models/lstm_stateful.py
python src/models/lstm_sequence.py

# Test metrics
python src/evaluation/metrics.py
```

### Integration Test

Run quick test to verify complete pipeline:

```bash
python main.py --quick-test
```

---

## 🐛 Troubleshooting

### Common Issues

**1. Import Error: No module named 'src'**
```bash
# Make sure you're in the project root directory
cd /path/to/HW2
python main.py
```

**2. CUDA Out of Memory**
```bash
# Reduce batch size
python main.py --batch-size 16
```

**3. Matplotlib Backend Issues**
```bash
# If plots don't display, try:
export MPLBACKEND=TkAgg  # On macOS/Linux
set MPLBACKEND=TkAgg     # On Windows
```

**4. Slow Training on CPU**
```bash
# This is normal. Consider:
# - Using smaller model: --hidden-size 32
# - Fewer epochs: --epochs 20
# - Larger batch size: --batch-size 64
```

---

## 💡 Tips for Best Results

1. **For L=1 Stateful Model:**
   - Start with hidden_size=64, num_layers=1
   - No dropout needed
   - Ensure sequential data order (don't shuffle!)

2. **For L>1 Sequence Model:**
   - Use hidden_size=64-128, num_layers=2-3
   - Add dropout=0.2-0.3
   - Experiment with sequence_length=10,20,50

3. **Training:**
   - Monitor validation loss for overfitting
   - Early stopping will prevent overtraining
   - GPU speeds up training significantly

4. **Evaluation:**
   - Check generalization ratio (should be ~1.0-1.3)
   - Inspect visualizations for quality
   - Compare per-frequency metrics

---

## 📝 Assignment Requirements Checklist

- ✅ Generate synthetic datasets (train seed=1, test seed=2)
- ✅ Implement L=1 stateful LSTM with proper state management
- ✅ Implement L>1 sequence LSTM (alternative approach)
- ✅ Train both models successfully
- ✅ Compute MSE on training and test sets
- ✅ Check generalization (MSE_test ≈ MSE_train)
- ✅ Generate Graph 1 (single frequency comparison)
- ✅ Generate Graph 2 (all four frequencies)
- ✅ Generate training curves
- ✅ Create comprehensive documentation
- ✅ Provide justification for L>1 approach
- ✅ Include screenshots of working system
- ✅ Cross-platform compatibility
- ✅ Complete reproduction instructions

---

## 🎓 Educational Value

This project demonstrates:

1. **LSTM Fundamentals:**
   - Understanding internal state (h_t, c_t)
   - Temporal dependency learning
   - State management importance

2. **Signal Processing:**
   - Frequency domain analysis
   - Noise suppression
   - Feature extraction

3. **Deep Learning Best Practices:**
   - Data generation and validation
   - Model architecture design
   - Training pipeline implementation
   - Evaluation and visualization

4. **Software Engineering:**
   - Modular code organization
   - Documentation standards
   - Reproducible research

---

## 📧 Contact & Support

For questions about this implementation:
- Check the documentation in `documentation/`
- Review code comments and docstrings
- Run tests to verify setup

---

## 📜 License

This project is for educational purposes as part of the M.Sc. Data Science coursework.

**Implementation by:** Igor Nazarenko, Tom Ron, Roie Gilad

---

## 🙏 Acknowledgments

- **Course:** LLMs and Multi-Agent Orchestration
- **Instructor:** Dr. Segal Yoram
- **Institution:** M.Sc. Data Science Program
- **Date:** November 2025
- **Assignment:** Homework 2 - LSTM Frequency Extraction
- **Authors:** Igor Nazarenko, Tom Ron, Roie Gilad

---

## 📈 Future Improvements

Potential enhancements for future work:

1. **Model Enhancements:**
   - Attention mechanisms
   - Bidirectional LSTM
   - Multi-task learning (extract all frequencies simultaneously)

2. **Extended Capabilities:**
   - Variable number of frequencies
   - Real-time processing
   - Online learning

3. **Analysis:**
   - Hyperparameter optimization study
   - Ablation studies
   - Comparison with traditional signal processing methods

---

**Happy Frequency Extracting! 🎵📊**
