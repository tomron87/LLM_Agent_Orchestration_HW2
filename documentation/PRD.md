# Product Requirements Document (PRD)
## LSTM System for Frequency Extraction from Mixed Signals

**Document Version:** 1.0
**Date:** November 2025
**Assignment by:** Dr. Segal Yoram
**Implementation by:** Igor Nazarenko, Tom Ron, Roie Gilad
**Course:** M.Sc. Data Science - LLMs and Multi-Agent Orchestration

---

## 1. Executive Summary

### 1.1 Product Overview
This project implements a Long Short-Term Memory (LSTM) neural network system capable of extracting individual pure frequency components from a noisy, mixed signal. The system uses conditional regression to selectively extract one of four target frequencies based on a one-hot encoded selection vector.

### 1.2 Problem Statement
Given a mixed signal S(t) composed of 4 sinusoidal frequencies with randomly varying amplitude and phase (noise), the system must learn to extract each pure frequency component independently while ignoring the noise and other frequency components.

### 1.3 Key Innovation
The critical innovation is the use of LSTM's internal state (hidden state hₜ and cell state cₜ) to maintain temporal dependency across samples, allowing the network to learn the underlying frequency structure despite per-sample noise variations.

---

## 2. Product Goals & Success Criteria

### 2.1 Primary Goals
1. **Accurate Frequency Extraction:** Extract pure sinusoids from noisy mixed signal with low MSE
2. **Generalization:** Demonstrate ability to handle unseen noise patterns (different random seed)
3. **State Management:** Properly utilize LSTM's temporal memory capabilities
4. **Educational Value:** Provide clear, reproducible implementation for learning purposes

### 2.2 Success Metrics

#### Quantitative Metrics
| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| MSE (Training) | < 0.1 | Mean Squared Error on 40,000 training samples |
| MSE (Test) | < 0.1 | Mean Squared Error on 40,000 test samples |
| Generalization | MSE_test ≈ MSE_train (within 20%) | Ratio comparison |
| Training Time | < 30 minutes | On standard GPU/CPU |

#### Qualitative Metrics
- Visual quality of extracted signals matches ground truth
- Clear separation of frequencies in visualizations
- Noise suppression visible in outputs

### 2.3 Non-Goals
- Real-time signal processing
- Handling more than 4 frequencies
- Variable sampling rates
- Online learning or adaptation

---

## 3. Technical Requirements

### 3.1 Signal Specifications

#### 3.1.1 Frequency Parameters
```
f₁ = 1 Hz
f₂ = 3 Hz
f₃ = 5 Hz
f₄ = 7 Hz
```

#### 3.1.2 Sampling Parameters
```
Sampling Rate (Fs): 1000 Hz
Time Range: 0 to 10 seconds
Total Samples: 10,000
Time Step (dt): 0.001 seconds
```

#### 3.1.3 Noisy Signal Generation

For each sample t and frequency i:

**1. Random Amplitude:**
```
Aᵢ(t) ~ Uniform(0.8, 1.2)
```

**2. Random Phase:**
```
φᵢ(t) ~ Uniform(0, 2π)
```

**3. Noisy Sinusoid:**
```
Sinusᵢⁿᵒⁱˢʸ(t) = Aᵢ(t) · sin(2π · fᵢ · t + φᵢ(t))
```

**4. Mixed Signal (System Input):**
```
S(t) = (1/4) · Σᵢ₌₁⁴ Sinusᵢⁿᵒⁱˢʸ(t)
```

#### 3.1.4 Ground Truth Targets

For each frequency i:
```
Targetᵢ(t) = sin(2π · fᵢ · t)
```

Pure sinusoid with:
- Unit amplitude
- Zero phase
- No noise

### 3.2 Dataset Requirements

#### 3.2.1 Training Dataset
- **Size:** 40,000 rows
- **Composition:** 10,000 samples × 4 frequencies
- **Random Seed:** #1
- **Format:** Each row contains `[S[t], C₁, C₂, C₃, C₄]`
  - S[t]: Scalar, mixed noisy signal value
  - C: 4-dimensional one-hot vector for frequency selection

**Example Rows:**
| Row | t (sec) | S[t] | C | Target Output |
|-----|---------|------|---|---------------|
| 1 | 0.000 | 0.8124 | [1,0,0,0] | 0.0000 |
| 10000 | 9.999 | 0.8124 | [1,0,0,0] | 0.0000 |
| 10001 | 0.000 | 0.8124 | [0,1,0,0] | 0.0000 |
| 10002 | 0.001 | 0.7932 | [0,1,0,0] | 0.0188 |

#### 3.2.2 Test Dataset
- **Size:** 40,000 rows
- **Composition:** 10,000 samples × 4 frequencies
- **Random Seed:** #2 (DIFFERENT from training)
- **Format:** Identical structure to training set
- **Purpose:** Evaluate generalization to unseen noise patterns

**Critical Note:** Same frequencies, same time points, but completely different random amplitude and phase values.

### 3.3 Model Architecture Requirements

#### 3.3.1 Input Specifications
```
Input Vector: [S[t], C₁, C₂, C₃, C₄]
- S[t]: Scalar (mixed signal value)
- C: 4-dimensional one-hot vector
Total Input Dimension: 5
```

#### 3.3.2 LSTM Configuration
```
Type: Long Short-Term Memory (LSTM)
Input Size: 5
Hidden Size: [To be determined through experimentation]
Number of Layers: [To be determined]
Output Size: 1 (single scalar value)
```

#### 3.3.3 Sequence Length (L) - Two Implementations Required

**Implementation 1: L = 1 (Default - Stateful)**
- Process one sample at a time
- **CRITICAL:** Manually preserve internal state (hₜ, cₜ) between consecutive samples
- **Must NOT:** Reset state between samples within a frequency sequence
- **Must Reset:** State when switching between different frequencies
- **Pedagogical Value:** Demonstrates explicit state management and temporal dependency

**State Management Protocol for L=1:**
```python
# Initialize state at start of frequency sequence
h₀, c₀ = initialize_zero_state()

# Process each sample sequentially
for t in range(num_samples):
    output, (h_t, c_t) = LSTM(input[t], (h_prev, c_prev))
    # DO NOT RESET - pass state to next iteration
    h_prev, c_prev = h_t, c_t
```

**Implementation 2: L > 1 (Alternative - Sliding Window)**
- Process sequences of length L (e.g., L=10 or L=50)
- Leverage LSTM's built-in temporal processing
- Standard sequence-to-sequence approach
- **Requires:** Detailed justification document explaining:
  - Choice of L value
  - Contribution to temporal advantage
  - Output handling strategy

### 3.4 Training Requirements

#### 3.4.1 Loss Function
```
Loss: Mean Squared Error (MSE)
MSE = (1/N) · Σⱼ₌₁ᴺ (LSTM_output[j] - Target[j])²
```

#### 3.4.2 Optimization
- **Optimizer:** Adam (recommended) or SGD
- **Learning Rate:** To be determined through experimentation
- **Batch Size:** To be determined (consider memory constraints with L=1)
- **Epochs:** Until convergence (monitor validation loss)

#### 3.4.3 Training Protocol
1. Train on full 40,000 training samples
2. Monitor both training and validation loss
3. Implement early stopping if validation loss plateaus
4. Save best model based on validation performance

### 3.5 Evaluation Requirements

#### 3.5.1 Required Metrics

**1. Training MSE:**
```
MSE_train = (1/40000) · Σⱼ₌₁⁴⁰⁰⁰⁰ (LSTM(S_train[j], C) - Target[j])²
```

**2. Test MSE:**
```
MSE_test = (1/40000) · Σⱼ₌₁⁴⁰⁰⁰⁰ (LSTM(S_test[j], C) - Target[j])²
```

**3. Generalization Check:**
```
If MSE_test ≈ MSE_train (within 20%), then system generalizes well
```

#### 3.5.2 Required Visualizations

**Graph 1: Single Frequency Detailed Comparison (f₂ = 3 Hz)**
- **Components:**
  1. Target (pure sinusoid) - solid line
  2. LSTM Output - scatter points/line
  3. S (noisy mixed signal) - background/faded
- **Time Range:** Show representative portion (e.g., 0-2 seconds)
- **Purpose:** Demonstrate extraction quality and noise suppression

**Graph 2: All Four Frequencies (4 Subplots)**
- **Layout:** 2×2 grid or 4×1 column
- **Each Subplot Shows:**
  - Target frequency fᵢ
  - LSTM extraction for that frequency
- **Purpose:** Show consistent performance across all frequencies

#### 3.5.3 Additional Recommended Visualizations
1. **Training Curves:**
   - Loss vs. Epoch
   - Training vs. Validation loss

2. **Frequency Domain Analysis:**
   - FFT of extracted signals
   - Compare spectral content: Target vs. LSTM Output vs. Mixed Signal

3. **Hyperparameter Study:**
   - Effect of hidden size
   - Effect of learning rate
   - Effect of sequence length (L)

---

## 4. Functional Requirements

### 4.1 Data Generation Module
**FR-1:** Generate synthetic dataset with specified parameters
**FR-2:** Support configurable random seeds for reproducibility
**FR-3:** Export datasets in structured format (CSV or NumPy arrays)
**FR-4:** Validate signal properties (frequency content, amplitude ranges)

### 4.2 Model Implementation Module
**FR-5:** Implement LSTM with L=1 (stateful mode)
**FR-6:** Implement LSTM with L>1 (sliding window mode)
**FR-7:** Support state preservation and reset functionality
**FR-8:** Configurable hyperparameters (hidden size, layers, etc.)

### 4.3 Training Module
**FR-9:** Train models on generated datasets
**FR-10:** Monitor and log training metrics
**FR-11:** Implement model checkpointing
**FR-12:** Support GPU acceleration if available

### 4.4 Evaluation Module
**FR-13:** Calculate MSE on training and test sets
**FR-14:** Generate all required visualizations
**FR-15:** Export evaluation results (metrics, graphs)
**FR-16:** Compare performance between L=1 and L>1 implementations

### 4.5 Documentation Module
**FR-17:** Comprehensive README with setup instructions
**FR-18:** Technical documentation explaining implementation
**FR-19:** Justification document for L>1 approach
**FR-20:** Screenshots demonstrating working system

---

## 5. Non-Functional Requirements

### 5.1 Usability
- **NFR-1:** Clear command-line interface for running experiments
- **NFR-2:** Jupyter notebook for interactive demonstration
- **NFR-3:** Comprehensive error messages and logging
- **NFR-4:** Cross-platform compatibility (Windows, macOS, Linux)

### 5.2 Performance
- **NFR-5:** Training time < 30 minutes on modern hardware
- **NFR-6:** Inference time < 1 second for full test set
- **NFR-7:** Memory usage reasonable for consumer hardware

### 5.3 Maintainability
- **NFR-8:** Modular code structure with clear separation of concerns
- **NFR-9:** Comprehensive code comments and docstrings
- **NFR-10:** Type hints for function signatures
- **NFR-11:** Unit tests for critical components

### 5.4 Reproducibility
- **NFR-12:** Fixed random seeds for deterministic results
- **NFR-13:** Version pinning for all dependencies
- **NFR-14:** Complete environment specification (requirements.txt)
- **NFR-15:** Step-by-step reproduction instructions

---

## 6. User Stories

### 6.1 As a Student
- I want to run the complete pipeline with a single command so that I can quickly reproduce results
- I want clear visualizations so that I can understand how LSTM extracts frequencies
- I want to experiment with hyperparameters so that I can learn their effects

### 6.2 As a Researcher
- I want modular code so that I can extend the system to more frequencies
- I want detailed metrics so that I can compare different architectures
- I want to understand the implementation so that I can apply it to my own problems

### 6.3 As an Instructor
- I want comprehensive documentation so that I can use this as teaching material
- I want to see proper state management so that students learn LSTM fundamentals
- I want reproducible results so that I can verify correct implementation

---

## 7. Technical Constraints

### 7.1 Hardware
- Must run on CPU (GPU acceleration optional)
- Maximum RAM usage: 8 GB
- No specialized hardware required

### 7.2 Software
- Python 3.8+
- PyTorch framework
- Standard scientific Python libraries (NumPy, Matplotlib, etc.)

### 7.3 Data
- All data synthetically generated (no external data sources)
- Dataset size fixed at specifications

---

## 8. Deliverables Checklist

### 8.1 Code
- [ ] Data generation module (`src/data/`)
- [ ] LSTM models (L=1 and L>1) (`src/models/`)
- [ ] Training pipeline (`src/training/`)
- [ ] Evaluation and visualization (`src/evaluation/`)
- [ ] Main execution script
- [ ] Jupyter demonstration notebook

### 8.2 Documentation
- [ ] README.md (root directory)
- [ ] PRD.md (this document)
- [ ] Technical Specification
- [ ] Implementation Guide
- [ ] L>1 Justification Document
- [ ] prompts.md

### 8.3 Results
- [ ] Trained model weights
- [ ] All required graphs (saved as PNG/PDF)
- [ ] Performance metrics (MSE values)
- [ ] Screenshots of working system
- [ ] Comparison results (L=1 vs L>1)

### 8.4 Configuration
- [ ] requirements.txt
- [ ] Setup instructions
- [ ] Environment configuration guide

---

## 9. Timeline & Milestones

### Milestone 1: Documentation & Setup (Completed)
- Project structure
- Documentation files
- Requirements specification

### Milestone 2: Core Implementation
- Data generation
- Model implementations
- Training pipeline

### Milestone 3: Evaluation & Analysis
- Generate all metrics
- Create visualizations
- Comparative analysis

### Milestone 4: Final Deliverables
- Complete documentation
- Screenshots
- Final testing and validation

---

## 10. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| State management errors in L=1 | Medium | High | Comprehensive testing, clear documentation |
| Poor generalization | Medium | High | Proper train/test split, monitor overfitting |
| Training instability | Low | Medium | Gradient clipping, learning rate scheduling |
| Insufficient documentation | Low | Medium | Follow checklist, peer review |

---

## 11. Acceptance Criteria

The project is considered complete and successful when:

1. ✅ Both L=1 and L>1 models are implemented and functional
2. ✅ MSE < 0.1 on both training and test sets
3. ✅ MSE_test ≈ MSE_train (generalization achieved)
4. ✅ All required visualizations generated and saved
5. ✅ Complete documentation delivered
6. ✅ System runs successfully on fresh environment
7. ✅ Screenshots demonstrate working implementation
8. ✅ Code follows best practices (modular, commented, typed)

---

## 12. References

1. Assignment Document: L2-homework.pdf (Dr. Segal Yoram, November 2025)
2. Hochreiter & Schmidhuber (1997): "Long Short-Term Memory"
3. PyTorch LSTM Documentation
4. Signal Processing fundamentals

---

**Document Status:** ✅ Complete
**Next Steps:** Proceed with Technical Specification and Implementation

