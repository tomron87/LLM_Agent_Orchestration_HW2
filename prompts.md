# Prompts Log - LSTM Frequency Extraction Assignment

This document stores all prompts given during the development session for complete traceability and reproducibility.

---

## Prompt 1
**Timestamp:** 2025-11-07
**Type:** Initial Assignment Request

### Prompt Text:
```
You are an excellent deep learning specialist. Read this assignment instructions carefully - @L2-homework.pdf.
Follow the instructions and create a working project in Python. Make sure create a PRD markdown document,
a README markdown document, and any other markdown document you see necessary so anyone can clone the repo
and do this on their own computer, regardless of operating system. Make sure to make excellent quality
documentation for everything possible. Store all documentation in documentation folder, except the readme
which should be in the root folder. Make a document called prompts.py where you store every prompt I give
you during this session. This assignment should receive a perfect 100% score. Ask me any follow up questions
you may have after you read my instructions and @L2-homework.pdf.
```

### Response Summary:
Read assignment PDF and asked clarifying questions about:
- Deep learning framework preference
- Sequence length implementation approach (L=1 vs L>1)
- Code structure preference
- Additional analysis requirements

---

## Prompt 2
**Timestamp:** 2025-11-07
**Type:** User Selections Response

### User Selections:
- **Framework:** PyTorch
- **Sequence Length:** Both (L=1 and L>1 implementations with comparison)
- **Code Structure:** Both (modular scripts + demonstration notebook)
- **Additional Analysis:** Training curves, Frequency domain analysis, Hyperparameter study

### Response Summary:
User confirmed all technical choices. Proceeding with comprehensive implementation including both approaches.

---

## Prompt 3
**Timestamp:** 2025-11-07
**Type:** Additional Requirement

### Prompt Text:
```
I forgot - include screenshots of relevant demonstrations that everything works.
```

### Response Summary:
Added requirement to include screenshots demonstrating:
- Training progress and loss curves
- Model predictions vs ground truth
- All required visualizations and graphs
- Terminal outputs showing successful execution

---

## Prompt 4
**Timestamp:** 2025-11-07
**Type:** Clarification

### Prompt Text:
```
the prompts file should be a markdown document.
```

### Response Summary:
Changed prompts.py to prompts.md (this document).

---

## Assignment Key Requirements Summary

Based on the PDF (L2-homework.pdf), here are the critical requirements:

### Problem Statement
- Extract pure frequency components from a mixed noisy signal using LSTM
- 4 sinusoidal frequencies: f₁=1 Hz, f₂=3 Hz, f₃=5 Hz, f₄=7 Hz
- Sampling rate: 1000 Hz
- Time range: 0-10 seconds (10,000 samples total)

### Signal Generation
- **Noisy signal:** Each component has random amplitude Aᵢ(t) ∼ Uniform(0.8, 1.2) and phase φᵢ(t) ∼ Uniform(0, 2π) that vary per sample
- **Mixed signal:** S(t) = (1/4) × Σ Sinusᵢⁿᵒⁱˢʸ(t)
- **Ground truth:** Targetᵢ(t) = sin(2π·fᵢ·t) - pure sinusoid with no noise

### Dataset Structure
- **Training set:** 40,000 rows (10,000 samples × 4 frequencies), Seed #1
- **Test set:** 40,000 rows (10,000 samples × 4 frequencies), Seed #2
- Each row: `[S[t], C₁, C₂, C₃, C₄]` where C is one-hot frequency selector

### Model Requirements
- **Architecture:** LSTM network
- **Input:** S[t] (noisy signal) + C (one-hot vector)
- **Output:** Pure extracted sinusoid for selected frequency
- **Sequence Length (L):** Default L=1 (must preserve internal state between samples)
- **Critical:** Internal state (hₜ, cₜ) must NOT reset between consecutive samples

### Evaluation Metrics
1. MSE on training set (Seed #1)
2. MSE on test set (Seed #2)
3. Generalization check: MSE_test ≈ MSE_train

### Required Visualizations
1. **Graph 1:** Comparison for one specific frequency (f₂) showing:
   - Target (clean line)
   - LSTM Output (dots)
   - Noisy mixed signal S (background)

2. **Graph 2:** Four sub-graphs, one for each frequency, showing extraction results

### Implementation Options
- **Default:** L=1 with stateful LSTM (manually preserve state)
- **Alternative:** L>1 (e.g., L=10 or L=50) with sliding window - requires justification document

---

## Implementation Plan

### Phase 1: Documentation
- [x] Create prompts.md
- [ ] Create PRD (Product Requirements Document)
- [ ] Create Technical Specification
- [ ] Create Implementation Guide
- [ ] Create README.md

### Phase 2: Core Implementation
- [ ] Data generation module (train/test sets with different seeds)
- [ ] LSTM model with L=1 (stateful)
- [ ] LSTM model with L>1 (sliding window)
- [ ] Training pipeline
- [ ] Evaluation and visualization module

### Phase 3: Analysis & Documentation
- [ ] Generate all required graphs
- [ ] Training curves and convergence analysis
- [ ] FFT frequency domain analysis
- [ ] Hyperparameter study
- [ ] Create justification document for L>1 approach
- [ ] Capture screenshots of all demonstrations

### Phase 4: Final Deliverables
- [ ] Demonstration Jupyter notebook
- [ ] requirements.txt
- [ ] Setup and installation guide
- [ ] Screenshots documentation
- [ ] Final README with complete instructions

---

*This document will be updated with any additional prompts during the development session.*
