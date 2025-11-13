# Architecture Decision Records (ADR)

This directory contains Architecture Decision Records (ADRs) documenting key design decisions made during the LSTM Frequency Extraction project development.

## What are ADRs?

Architecture Decision Records are lightweight documents that capture important architectural decisions along with their context and consequences. Each ADR describes:
- **Context**: The situation and problem requiring a decision
- **Decision**: The chosen solution
- **Consequences**: The trade-offs and implications
- **Alternatives**: Other options considered and why they were rejected

## ADR Index

### [ADR-001: PyTorch Framework Selection](./ADR-001-pytorch-framework-selection.md)
**Status**: Accepted | **Date**: November 2025

**Decision**: Use PyTorch 2.0+ as the deep learning framework

**Why**: Dynamic computational graphs, Pythonic API, excellent hardware support (CUDA, MPS, CPU), active community, and de facto standard in ML research. Enables flexible LSTM state management critical for both L=1 and L>1 architectures.

**Key Trade-offs**:
- ✅ Rapid development, intuitive debugging, strong ecosystem
- ⚠️ Slightly higher memory overhead vs static graphs
- ⚠️ Requires TorchScript/ONNX for production deployment

**Alternatives Rejected**: TensorFlow/Keras (more verbose), JAX (steeper learning curve), MXNet (declining community)

---

### [ADR-002: Stateful LSTM (L=1) Architecture](./ADR-002-stateful-lstm-architecture.md)
**Status**: Accepted | **Date**: November 2025

**Decision**: Implement custom StatefulLSTM class with explicit hidden state management for single-sample processing (L=1)

**Why**: Assignment requires demonstrating LSTM fundamentals with manual state tracking. Enables sequential processing one time step at a time with persistent (h_t, c_t) state between calls.

**Key Trade-offs**:
- ✅ Educational value, demonstrates LSTM mechanics, enables streaming inference
- ⚠️ 3.3x slower than sequence-based processing (L=50)
- ⚠️ More complex state management (risk of bugs, memory overhead)

**Alternatives Rejected**: Standard LSTM with seq_len=1 (insufficient for requirements), manual LSTM cell implementation (excessive complexity), LSTMCell (harder to scale to multiple layers)

---

### [ADR-003: Dual Architecture Strategy (L=1 and L>1)](./ADR-003-dual-architecture-strategy.md)
**Status**: Accepted | **Date**: November 2025

**Decision**: Implement both StatefulLSTM (L=1) and SequenceLSTM (L>1) architectures for comprehensive comparison

**Why**: Assignment requires comparing single-sample vs sequence processing. Empirical results show L=50 optimal (MSE: 0.199, Gen Ratio: 1.04), while L=100 severely overfits (Gen Ratio: 3.43).

**Key Trade-offs**:
- ✅ Comprehensive analysis, flexibility, educational value, clear production recommendation (L=50)
- ⚠️ 2x development/maintenance overhead, potential user confusion

**Alternatives Rejected**: Single architecture only (doesn't meet requirements), unified architecture with mode parameter (violates SRP), three+ separate classes (excessive duplication)

**Winner**: SequenceLSTM with L=50 (best balance of accuracy, speed, generalization)

---

### [ADR-004: Phase Scaling Factor (0.01)](./ADR-004-phase-scaling-factor.md)
**Status**: Accepted | **Date**: November 2025
**⚠️ MOST CRITICAL DECISION IN PROJECT**

**Decision**: Scale random signal phases by 0.01, making ϕ ∈ [0, 0.0628] instead of [0, 6.28]

**Why**: Full phase randomization (ϕ ∈ [0, 2π]) made task **impossible to learn** (MSE: 0.502, correlation: 0.018). Phase scaling transforms task from impossible → learnable with **2.5x better MSE, 51x better correlation**.

**Key Trade-offs**:
- ✅ Task becomes learnable (MSE: 0.199, Gen Ratio: 1.04)
- ✅ Maintains dataset diversity and generalization capability
- ⚠️ Assumes real-world signals have constrained phase variation
- ⚠️ Risk of perceived as "cheating" (mitigated by transparency)

**Impact**: **Before**: MSE 0.502, no learning | **After**: MSE 0.199, excellent generalization

**Alternatives Rejected**: Fixed phases (too easy, memorization), amplitude scaling only (doesn't solve problem), phase conditioning with FiLM (defeats purpose), discrete phase groups (not continuous)

**Validation**: Sensitivity analysis shows optimal range: phase_scale ∈ [0.005, 0.02]

---

### [ADR-005: Configuration Management (YAML + Environment Variables)](./ADR-005-configuration-management.md)
**Status**: Accepted | **Date**: November 2025

**Decision**: Implement hierarchical configuration: YAML (defaults) → Environment Variables (overrides) → Command-line (runtime)

**Why**: Production-ready configuration enables reproducibility, security (secrets in .env), flexibility (easy experiments), and team collaboration (consistent defaults).

**Key Trade-offs**:
- ✅ Reproducible research, no secrets in git, flexible experimentation, single source of truth
- ⚠️ Three configuration layers = cognitive load, debugging complexity

**Alternatives Rejected**: Python config files (less safe), JSON (no comments), TOML (less common), database-backed (overkill)

**Files**:
- `config.yaml` (92 lines) - defaults, committed to git
- `.env` - secrets, NOT committed
- `.env.example` (22 lines) - template, committed

---

### [ADR-006: Testing Strategy (pytest + 42% Initial Coverage)](./ADR-006-testing-strategy.md)
**Status**: Accepted with Improvement Plan | **Date**: November 2025

**Decision**: Phased testing strategy starting at 40-50% coverage, with documented roadmap to 85%+

**Why**: Balance test quality vs development time in 3-week academic project. Strategic testing on critical paths (models, data loading) enables safe refactoring, with clear expansion plan.

**Key Trade-offs**:
- ✅ Safe refactoring, 72 high-quality tests (9 sec runtime), solid foundation
- ⚠️ 42% < 85% target, trainer.py only 20% coverage (critical gap)

**Current Coverage**:
- High (≥70%): config.py, all __init__.py
- Medium (50-69%): dataset, data_loader, stateful LSTM
- **Low (<50%)**: trainer.py (20%), metrics (37%), visualization (42%)

**Alternatives Rejected**: TDD from start (2-3x longer), no tests (too risky), 85% from day 1 (50-70 hours, timeline conflict)

**Roadmap**: Phase 2 (60-75%, +15 hrs) → Phase 3 (85%+, +20 hrs)

---

### [ADR-007: Structured Logging Framework](./ADR-007-structured-logging-framework.md)
**Status**: Accepted | **Date**: November 2025

**Decision**: Implement custom structured logging with Python's logging module, featuring color-coded console output and persistent file logs

**Why**: Production-ready logging enables better debugging, performance profiling, and post-mortem analysis. Replaces print() statements with structured, filterable, persistent logs.

**Key Trade-offs**:
- ✅ Better debugging (timestamps, module context), production monitoring, color-coded readability
- ⚠️ Slight overhead (~0.3ms/call), log file management needed

**Features**:
- Color-coded severity levels (DEBUG=cyan, INFO=green, ERROR=red)
- Module-level loggers (context: which file generated log)
- Dual output (console + file)
- Configurable levels (DEBUG/INFO/WARNING/ERROR/CRITICAL)
- Daily log rotation

**Alternatives Rejected**: Continue with print() (unprofessional), third-party loggers (loguru, overkill), cloud logging (not needed for research)

**Performance**: <0.1% impact on training (negligible)

---

## How to Use These ADRs

### For New Developers
1. Read ADR-004 (Phase Scaling) first - explains the critical breakthrough
2. Read ADR-003 (Dual Architecture) - understand L=1 vs L>1 comparison
3. Skim other ADRs to understand design rationale

### When Making Changes
- **Modifying core architecture?** Review relevant ADR, update if decision changes
- **Adding new features?** Consider creating new ADR for significant decisions
- **Questioning design?** ADRs explain "why", not just "what"

### When Creating New ADRs
Use this template:
```markdown
# ADR-XXX: Title

**Date**: YYYY-MM
**Status**: Proposed | Accepted | Deprecated | Superseded
**Deciders**: Names

## Context
[Problem description, requirements, constraints]

## Decision
[Chosen solution]

## Consequences
[Trade-offs, positive/negative impacts]

## Alternatives Considered
[Other options and why rejected]

## References
[Links, papers, documentation]
```

## ADR Lifecycle

- **Proposed**: Under discussion, not yet implemented
- **Accepted**: Decision made and implemented
- **Deprecated**: No longer recommended, but still in use
- **Superseded**: Replaced by newer ADR (link to replacement)

## Links to Project Documentation

- **Architecture Overview**: [ARCHITECTURE_DIAGRAMS.md](../ARCHITECTURE_DIAGRAMS.md)
- **Implementation Guide**: [EXTENSIBILITY.md](../EXTENSIBILITY.md)
- **Development History**: [DEVELOPMENT_JOURNEY.md](../DEVELOPMENT_JOURNEY.md)
- **Research Findings**: [README.md](../../README.md#research-findings)

## References

- [Documenting Architecture Decisions by Michael Nygard](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
- [ADR GitHub Organization](https://adr.github.io/)
- [When to Use Architecture Decision Records](https://engineering.atspotify.com/2020/04/when-should-i-write-an-architecture-decision-record/)

---

**Last Updated**: November 13, 2025
**Total ADRs**: 7
**Project**: LSTM Frequency Extraction
**Team**: Igor Nazarenko, Tom Ron, Roie Gilad
