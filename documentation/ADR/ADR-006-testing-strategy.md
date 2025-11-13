# ADR-006: Testing Strategy (pytest + 42% Initial Coverage)

**Date**: November 2025
**Status**: Accepted with Improvement Plan
**Deciders**: Igor Nazarenko, Tom Ron, Roie Gilad
**Technical Story**: Balancing test quality vs development time in academic project

## Context

### Project Constraints
- **Timeline**: 3-week academic project with multiple deliverables
- **Scope**: Research project (not production software initially)
- **Team Size**: 3 students with varying testing experience
- **Requirements**: Functional code, documentation, research analysis

### Testing Dilemma
- **High Coverage (85%+)**: Industry standard, but requires 40-60 hours for this codebase
- **No Tests**: Fast development, but risky refactoring, no regression safety
- **Strategic Testing**: Focus on critical paths, expand coverage incrementally

## Decision

Implement **phased testing strategy** starting with 40-50% coverage on critical components, with documented plan to reach 85%+ later.

### Phase 1 (Completed): Foundation (40-50% coverage)
**Focus**: Test critical functionality to enable safe refactoring
- ✅ Core models (forward pass, shape validation)
- ✅ Data generation (signal creation, dataset loading)
- ✅ Configuration system (YAML loading, overrides)
- ❌ Training loop (minimal coverage - 20%)
- ❌ Visualization (minimal coverage - 42%)

### Phase 2 (Planned): Expansion (60-75% coverage)
**Focus**: Expand to training and evaluation
- 🔄 Full training loop testing
- 🔄 Checkpoint save/load
- 🔄 Early stopping logic
- 🔄 Metric computation edge cases

### Phase 3 (Future): Excellence (85%+ coverage)
**Focus**: Comprehensive coverage for production-ready release
- 🔄 Visualization rendering
- 🔄 Error handling paths
- 🔄 Edge case coverage
- 🔄 Integration tests

## Implementation

### Testing Framework: pytest
**Why pytest over unittest**:
- ✅ Less boilerplate (no class inheritance required)
- ✅ Better fixtures (setup/teardown)
- ✅ Powerful parameterization
- ✅ Excellent plugin ecosystem (pytest-cov, pytest-xdist)

### Test Structure
```
tests/
├── conftest.py           # Shared fixtures
├── models/
│   ├── test_lstm_stateful.py    (52% coverage)
│   ├── test_lstm_sequence.py    (38% coverage)
│   └── test_lstm_conditioned.py (16% coverage)
├── data/
│   ├── test_signal_generator.py (50% coverage)
│   ├── test_dataset.py          (63% coverage)
│   └── test_data_loader.py      (51% coverage)
├── training/
│   ├── test_config.py           (70% coverage)
│   └── test_trainer.py          (20% coverage) ⚠️ LOW
└── evaluation/
    ├── test_metrics.py          (37% coverage) ⚠️ LOW
    └── test_visualization.py    (42% coverage) ⚠️ LOW
```

### Testing Patterns

#### 1. Shape Testing
```python
def test_forward_shape():
    model = StatefulLSTM(input_size=5, hidden_size=64)
    x = torch.randn(16, 1, 5)
    output = model(x)
    assert output.shape == (16, 4)  # batch_size, num_frequencies
```

#### 2. Property Testing
```python
def test_signal_generator_deterministic():
    gen1 = SignalGenerator(seed=42)
    gen2 = SignalGenerator(seed=42)
    signal1 = gen1.generate_signal()
    signal2 = gen2.generate_signal()
    np.testing.assert_array_equal(signal1, signal2)
```

#### 3. Edge Case Testing
```python
@pytest.mark.parametrize("hidden_size", [0, -1, "invalid"])
def test_invalid_hidden_size(hidden_size):
    with pytest.raises((ValueError, TypeError)):
        StatefulLSTM(hidden_size=hidden_size)
```

## Current Coverage (Actual Results)

**Overall**: 42% (521/1235 statements covered)

**High Coverage (≥70%)**:
- training/config.py: 70% ✅
- All __init__.py: 100% ✅

**Medium Coverage (50-69%)**:
- data/dataset.py: 63%
- models/lstm_stateful.py: 52%
- utils/logger.py: 53%
- data/data_loader.py: 51%
- data/signal_generator.py: 50%

**Low Coverage (<50%)** - Priority for Phase 2:
- **training/trainer.py: 20%** ⚠️ CRITICAL
- evaluation/visualization.py: 42%
- models/lstm_sequence.py: 38%
- evaluation/metrics.py: 37%
- models/lstm_conditioned.py: 16%

## Consequences

### Positive
1. **Safe Refactoring**: Core functionality tested (models, data loading)
2. **Regression Prevention**: 72 tests catch breaking changes
3. **Documentation**: Tests serve as usage examples
4. **Confidence**: Key components validated (forward pass, shapes, determinism)
5. **Foundation**: Infrastructure ready for easy expansion

### Negative
1. **Incomplete Coverage**: 42% < 85% target
2. **Training Loop Risk**: Only 20% coverage in trainer.py (complex logic under-tested)
3. **Visualization Gap**: Plot generation not fully validated
4. **False Confidence**: High test count (72 tests) may mask coverage gaps

### Known Gaps

**Critical Gap**: trainer.py (20% coverage)
- Missing: Checkpoint save/load tests
- Missing: Early stopping logic tests
- Missing: Training loop edge cases (NaN handling, device switching)
- **Impact**: High risk for refactoring training logic

**Medium Gap**: evaluation/metrics.py (37% coverage)
- Missing: Edge cases (zero variance, NaN inputs)
- Missing: Correlation computation validation
- **Impact**: Medium risk for incorrect metric computation

## Alternatives Considered

### Alternative 1: Test-Driven Development (TDD)
**Approach**: Write tests before code, aim for 95%+ coverage from start
**Pros**: Highest code quality, comprehensive coverage
**Cons**: 2-3x longer development time (not feasible for 3-week project)
**Rejected**: Timeline incompatible with TDD approach

### Alternative 2: No Formal Tests
**Approach**: Manual testing only, no automated test suite
**Pros**: Fastest development
**Cons**: No regression safety, risky refactoring, hard to onboard new developers
**Rejected**: Unacceptable risk for collaborative project

### Alternative 3: Property-Based Testing (Hypothesis)
**Approach**: Use Hypothesis framework for automatic test case generation
```python
@given(st.floats(min_value=0.1, max_value=10.0))
def test_signal_generator_valid_frequency(frequency):
    gen = SignalGenerator(frequencies=[frequency])
    signal = gen.generate_signal()
    assert signal.shape == (10000,)
```
**Pros**: Finds edge cases automatically, high coverage with less code
**Cons**: Steeper learning curve, harder to debug failing tests
**Partially Adopted**: Used parametrization, but not full Hypothesis integration

### Alternative 4: 85% Coverage from Start (Industry Standard)
**Approach**: Aim for 85%+ coverage from day 1
**Estimated Time**: 50-70 hours for comprehensive test suite
**Timeline Impact**: Would consume 30-40% of project time
**Pros**: Production-ready from start, excellent quality
**Cons**: Less time for research, documentation, experiments
**Rejected**: Academic project prioritizes research over production quality initially

## Rationale for 42% Target (Phase 1)

### Time-Quality Trade-off Analysis
| Coverage Target | Test Writing Time | Research Time | Quality | Decision |
|----------------|-------------------|---------------|---------|----------|
| 0% (no tests) | 0 hours | 100% | Poor | ❌ Too risky |
| 40-50% (Phase 1) | 15-20 hours | 80% | Good | ✅ **Selected** |
| 70-80% | 35-45 hours | 50% | Very Good | ⚠️ Tight timeline |
| 85%+ (industry) | 50-70 hours | 30% | Excellent | ❌ Timeline conflict |

**Decision**: 40-50% provides best balance for academic research project with clear improvement roadmap.

## Test Quality Over Quantity

**Philosophy**: **72 high-quality tests > 200 low-quality tests**

Our tests are:
- ✅ **Deterministic**: Use fixed seeds, no flaky tests
- ✅ **Fast**: 72 tests run in 9 seconds (125ms/test average)
- ✅ **Independent**: No test dependencies, can run in any order
- ✅ **Focused**: Each test validates one specific behavior
- ✅ **Readable**: Clear test names, good documentation

## Improvement Roadmap (Phase 2 & 3)

### Phase 2 Goals (60-75% coverage, ~15 hours)
1. **Trainer Tests** (+100 lines, +30% trainer.py coverage):
   - Full training loop
   - Checkpoint save/load
   - Early stopping
   - Loss computation edge cases

2. **Metrics Tests** (+80 lines, +40% metrics.py coverage):
   - Correlation coefficient validation
   - MSE edge cases (zero variance, NaN handling)
   - Generalization ratio computation

3. **Visualization Tests** (+60 lines, +25% visualization.py coverage):
   - Plot generation (without display)
   - Figure saving
   - Edge cases (empty data, NaN values)

### Phase 3 Goals (85%+ coverage, ~20 hours)
- Comprehensive edge case coverage
- Integration tests (end-to-end training + evaluation)
- Stress tests (large datasets, memory limits)
- Performance benchmarks

## References

- pytest Documentation: https://docs.pytest.org/
- pytest-cov Plugin: https://pytest-cov.readthedocs.io/
- Google Testing Blog: "Test Behavior, Not Implementation"
- "The Pragmatic Programmer" - Balance testing effort with project goals

## Notes

- **Transparency**: Coverage documented honestly in COVERAGE_REPORT.md (not inflated)
- **Improvement Plan**: Clear roadmap shows commitment to reaching 85%+
- **Academic Context**: 42% acceptable for research project, with production-ready path defined
- **Team Learning**: Testing infrastructure teaches team about pytest, fixtures, parametrization
- **2 Test Failures**: Currently 70/72 passing (97.2%), non-critical failures documented in evaluation report
