# ADR-001: PyTorch Framework Selection

**Date**: November 2025
**Status**: Accepted
**Deciders**: Igor Nazarenko, Tom Ron, Roie Gilad
**Technical Story**: Selection of deep learning framework for LSTM frequency extraction project

## Context

We need to select a deep learning framework for implementing LSTM models for frequency extraction from mixed signals. The project requires:
- Dynamic computational graphs for flexible LSTM architectures
- Strong support for custom neural network layers
- Efficient GPU acceleration (CUDA, MPS)
- Active community and comprehensive documentation
- Production-ready deployment capabilities

### Considered Frameworks
1. **PyTorch** - Dynamic graphs, Pythonic API, research-friendly
2. **TensorFlow/Keras** - Production-focused, static graphs (TF 1.x), more verbose
3. **JAX** - Functional programming, excellent performance, steeper learning curve
4. **MXNet** - Multi-language support, declining community

## Decision

We will use **PyTorch 2.0+** as the deep learning framework.

### Rationale
1. **Dynamic Computational Graphs**: PyTorch's define-by-run paradigm allows for flexible LSTM state management, critical for implementing both stateful (L=1) and sequence-based (L>1) architectures
2. **Research-Friendly**: Intuitive, Pythonic API accelerates prototyping and experimentation
3. **Hardware Support**: Native support for CUDA (NVIDIA), MPS (Apple Silicon), and CPU backends
4. **Ecosystem Maturity**: Extensive library ecosystem (torchvision, torchaudio) and tooling (TorchScript, ONNX export)
5. **Active Community**: Large community, frequent updates, extensive documentation and tutorials
6. **Academic Standard**: De facto standard in ML research (>70% of papers at major conferences)

## Consequences

### Positive
- **Rapid Development**: Intuitive API reduces development time for custom LSTM variants
- **Debugging**: Dynamic graphs enable standard Python debugging tools (pdb, IDE debuggers)
- **Flexibility**: Easy to implement custom training loops, loss functions, and data loaders
- **Deployment Options**: Multiple deployment paths (TorchScript, ONNX, TorchServe)
- **Apple Silicon Support**: Native MPS backend for efficient training on M1/M2/M3 Macs

### Negative
- **Memory Overhead**: Dynamic graphs have slightly higher memory overhead than static graphs
- **Production Complexity**: Requires additional tooling (TorchScript, ONNX) for production deployment compared to TensorFlow Serving
- **Breaking Changes**: PyTorch 2.0+ introduced some API changes, requiring version management

### Risks and Mitigation
- **Risk**: Version incompatibility across team members
  - **Mitigation**: Pin PyTorch version in requirements.txt (torch>=2.0.0,<2.5.0)
- **Risk**: Device compatibility issues (CUDA vs MPS vs CPU)
  - **Mitigation**: Implement automatic device detection with fallback (see config.yaml:device)

## Alternatives Considered

### TensorFlow/Keras
- **Pros**: Better production tooling (TensorFlow Serving), wider industry adoption
- **Cons**: More verbose API, less intuitive for custom architectures, static graph complications
- **Rejected**: Development complexity outweighs production benefits for research project

### JAX
- **Pros**: Excellent performance, functional programming paradigm, automatic differentiation
- **Cons**: Steeper learning curve, smaller ecosystem, less suitable for prototyping
- **Rejected**: Learning curve incompatible with project timeline

### MXNet
- **Pros**: Multi-language support, efficient parameter server
- **Cons**: Declining community, uncertain future after AWS focus shift
- **Rejected**: Community decline poses long-term risk

## References

- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- PyTorch 2.0 Release Notes: https://pytorch.org/blog/pytorch-2.0-release/
- "PyTorch: An Imperative Style, High-Performance Deep Learning Library" (NeurIPS 2019)

## Notes

- This decision was made in alignment with course requirements for demonstrating LSTM fundamentals
- PyTorch's dynamic graph model is particularly well-suited for implementing stateful LSTMs with explicit state management
- The choice enables easy experimentation with different sequence lengths (L=1, L=50, L=100) without framework limitations
