# Flux Roadmap

## Current Status

Flux currently has the core pieces of a small automatic differentiation / machine-learning framework:

- Tensor storage and basic tensor operations
- Computation graph / operation nodes
- Reverse-mode automatic differentiation
- Gradient accumulation and zeroing
- Broadcasting for the current tensor representation
- L2 loss
- SGD optimiser
- Linear regression as an end-to-end example

The next phase is focused on finishing and hardening the tensor/autograd foundations before expanding into higher-level ML functionality.

---

## 1. Encapsulate `TensorState`

**Status: In progress**

Remove direct access to `TensorState` internals and expose the required functionality through getters/setters.

Goals:

- Keep `TensorState` implementation details private.
- Provide a clean public Tensor API.
- Ensure autograd internals do not unnecessarily depend on storage details.
- Make future changes to the Tensor representation easier.

---

## 2. Complete Autograd Coverage

**Status: Next**

Ensure every existing differentiable Tensor operation participates correctly in the autograd graph.

Tasks:

- Audit all existing operations.
- Add an autograd node to every differentiable operation.
- Implement missing backward operations.
- Ensure scalar operations do not accidentally terminate gradient chains.
- Ensure `requires_grad` is propagated correctly.
- Test graph construction for every operation.
- Test gradient accumulation and repeated backward passes.

The goal is:

> Every differentiable operation currently exposed by Flux has a correct forward implementation and corresponding backward implementation.

---

## 3. Generalise Tensors to Arbitrary Rank

**Status: Planned**

Remove the current assumptions around one-dimensional tensors.

Goals:

- Support tensors of any rank.
- Keep flat contiguous storage internally.
- Make indexing independent of tensor rank.
- Make shape handling consistent throughout the Tensor API.
- Ensure operations work correctly regardless of rank.

Examples:

```text
[4]
[2, 3]
[2, 3, 4]
[2, 3, 4, 5]
```

---

## 4. General N-Dimensional Broadcasting

**Status: Planned**

Replace the current limited broadcasting behaviour with proper shape-based broadcasting.

Goals:

- Broadcast dimensions according to standard tensor broadcasting rules.
- Support broadcasting across arbitrary ranks.
- Validate incompatible shapes.
- Produce correct output shapes.
- Reuse broadcasting logic throughout Tensor operations.

Examples:

```text
[3]       + [3]       → [3]

[3, 1]    + [1, 4]    → [3, 4]

[2, 3, 1] + [1, 4]   → [2, 3, 4]
```

### Backward Broadcasting

Backward propagation must reverse broadcasting correctly.

For example:

```text
[3, 1] + [1, 4] → [3, 4]
```

The gradients must be reduced back to:

```text
dA → [3, 1]
dB → [1, 4]
```

This is a fundamental requirement before considering the tensor system complete.

---

## 5. Documentation

**Status: Planned**

Document the public API and the architecture once the tensor/autograd foundations have stabilised.

Documentation should cover:

### User-facing API

- Tensor creation
- Tensor shapes
- Tensor operations
- Broadcasting
- Reductions
- Gradients
- `backward()`
- `zero_grad()`
- `requires_grad`
- Optimisers
- SGD

### Autograd architecture

Explain:

```text
Tensor
  ↓
OperationNode
  ↓
Parents
  ↓
Topological traversal
  ↓
Reverse-mode backward
  ↓
Gradient accumulation
```

Document how individual operations implement their derivatives and how broadcasting is handled during backward propagation.

### Examples

Provide progressively more useful examples:

1. Basic Tensor arithmetic
2. Broadcasting
3. Computing gradients
4. L2 loss
5. Linear regression
6. SGD training

The documentation should be developed alongside the API rather than left until the end of the project.

---

## 6. Expand Mathematical Operations

**Status: Planned**

Add useful mathematical operations needed for machine learning.

Potential operations:

- `exp`
- `log`
- `sqrt`
- `pow`
- `abs`
- `sum`
- `min`
- `max`

Then add common activation functions:

- ReLU
- Sigmoid
- Tanh

Every differentiable operation should include its corresponding autograd implementation and tests.

---

## 7. Matrix Operations

**Status: Planned**

Introduce operations required for practical neural networks.

Initial targets:

- Matrix multiplication (`matmul`)
- Transpose
- Matrix/vector multiplication where useful

For:

```text
C = A @ B
```

the backward pass should correctly calculate:

```text
dA = dC @ Bᵀ
dB = Aᵀ @ dC
```

This is a major milestone for Flux because it enables proper dense neural-network layers.

---

## 8. Neural-Network Primitives

**Status: Planned**

Build higher-level ML abstractions on top of the completed Tensor/autograd system.

Potential components:

- `Parameter`
- `Module`
- `Sequential`
- `Linear`
- Activation modules
- Model parameter collection

Target usage:

```rust
let model = Sequential::new([
    Linear::new(4, 16),
    ReLU::new(),
    Linear::new(16, 1),
]);
```

The goal is to move from manually manipulating tensors toward building actual trainable models.

---

## 9. GPU Backend

**Status: Future**

Only begin GPU work once the CPU Tensor/autograd semantics are stable.

Goals:

- Establish a device/backend abstraction.
- Separate Tensor semantics from execution backend.
- Investigate GPU tensor storage.
- Implement GPU kernels for core operations.
- Preserve the existing autograd interface.

The CPU implementation should remain useful independently of the GPU backend.

---

# Milestones

## Foundation

- [x] Tensor fundamentals
- [x] Basic arithmetic
- [x] Computation graph
- [x] Reverse-mode autograd
- [x] Gradient accumulation
- [x] `zero_grad`
- [x] SGD
- [ ] TensorState encapsulation
- [ ] Complete autograd coverage
- [ ] Arbitrary-rank tensors
- [ ] N-dimensional broadcasting
- [ ] Backward broadcasting

## Stability

- [ ] Comprehensive autograd tests
- [ ] Numerical gradient checking
- [ ] API cleanup
- [ ] Documentation

## ML Functionality

- [ ] Additional mathematical operations
- [ ] Activation functions
- [ ] Matrix multiplication
- [ ] Transpose
- [ ] Neural-network primitives
- [ ] Linear layers
- [ ] Model composition

## Future

- [ ] GPU backend
- [ ] GPU kernels
- [ ] Device abstraction

---

# Guiding Principle

Flux should be built from the bottom up:

```text
Tensor
  ↓
Tensor operations
  ↓
Broadcasting
  ↓
Autograd
  ↓
Optimisation
  ↓
Matrix operations
  ↓
Neural-network primitives
  ↓
Models
  ↓
GPU backend
```

The priority is not to accumulate features as quickly as possible. Each layer should be correct and well-tested before the next layer depends on it.
