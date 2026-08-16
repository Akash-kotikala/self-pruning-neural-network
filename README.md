# Self-Pruning Neural Network

A PyTorch implementation of a neural network that learns to suppress unnecessary connections during training using **learnable gates** and **sparsity regularization**.

Unlike conventional pruning, where a trained model is pruned afterward, this approach integrates the pruning mechanism directly into the training process.

---

## Overview

Each weight in a custom `PrunableLinear` layer is associated with a learnable gate score.

The gate is obtained using a sigmoid transformation:

```python
gate = torch.sigmoid(gate_score)
```

The effective weight used by the network is:

```python
effective_weight = weight * gate
```

This produces a differentiable gating mechanism:

- Gate close to `1` → connection remains strongly active.
- Gate close to `0` → connection is strongly suppressed.

A sparsity penalty is added to the classification loss to encourage the model to reduce unnecessary gate values.

After training, a connection is counted as pruned when its gate value is below `0.01`.

---

## Objectives

This project aims to:

- Implement a custom `PrunableLinear` layer from scratch.
- Associate every weight with a learnable gate parameter.
- Train weights and gate parameters jointly.
- Add L1-based sparsity regularization.
- Measure the resulting connection sparsity.
- Compare different sparsity coefficients (`λ`).
- Analyze the accuracy-sparsity trade-off.
- Visualize the learned gate distribution.

---

## Method

### 1. Prunable Linear Layer

The custom layer contains three learnable components:

```text
Weight
Bias
Gate Scores
```

For every forward pass:

```text
Gate Scores
     │
     ▼
  Sigmoid
     │
     ▼
 Gate Values
     │
     ×
     │
Weights
     │
     ▼
Effective Weights
     │
     ▼
Linear Transformation
```

The effective weights are calculated as:

```python
pruned_weights = self.weight * torch.sigmoid(self.gate_scores)
```

Because the operation is differentiable, gradients can flow through both the weights and gate scores during backpropagation.

---

## 2. Sparsity Regularization

The training objective is:

```text
Total Loss = Classification Loss + λ × Sparsity Loss
```

Cross-entropy loss is used for classification.

The sparsity term is the sum of the gate values:

```text
Sparsity Loss = Σ gate
```

Since sigmoid produces values in `[0, 1]`, the gates are non-negative.

The parameter `λ` controls how strongly the model is encouraged to suppress connections.

```text
Small λ  → weaker sparsity pressure
Large λ  → stronger sparsity pressure
```

The sigmoid itself approaches zero continuously rather than producing exact zeros. Therefore, the implementation uses a threshold of `0.01` when reporting the final pruning level.

---

## Model Architecture

The implementation uses fully connected prunable layers for CIFAR-10:

```text
CIFAR-10 Image
      │
      ▼
   Flatten
      │
      ▼
PrunableLinear
  3072 → 512
      │
     ReLU
      │
      ▼
PrunableLinear
   512 → 256
      │
     ReLU
      │
      ▼
PrunableLinear
    256 → 10
      │
      ▼
 Class Prediction
```

CIFAR-10 images have dimensions:

```text
32 × 32 × 3
```

Therefore:

```text
32 × 32 × 3 = 3072
```

input features are used after flattening.

---

## Dataset

The model is trained and evaluated using the **CIFAR-10** dataset through `torchvision`.

| Property | Value |
|---|---:|
| Training images | 50,000 |
| Test images | 10,000 |
| Classes | 10 |
| Image size | 32 × 32 |
| Channels | 3 |

The dataset is downloaded automatically by `torchvision` when it is not already available locally.

The dataset is not included in this repository.

---

## Pruning Criterion

A connection is considered pruned when:

```text
gate < 0.01
```

The reported sparsity is:

```text
Sparsity (%) =
(Number of gates below 0.01 / Total number of gates) × 100
```

A higher sparsity percentage indicates that more connections have been suppressed.

---

## Experimental Configuration

The final experiment used:

```python
NUM_EPOCHS = 20

LAMBDA_VALUES = [
    1e-6,
    5e-6,
    1e-5
]
```

The implementation automatically uses CUDA when a compatible GPU is available; otherwise, it falls back to CPU.

---

## Results

The model was trained for 20 epochs for each value of `λ`.

| Lambda | Test Accuracy | Sparsity |
|---:|---:|---:|
| `1e-6` | 54.29% | 0.56% |
| `5e-6` | **55.71%** | 2.32% |
| `1e-5` | 55.18% | **6.02%** |

### Best Accuracy

```text
Lambda       : 5e-6
Test Accuracy: 55.71%
Sparsity     : 2.32%
```

### Highest Sparsity

```text
Lambda       : 1e-5
Test Accuracy: 55.18%
Sparsity     : 6.02%
```

---

## Results Analysis

The experiments show that increasing `λ` increases the pressure placed on the gate values.

Observed sparsity:

```text
λ = 1e-6  → 0.56%
λ = 5e-6  → 2.32%
λ = 1e-5  → 6.02%
```

The result demonstrates the expected sparsity-accuracy trade-off.

The highest test accuracy was obtained with `λ = 5e-6`.

Increasing `λ` from `5e-6` to `1e-5` increased sparsity from `2.32%` to `6.02%`, while test accuracy decreased from `55.71%` to `55.18%`.

Thus, `λ = 5e-6` provided the best accuracy among the tested configurations, while `λ = 1e-5` produced the highest sparsity.

---

## Gate Distribution

The project generates a histogram of the learned gate values for the selected model.

The distribution helps visualize:


- Gates close to zero that correspond to suppressed connections.
- Gates away from zero that correspond to active connections.

The generated visualization is stored as:

```text
results/gate_distribution.png
```

---

## Training Pipeline

```text
                 CIFAR-10
                    │
                    ▼
              Preprocessing
                    │
                    ▼
          Prunable Neural Network
                    │
          ┌─────────┴─────────┐
          │                   │
       Weights            Gate Scores
          │                   │
          │                 Sigmoid
          │                   │
          └─────────┬─────────┘
                    ▼
             Effective Weights
                    │
                    ▼
              Classification
                    │
          ┌─────────┴─────────┐
          │                   │
   Cross-Entropy Loss    Sparsity Loss
          │                   │
          └─────────┬─────────┘
                    ▼
                Total Loss
                    │
                    ▼
             Backpropagation
                    │
                    ▼
          Update Weights + Gates
                    │
                    ▼
                 Testing
               /         \
              /           \
         Accuracy        Sparsity
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Akash-kotikala/self-pruning-neural-network.git
```

### 2. Enter the project directory

```bash
cd self-pruning-neural-network
```




The CIFAR-10 dataset will be downloaded automatically if it is not already present.

---

## Requirements

- Python 3.10+
- PyTorch
- Torchvision
- Matplotlib
- CUDA-enabled GPU (optional)

The required Python packages are listed in:

```text
requirements.txt
```

---

## Project Structure

```text
self-pruning-neural-network/
│
├── submission.ipynb
├── README.md

```

### `submission.ipynb`

Contains the complete implementation:

- `PrunableLinear`
- Neural network architecture
- Sparsity regularization
- Training loop
- Evaluation loop
- Sparsity calculation
- Gate distribution visualization
- Lambda experiments



### `results/`

Stores generated experiment visualizations.

---

## Technologies

- **Python**
- **PyTorch**
- **Torchvision**
- **Matplotlib**
- **CUDA**

---

## Limitations

The current implementation uses fully connected layers rather than convolutional layers.

CIFAR-10 is an image dataset with important spatial structure, so a convolutional architecture could provide better classification performance.

The current results are primarily intended to demonstrate the self-pruning mechanism and the effect of learnable gates and sparsity regularization.

The final pruning level depends on factors such as:

- Sparsity coefficient `λ`
- Number of training epochs
- Gate initialization
- Optimizer
- Pruning threshold

---

## Future Improvements

Possible extensions include:

- Implement prunable convolutional layers.
- Explore structured neuron and channel pruning.
- Add a hard-pruning stage after training.
- Introduce adaptive `λ` scheduling.
- Compare different gate parameterizations.
- Measure actual parameter and memory reduction.
- Measure inference-time improvements.
- Compare against conventional post-training pruning.
- Evaluate on larger datasets.

---

## Conclusion

This project demonstrates a neural network that learns both its predictive weights and the importance of its individual connections during training.

The pruning mechanism is integrated into the model using learnable sigmoid gates and an L1-based sparsity penalty rather than applying pruning only after training.

The experiments show that increasing the sparsity coefficient results in a higher percentage of suppressed connections.

For the tested configurations:

- **Best test accuracy:** 55.71% at `λ = 5e-6`
- **Highest sparsity:** 6.02% at `λ = 1e-5`

The results demonstrate the expected trade-off between maintaining classification performance and increasing network sparsity.
