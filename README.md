# Self-Pruning Neural Network

A PyTorch implementation of a self-pruning neural network that learns to suppress unnecessary connections during training using learnable gates and sparsity regularization.

The model is trained and evaluated on the CIFAR-10 dataset.

## Approach

Each weight in the custom `PrunableLinear` layer has a corresponding learnable gate.

The effective weight is calculated as:

```text
Effective Weight = Weight × Sigmoid(Gate Score)
```

The training objective is:

```text
Total Loss = Classification Loss + λ × Sparsity Loss
```

The sparsity loss is based on the sum of the gate values.

After training, connections with gate values below `0.01` are considered pruned.

## Model Architecture

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

## Dataset

| Property | Value |
|---|---:|
| Training Images | 50,000 |
| Test Images | 10,000 |
| Classes | 10 |
| Image Size | 32 × 32 × 3 |

## Training

The model was trained for **20 epochs** using:

```python
LAMBDA_VALUES = [
    1e-6,
    5e-6,
    1e-5
]
```

## Results

| Lambda | Test Accuracy | Sparsity |
|---:|---:|---:|
| `1e-6` | 54.29% | 0.56% |
| `5e-6` | **55.71%** | 2.32% |
| `1e-5` | 55.18% | **6.02%** |

The best test accuracy was **55.71%** with `λ = 5e-6`.

The highest sparsity was **6.02%** with `λ = 1e-5`.

## Gate Distribution

The following image shows the learned gate value distribution after training.

<p align="center">
  <img src="result.png" alt="Gate Distribution" width="700">
</p>

## Files

```text
self-pruning-neural-network/
│
├── submission.ipynb
├── result.png
└── README.md
```

- `submission.ipynb` - Complete implementation, training, evaluation, and visualization.
- `result.png` - Gate distribution generated from the trained model.
- `README.md` - Project documentation.

## Technologies

- Python
- PyTorch
- Torchvision
- Matplotlib
- CUDA
