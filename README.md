# Self-Pruning Neural Network

A PyTorch implementation of a neural network that learns to prune unnecessary connections during training using learnable gates and sparsity regularization.

## Overview

Neural network pruning is commonly performed after a model has been trained. In this project, pruning is incorporated directly into the training process.

Each weight in the network is associated with a learnable gate parameter. The gate determines whether the corresponding connection should remain active or be suppressed.

The effective weight is calculated as:

```text
Effective Weight = Weight × Sigmoid(Gate Score)
A sparsity regularization term encourages gate values to move toward zero. After training, connections with gate values below 0.01 are considered pruned.

The model is trained and evaluated on the CIFAR-10 image classification dataset.

Objectives

The main objectives of this project are:

Implement a custom PrunableLinear layer from scratch.
Associate every weight with a learnable gate.
Train network weights and gate parameters jointly.
Encourage sparse connections using L1 regularization.
Measure the resulting network sparsity.
Compare different values of the sparsity coefficient λ.
Analyze the trade-off between classification accuracy and sparsity.
Visualize the distribution of learned gate values.
Approach

The model extends a standard feed-forward neural network by introducing a learnable gate for every weight.

Gate Mechanism

Each PrunableLinear layer contains:

Weight
Bias
Gate Scores

The gate values are obtained using the sigmoid function:

gates = torch.sigmoid(gate_scores)

This constrains the gate values to the range:

0 ≤ gate ≤ 1

The effective weights used during the forward pass are:

pruned_weights = weight * gates

Therefore:

Gate close to 1 → connection remains active.
Gate close to 0 → connection is strongly suppressed.
Sparsity Regularization

The training objective combines classification loss with a sparsity penalty:

Total Loss = Classification Loss + λ × Sparsity Loss

Cross-entropy loss is used for classification.

The sparsity loss is the sum of all gate values:

Sparsity Loss = Σ |gate|

Since sigmoid gates are always non-negative, this is equivalent to:

Sparsity Loss = Σ gate

The coefficient λ controls the strength of the sparsity constraint.

A larger λ applies greater pressure to suppress unnecessary connections.

Model Architecture

The current implementation uses fully connected prunable layers:

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

CIFAR-10 images have dimensions:

32 × 32 × 3

Therefore, each image is flattened into:

3072 input features
Dataset

The project uses the CIFAR-10 dataset through torchvision.

Property	Value
Training images	50,000
Test images	10,000
Classes	10
Image size	32 × 32
Channels	3

The dataset is automatically downloaded if it is not already available locally.

The dataset itself is not included in this repository.

Pruning Criterion

After training, a connection is considered pruned when:

Gate Value < 0.01

The sparsity level is calculated as:

Sparsity (%) =
(Number of gates < 0.01 / Total number of gates) × 100

A higher sparsity percentage indicates that more network connections have been suppressed.

Training Configuration

The final experiment was performed for 20 epochs using three sparsity coefficients:

NUM_EPOCHS = 20


LAMBDA_VALUES = [
    1e-6,
    5e-6,
    1e-5
]

The implementation automatically selects CUDA when a compatible GPU is available. Otherwise, it uses the CPU.

Experimental Results

The model was trained for 20 epochs for each value of λ.

Lambda	Test Accuracy	Sparsity
1e-6	54.29%	0.56%
5e-6	55.71%	2.32%
1e-5	55.18%	6.02%
Best Test Accuracy

The highest test accuracy was achieved with:

Lambda      : 5e-6
Test Accuracy: 55.71%
Sparsity     : 2.32%
Highest Sparsity

The highest sparsity was achieved with:

Lambda      : 1e-5
Test Accuracy: 55.18%
Sparsity     : 6.02%
Results Analysis

The experiments show that increasing the sparsity coefficient increases the number of suppressed connections:

λ = 1e-6  →  0.56% sparsity


λ = 5e-6  →  2.32% sparsity


λ = 1e-5  →  6.02% sparsity

This demonstrates that the learned gates respond to the sparsity regularization.

The best classification accuracy was obtained with λ = 5e-6.

Increasing the coefficient to 1e-5 increased sparsity from 2.32% to 6.02%, while test accuracy decreased only slightly from 55.71% to 55.18%.

This demonstrates the expected trade-off between model accuracy and network sparsity.

Gate Distribution

The project generates a histogram of the learned gate values for the selected model.

The distribution is useful for analyzing whether the network has learned to separate:

Connections that are strongly suppressed.
Connections that remain active.

A successful pruning process should result in an increasing concentration of gate values near zero.

The generated plot can be stored in:

results/gate_distribution.png
Training Pipeline

The complete training process can be summarized as:

             CIFAR-10
                │
                ▼
           Preprocessing
                │
                ▼
       Prunable Neural Network
                │
       ┌────────┴────────┐
       │                 │
     Weights          Gate Scores
       │                 │
       │              Sigmoid
       │                 │
       └────────┬────────┘
                ▼
          Gated Weights
                │
                ▼
          Classification
                │
                ▼
       Cross-Entropy Loss
                │
                +
       Sparsity Regularization
                │
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
            /       \
           /         \
      Accuracy      Sparsity
Installation
1. Clone the repository
git clone https://github.com/Akash-kotikala/self-pruning-neural-network.git
2. Enter the project directory
cd self-pruning-neural-network
3. Install dependencies
pip install -r requirements.txt
4. Run the project
python submission.py

The CIFAR-10 dataset will be downloaded automatically if required.

Requirements

The project requires:

Python 3.10+
PyTorch
Torchvision
Matplotlib

Dependencies are listed in requirements.txt.

Project Structure
self-pruning-neural-network/
│
├── submission.py
├── README.md
├── requirements.txt
├── .gitignore
│
└── results/
    └── gate_distribution.png
submission.py

Contains the complete implementation:

PrunableLinear
Neural network architecture
Sparsity loss
Training loop
Evaluation loop
Sparsity calculation
Gate analysis
Experiment execution
Visualization
requirements.txt

Contains the Python packages required to run the project.

results/

Contains generated experimental visualizations.

Technologies Used
Python
PyTorch
Torchvision
Matplotlib
CUDA
Limitations

The current implementation uses fully connected layers rather than convolutional layers.

CIFAR-10 contains spatial image information, so a convolutional architecture could potentially provide better classification performance.

The primary objective of this project is to demonstrate the self-pruning mechanism and the effect of learnable gates and sparsity regularization.

The amount of pruning also depends on:

Sparsity coefficient λ
Number of training epochs
Gate initialization
Optimizer
Pruning threshold
Future Improvements

Potential improvements include:

Implement prunable convolutional layers.
Explore structured neuron and channel pruning.
Add a hard-pruning stage after training.
Introduce adaptive λ scheduling.
Compare different gate functions.
Measure actual memory reduction.
Measure inference-time improvements.
Compare self-pruning with conventional post-training pruning.
Evaluate the approach on larger datasets.
Conclusion

This project demonstrates a neural network that learns both its predictive parameters and which connections should remain active.

Instead of training a dense network and pruning it afterward, the pruning mechanism is integrated directly into the training process through learnable sigmoid gates and sparsity regularization.

The experiments show that increasing the sparsity coefficient increases the percentage of suppressed connections while maintaining comparable classification accuracy across the tested range.

The best test accuracy obtained in the final experiment was 55.71%, while the highest observed sparsity was 6.02%.