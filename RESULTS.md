
# Experimental Results

This document tracks the performance of different model configurations on the CIFAR-10 dataset.

## Experiment 1: Baseline (No Dropout)
- **Command:** `python train.py --epochs 50 --dropout 0.0 --experiment_name exp_no_dropout`
- **Goal:** Establish a baseline performance without regularization in the classifier.
- **Expected Outcome:** High training accuracy but significant overfitting (large gap between train/val accuracy).
- **Recorded Results:**
    -   Train Accuracy: [Enter Value]
    -   Test Accuracy: [Enter Value]
    -   Observations: [Enter Observations]

## Experiment 2: Regularized (Dropout 0.5)
- **Command:** `python src/train.py --epochs 10 --dropout 0.5 --experiment_name exp_dropout`
- **Goal:** Reduce overfitting using Dropout in the fully connected layers.
- **Expected Outcome:** Better generalization.
- **Recorded Results:**
    -   Train Accuracy: 88.15% (Epoch 10)
    -   Val Accuracy: 87.54% (Epoch 10)
    -   Test Accuracy: **87.25%**
    -   Test Loss: 0.3803
    -   Observations: Achieved ~87% accuracy in just 10 epochs on CPU. Validation loss consistently decreased, indicating effective learning without significant overfitting.

## Experiment 3: Optimizer Comparison (SGD)
- **Command:** `python train.py --epochs 50 --optimizer sgd --dropout 0.5 --experiment_name exp_sgd`
- **Goal:** Compare AdamW with SGD (classic optimizer for CNNs).
- **Expected Outcome:** SGD might converge slower but could potentially reach a better final minima or generalize differently.
- **Recorded Results:**
    -   Train Accuracy: [Enter Value]
    -   Test Accuracy: [Enter Value]
    -   Observations: [Enter Observations]

## Analysis & Conclusion

[Summarize which configuration worked best and why. Discuss the trade-offs observed.]
