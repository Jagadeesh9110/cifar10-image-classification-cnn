
# CIFAR-10 Classification with Custom VGG-Style CNN

A professional, modular deep learning project implementing a custom VGG-style CNN for image classification on the CIFAR-10 dataset using PyTorch.

## Project Overview

This project demonstrates a production-level implementation of a Convolutional Neural Network (CNN) from scratch, adhering to best practices in deep learning engineering.

**Key Features:**
-   **Modular Architecture:** Clean separation of concerns (`model.py`, `data_setup.py`, `engine.py`, `train.py`).
-   **Custom VGG-Style CNN:** Custom VGG-style CNN designed for CIFAR-10 classification without relying on pretrained models.
-   **Robust Training Pipeline:** Includes **Early Stopping**, **Cosine Annealing Learning Rate Scheduler**, and **Model Checkpointing**.
-   **Comprehensive Evaluation:** Per-class accuracy, Confusion Matrix generation, and Loss/Accuracy curve plotting.
-   **Experimentation Ready:** Easily configurable for ablation studies (Dropout, Optimizer comparisons).

## Architecture

The model uses a VGG-inspired design with 3 main convolutional blocks followed by a classifier.

**Input:** `(Batch_Size, 3, 32, 32)`

1.  **Block 1:**
    -   `Conv2d` (3 -> 32) -> `BatchNorm` -> `ReLU`
    -   `Conv2d` (32 -> 64) -> `BatchNorm` -> `ReLU`
    -   `MaxPool2d` (2x2) -> Output: `(64, 16, 16)`

2.  **Block 2:**
    -   `Conv2d` (64 -> 128) -> `BatchNorm` -> `ReLU`
    -   `Conv2d` (128 -> 128) -> `BatchNorm` -> `ReLU`
    -   `MaxPool2d` (2x2) -> Output: `(128, 8, 8)`

3.  **Block 3:**
    -   `Conv2d` (128 -> 256) -> `BatchNorm` -> `ReLU`
    -   `Conv2d` (256 -> 256) -> `BatchNorm` -> `ReLU`
    -   `MaxPool2d` (2x2) -> Output: `(256, 4, 4)`

4.  **Classifier:**
    -   **Global Average Pooling** (AdaptiveAvgPool2d to 1x1)
    -   `Flatten`
    -   `Linear` (256 -> 128) -> `ReLU` -> `Dropout` (0.5)
    -   `Linear` (128 -> 10)

## Project Structure
```bash
├── src/
│   ├── data_setup.py    # Data loading, splitting, and transformations
│   ├── engine.py        # Training and evaluation loops, EarlyStopping
│   ├── model.py         # SimpleVGG class definition
│   ├── train.py         # Main entry point for training and experiments
│   └── utils.py         # Utility functions (plotting, metrics, seed)
├── results/             # Directory for visualizations and logs
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/cifar10-project.git
    cd cifar10-project
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Train the Model
To train the model with default settings (20 epochs, AdamW, Dropout 0.5):
```bash
python src/train.py --epochs 20 --batch_size 64 --experiment_name my_model
```

### Run Experiments
You can easily replicate the experimental results by running the following commands:

**Experiment 1: No Dropout**
```bash
python src/train.py --epochs 50 --dropout 0.0 --experiment_name exp_no_dropout
```

**Experiment 2: With Dropout (0.5)**
```bash
python src/train.py --epochs 50 --dropout 0.5 --experiment_name exp_dropout
```

**Experiment 3: SGD Optimizer**
```bash
python src/train.py --epochs 50 --dropout 0.5 --optimizer sgd --experiment_name exp_sgd
```

## Results

## Results

### Final Experiment Results (CPU Run)

**Configuration:**
- **Epochs:** 10
- **Batch Size:** 64
- **Optimizer:** AdamW
- **Dropout:** 0.5
- **Device:** CPU

**Final Test Accuracy:** **87.25%**

### Loss Curves
![Loss Curves](results/exp_dropout/loss_curves.png)

### Confusion Matrix
![Confusion Matrix](results/exp_dropout/confusion_matrix.png)

### Final Metrics

- **Test Loss:** 0.3803
- **Test Accuracy:** 87.25%
- **Per-Class Accuracy:**
    - Automobile: 95.1%
    - Ship: 94.0%
    - Truck: 92.5%
    - Airplane: 91.6%
    - Frog: 90.4%

*Note: Results obtained after training for 10 epochs on CPU.*

## License
MIT