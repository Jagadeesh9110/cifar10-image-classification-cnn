
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path

def set_seed(seed: int = 42):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    """Saves a PyTorch model to a target directory."""
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with .pt or .pth"
    model_save_path = target_dir_path / model_name
    
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

def plot_loss_curves(results: dict, save_dir: str = "results"):
    """Plots training curves of a results dictionary."""
    loss = results['train_loss']
    test_loss = results['val_loss']

    accuracy = results['train_acc']
    test_accuracy = results['val_acc']

    epochs = range(len(results['train_loss']))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    
    save_path = Path(save_dir) / "loss_curves.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def compute_confusion_matrix(model, dataloader, device, class_names, save_dir: str = "results"):
    """Computes and plots a confusion matrix."""
    y_preds = []
    y_true = []
    
    model.eval()
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_logits = model(X)
            y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
            y_preds.append(y_pred.cpu())
            y_true.append(y.cpu())
            
    y_pred_tensor = torch.cat(y_preds)
    y_true_tensor = torch.cat(y_true)
    
    cm = confusion_matrix(y_true_tensor, y_pred_tensor)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    save_path = Path(save_dir) / "confusion_matrix.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def per_class_accuracy(model, dataloader, device, class_names):
    """Computes and prints accuracy for each class."""
    nb_classes = len(class_names)
    conf_mat = torch.zeros(nb_classes, nb_classes)
    
    model.eval()
    with torch.inference_mode():
        for inputs, classes in dataloader:
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                conf_mat[t.long(), p.long()] += 1

    per_class_acc = conf_mat.diag() / conf_mat.sum(1)

    print("\nPer-class Accuracy:")
    for i, c in enumerate(class_names):
        print(f"{c}: {per_class_acc[i].item() * 100:.2f}%")
