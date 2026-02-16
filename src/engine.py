
import torch
from tqdm.auto import tqdm

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, delta=0, verbose=False, path='checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train_step(model, dataloader, loss_fn, optimizer, device):
    """
    Performs a single training step.
    
    Returns:
        train_loss (float), train_acc (float)
    """
    model.train()
    train_loss, train_acc = 0, 0
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        y_pred = model(X)
        
        # Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        
        # Calculate accuracy
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
        
        # Optimizer zero grad
        optimizer.zero_grad()
        
        # Loss backward
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
    # Adjust metrics
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def validate_step(model, dataloader, loss_fn, device):
    """
    Performs a single validation step.
    
    Returns:
        val_loss (float), val_acc (float)
    """
    model.eval()
    val_loss, val_acc = 0, 0
    
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            
            # Forward pass
            val_pred = model(X)
            
            # Calculate loss
            loss = loss_fn(val_pred, y)
            val_loss += loss.item()
            
            # Calculate accuracy
            val_pred_class = val_pred.argmax(dim=1)
            val_acc += (val_pred_class == y).sum().item()/len(val_pred)
            
    # Adjust metrics
    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)
    return val_loss, val_acc

def train_model(model, train_dataloader, val_dataloader, loss_fn, optimizer, scheduler, output_path, epochs, device, early_stopping_patience=5):
    """
    Trains and validates the model over a number of epochs.
    
    Args:
        model: PyTorch model to train.
        train_dataloader: DataLoader for training data.
        val_dataloader: DataLoader for validation data.
        loss_fn: Loss function.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        output_path: Path to save the best model.
        epochs: Number of epochs to train.
        device: Device to run on.
        early_stopping_patience: Patience for early stopping.
        
    Returns:
        results: Dictionary containing training and validation metrics.
    """
    results = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True, path=output_path)
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        
        val_loss, val_acc = validate_step(model=model,
                                          dataloader=val_dataloader,
                                          loss_fn=loss_fn,
                                          device=device)
        
        # Step the scheduler
        if scheduler:
            scheduler.step()
            
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_acc: {val_acc:.4f}"
        )
        
        # Save metrics
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        
        # Early Stopping check
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
            
    # Load the best model
    model.load_state_dict(torch.load(output_path))
    return results

