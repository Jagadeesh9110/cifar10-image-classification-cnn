
import torch
import torch.nn as nn
import argparse
import model
import data_setup
import engine
import utils
import os

def main():
    parser = argparse.ArgumentParser(description="Resume-Level CIFAR-10 Training Script")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"], help="Optimizer (adamw or sgd)")
    parser.add_argument("--experiment_name", type=str, default="experiment", help="Name of the experiment")
    
    args = parser.parse_args()
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    utils.set_seed(42)
    
    # 1. Data Setup
    print("Loading data...")
    train_dataloader, val_dataloader, test_dataloader, class_names = data_setup.get_dataloaders(
        batch_size=args.batch_size,
        valid_split=0.1
    )
    print(f"Number of training batches: {len(train_dataloader)}")
    print(f"Number of validation batches: {len(val_dataloader)}")
    print(f"Number of testing batches: {len(test_dataloader)}")
    
    # 2. Model Setup
    print("Initializing model...")
    simple_vgg = model.SimpleVGG(num_classes=len(class_names), dropout_rate=args.dropout).to(device)
    
    # 3. Loss, Optimizer, Scheduler
    loss_fn = nn.CrossEntropyLoss()
    
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(simple_vgg.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(simple_vgg.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    else:
        raise ValueError("Invalid optimizer choice")
        
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 4. Training
    # Create experiment directory
    experiment_dir = os.path.join("experiments", args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 4. Training
    print(f"Starting training for {args.epochs} epochs...")
    results = engine.train_model(
        model=simple_vgg,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        output_path=os.path.join(experiment_dir, "model.pth"),
        epochs=args.epochs,
        device=device,
        early_stopping_patience=5
    )
    
    # 5. Metrics & Plotting
    print("Generating loss curves...")
    utils.plot_loss_curves(results)
    
    print("Computing confusion matrix...")
    utils.compute_confusion_matrix(simple_vgg, test_dataloader, device, class_names)
    
    print("Computing per-class accuracy...")
    utils.per_class_accuracy(simple_vgg, test_dataloader, device, class_names)
    
    # 6. Final Test Evaluation
    print("Evaluating on test set...")
    test_loss, test_acc = engine.validate_step(simple_vgg, test_dataloader, loss_fn, device)
    print(f"Final Test Loss: {test_loss:.4f} | Final Test Accuracy: {test_acc:.4f}")
    
    # Save logs
    import json
    log_data = {
        "experiment_name": args.experiment_name,
        "config": vars(args),
        "results": results,
        "final_test_loss": test_loss,
        "final_test_accuracy": test_acc
    }
    
    os.makedirs("results", exist_ok=True)
    log_file_path = os.path.join("results", "logs.json")
    
    # Append to existing logs if file exists, else create new list
    if os.path.exists(log_file_path):
        try:
            with open(log_file_path, "r") as f:
                logs = json.load(f)
                if not isinstance(logs, list):
                    logs = [logs]
        except json.JSONDecodeError:
            logs = []
    else:
        logs = []
        
    logs.append(log_data)
    
    with open(log_file_path, "w") as f:
        json.dump(logs, f, indent=4)
    print(f"Logs saved to {log_file_path}")

if __name__ == "__main__":
    main()
