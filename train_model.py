import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import json
import os
import time
from model_generator import create_model_from_chromosome

def load_dataset(dataset_name):
    """Load and prepare CIFAR-10 or CIFAR-100 dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10(
            root='./data', 
            train=True, 
            download=True, 
            transform=transform
        )
        
        test_dataset = datasets.CIFAR10(
            root='./data', 
            train=False, 
            download=True, 
            transform=transform
        )
        
        num_classes = 10
    
    elif dataset_name == 'cifar100':
        train_dataset = datasets.CIFAR100(
            root='./data', 
            train=True, 
            download=True, 
            transform=transform
        )
        
        test_dataset = datasets.CIFAR100(
            root='./data',
            train=False, 
            download=True, 
            transform=transform
        )
        
        num_classes = 100
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train_dataset, test_dataset, num_classes

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    # Calculate epoch metrics
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    # Calculate validation metrics
    val_loss = running_loss / total
    val_acc = correct / total
    
    return val_loss, val_acc

def train_model(args):
    """Train a model from chromosome and return fitness"""
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    train_dataset, test_dataset, num_classes = load_dataset(args.dataset)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Create model from chromosome (with compilation disabled by default)
    model = create_model_from_chromosome(
        args.chromosome, 
        num_classes, 
        use_compile=args.use_compile
    )
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=3, 
        verbose=True
    )
    
    # Initialize tracking variables
    start_time = time.time()
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0.0
    
    # Training loop
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Save metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{args.output_dir}/best_model.pt")
        
        # Log progress
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
              f"Time: {epoch_time:.2f}s")
    
    # Save final model
    torch.save(model.state_dict(), f"{args.output_dir}/final_model.pt")
    
    # Save training history
    history = {
        'train_loss': train_losses,
        'train_accuracy': train_accs,
        'val_loss': val_losses,
        'val_accuracy': val_accs,
        'best_val_accuracy': best_val_acc,
        'training_time': time.time() - start_time,
        'chromosome': args.chromosome,
        'epochs': args.epochs
    }
    
    with open(f"{args.output_dir}/history.json", "w") as f:
        json.dump(history, f)
    
    # Save model summary as text
    with open(f"{args.output_dir}/model_summary.txt", "w") as f:
        f.write(f"Chromosome: {args.chromosome}\n")
        f.write(f"Final validation accuracy: {val_accs[-1]:.4f}\n")
        f.write(f"Best validation accuracy: {best_val_acc:.4f}\n")
    
    # Return best validation accuracy as fitness
    return best_val_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN model from chromosome")
    parser.add_argument("--chromosome", type=str, required=True, help="Binary chromosome string")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--generation", type=int, required=True, help="Generation number")
    parser.add_argument("--individual_id", type=int, required=True, help="Individual ID")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"], 
                        help="Dataset to use")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--use_compile", action="store_true", help="Use torch.compile for optimization (default: False)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train model and get fitness
    fitness = train_model(args)
    
    print(f"Training completed. Fitness: {fitness:.4f}")
    
    # Write fitness to a separate file for easy access
    with open(f"{args.output_dir}/fitness.txt", "w") as f:
        f.write(f"{fitness}")