import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from model_generator import create_model_from_chromosome

def load_dataset(dataset_name):
    """Load and prepare dataset"""
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
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
    
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
        class_names = [f"Class_{i}" for i in range(100)]  # CIFAR-100 has many classes
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train_dataset, test_dataset, num_classes, class_names

def load_model(model_path, chromosome, num_classes):
    """Load a trained model from a saved checkpoint"""
    # Create model with the same architecture
    model = create_model_from_chromosome(chromosome, num_classes)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path))
    
    return model

def evaluate_model(model, test_loader, device):
    """Evaluate model performance on test set"""
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # Collect statistics
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Store for confusion matrix
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    accuracy = correct / total
    
    return accuracy, all_predictions, all_targets

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_sample_predictions(test_dataset, model, device, class_names, num_samples=5, save_path=None):
    """Plot sample predictions from the model"""
    # Get random samples
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    
    # Set up subplot
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Get sample
            image, label = test_dataset[idx]
            
            # Get prediction
            output = model(image.unsqueeze(0).to(device))
            _, predicted = output.max(1)
            predicted = predicted.item()
            
            # Convert image for display
            image = image.cpu().numpy().transpose(1, 2, 0)
            image = (image * 0.5 + 0.5).clip(0, 1)  # Unnormalize
            
            # Plot
            axes[i].imshow(image)
            color = 'green' if predicted == label else 'red'
            title = f"True: {class_names[label]}\nPred: {class_names[predicted]}"
            axes[i].set_title(title, color=color)
            axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main(args):
    """Main evaluation function"""
    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    train_dataset, test_dataset, num_classes, class_names = load_dataset(args.dataset)
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model info
    if args.info_path:
        with open(args.info_path, 'r') as f:
            info = json.load(f)
            chromosome = info.get('chromosome')
    else:
        chromosome = args.chromosome
    
    if not chromosome:
        raise ValueError("Either --info_path or --chromosome must be provided")
    
    # Load and prepare model
    model = load_model(args.model_path, chromosome, num_classes)
    model = model.to(device)
    
    # Evaluate model
    accuracy, all_predictions, all_targets = evaluate_model(model, test_loader, device)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Print detailed metrics
    print("\nClassification Report:")
    report = classification_report(all_targets, all_predictions, 
                                  target_names=class_names if len(class_names) <= 20 else None)
    print(report)
    
    # Save report to file
    with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
        f.write(f"Test accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    # Generate confusion matrix if not too many classes
    if num_classes <= 20:  # Only for CIFAR-10 or smaller datasets
        plot_confusion_matrix(
            all_targets, 
            all_predictions,
            class_names,
            save_path=os.path.join(args.output_dir, "confusion_matrix.png")
        )
    
    # Plot sample predictions
    plot_sample_predictions(
        test_dataset,
        model,
        device,
        class_names,
        num_samples=8,
        save_path=os.path.join(args.output_dir, "sample_predictions.png")
    )
    
    print(f"Evaluation results saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained CNN model")
    
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to the model checkpoint (.pt file)")
    
    # Model identification (one of these is required)
    parser.add_argument("--info_path", type=str, default=None,
                      help="Path to the model info.json file (contains chromosome)")
    parser.add_argument("--chromosome", type=str, default=None,
                      help="Binary chromosome string for the model architecture")
    
    # Other settings
    parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"],
                      default="cifar10", help="Dataset to evaluate on")
    parser.add_argument("--batch_size", type=int, default=128,
                      help="Batch size for evaluation")
    parser.add_argument("--gpu", type=int, default=0,
                      help="GPU ID to use")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                      help="Directory to save evaluation results")
    
    args = parser.parse_args()
    main(args)