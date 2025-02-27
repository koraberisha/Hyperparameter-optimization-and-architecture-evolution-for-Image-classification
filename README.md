# Hyperparameter Optimization and Architecture Evolution in CNNs for Image Classification

This repository contains a modernized implementation for evolving convolutional neural network (CNN) architectures and hyperparameters using genetic algorithms, focused on image classification tasks.

## Overview

The project implements a novel approach to autonomously generating effective CNN architectures through evolutionary algorithms. Key aspects include:

- Utilizing genetic algorithms to evolve CNN architecture-parameter combinations
- Task-based parallel training for efficient multi-GPU utilization
- Evaluating performance on CIFAR-10 and CIFAR-100 datasets
- Encoding CNN structures and hyperparameters into binary chromosomes
- Analyzing the impact of different CNN components on network efficacy

## Architecture

The modernized codebase is implemented in PyTorch with horizontal scaling capabilities:

- **Distributed Training System**: Each model trains independently on available GPUs
- **PyTorch Implementation**: Modern CNN architecture with torch.compile optimization
- **Checkpointing**: Save and resume experiments at any generation
- **Result Analysis**: Comprehensive visualization of fitness progression

## Core Components

- `genetic_algorithm.py`: Implements the genetic algorithm engine
- `model_generator.py`: Generates PyTorch CNN models from binary chromosome encoding
- `train_model.py`: Single model training and evaluation
- `worker_manager.py`: Manages GPU resources and job assignments
- `job_queue.py`: Prioritized job queue for model training
- `evolution.py`: Main script coordinating the evolution process
- `evaluate.py`: Detailed evaluation of trained models

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cnn-hyperparameter-evolution.git
cd cnn-hyperparameter-evolution

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision matplotlib numpy scikit-learn seaborn
```

## Usage

### Running an Evolution Experiment

To start a new evolutionary run with default parameters:

```bash
python evolution.py --generations 20 --population 10 --gpus 0 1 2 3
```

Common configuration options:

```bash
# Run on CIFAR-100 with larger population
python evolution.py --dataset cifar100 --population 20 --generations 30 --gpus 0 1 2 3

# Run with custom genetic algorithm parameters
python evolution.py --mutation_prob 0.2 --tournament_size 4 --elitism 3 --gpus 0 1

# Run with specific training settings
python evolution.py --epochs 10 --batch_size 64 --gpus 0 1 2 3
```

### Resuming a Previous Run

To resume from a checkpoint:

```bash
python evolution.py --resume checkpoints/generation_5/ga_state.json --gpus 0 1 2 3
```

### Evaluating a Trained Model

To evaluate a trained model:

```bash
# Evaluate using model info file (contains chromosome)
python evaluate.py --model_path best_model/best_model.pt --info_path best_model/info.json

# Evaluate with explicit chromosome (for custom evaluations)
python evaluate.py --model_path results/gen3/ind2/best_model.pt --chromosome 0110010110110011010 --dataset cifar10
```

## Binary Encoding Scheme

The CNN architecture and hyperparameters are encoded in a binary chromosome:

- Bits 0-1: Filter size multiplier
- Bits 3-4: Kernel size selection 
- Bits 6-7: Activation function type
- Bits 9-10: Pooling settings
- Bits 12-18: Feature extraction layers (1 = single conv layer, 0 = double conv layer)

## Results

The project demonstrates the potential of genetic algorithms in CNN optimization:

- Performance improves over generations, with later generations showing more consistent results
- Best evolved architectures achieve competitive accuracy on classification tasks
- Task-based parallelism allows larger populations and more generations than the original implementation
- Modern PyTorch implementation offers better performance and compatibility

## Scaling

The implementation is designed for efficient scaling:

- **Single Machine, Multiple GPUs**: Uses all available GPUs on a single machine
- **Checkpoint-Based Distribution**: Run different generations on different machines
- **Resource Scalability**: Performance scales linearly with available GPUs

## Future Work

- Implementation of early stopping for poor-performing models
- Support for more diverse datasets beyond CIFAR
- Integration with hyperparameter optimization libraries
- Multi-objective optimization (accuracy, model size, inference speed)
- Advanced visualizations of model architectures

## Original Paper

For a deeper understanding of the underlying concepts, refer to the original research paper [Hyperparameter optimization and architecture evolution in Convolutional Neural Networks for Image classification](oldcode/Neuroevolution-2021-05-28-21-25.pdf) included in this repository.