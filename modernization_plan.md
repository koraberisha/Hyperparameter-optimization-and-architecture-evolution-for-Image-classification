# Modernization Plan: Task-Based Parallel CNN Evolution

## Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Population     │────▶│  Job Dispatcher │────▶│  GPU Workers    │
│  Manager        │     │                 │     │                 │
│                 │     │                 │     │                 │
└────────┬────────┘     └─────────────────┘     └────────┬────────┘
         │                                               │
         │                                               │
         │                                               │
         │                                               ▼
┌────────▼────────┐                           ┌─────────────────┐
│                 │                           │                 │
│  Genetic        │◀──────────────────────────│  Results        │
│  Operations     │                           │  Collector      │
│                 │                           │                 │
└─────────────────┘                           └─────────────────┘
```

## Core Components

### 1. Job Queue System
- Simple Python queue implementation
- Each job = (chromosome, train_config)
- Priority queue based on job importance/wait time

```python
# job_queue.py
import queue
from dataclasses import dataclass

@dataclass
class ModelJob:
    chromosome: str
    generation: int
    individual_id: int
    config: dict
    priority: int = 0

class JobQueue:
    def __init__(self):
        self.queue = queue.PriorityQueue()
    
    def add_job(self, job: ModelJob):
        self.queue.put((job.priority, job))
    
    def get_job(self):
        if self.queue.empty():
            return None
        return self.queue.get()[1]
```

### 2. GPU Worker Manager
- Tracks available GPUs
- Assigns jobs to free GPUs
- Manages worker processes

```python
# worker_manager.py
import os
import torch
import subprocess
from typing import List

class GPUManager:
    def __init__(self, gpu_ids: List[int]):
        self.gpu_ids = gpu_ids
        self.available_gpus = set(gpu_ids)
        self.running_jobs = {}  # gpu_id -> job
    
    def get_available_gpu(self):
        if not self.available_gpus:
            return None
        return next(iter(self.available_gpus))
    
    def assign_job(self, job, gpu_id):
        self.available_gpus.remove(gpu_id)
        self.running_jobs[gpu_id] = job
        # Launch process with specific GPU
        process = subprocess.Popen([
            "python", "train_model.py",
            "--chromosome", job.chromosome,
            "--gpu", str(gpu_id),
            "--generation", str(job.generation),
            "--individual_id", str(job.individual_id),
            "--output_dir", f"results/gen{job.generation}/ind{job.individual_id}"
        ])
        return process
    
    def release_gpu(self, gpu_id):
        if gpu_id in self.running_jobs:
            del self.running_jobs[gpu_id]
            self.available_gpus.add(gpu_id)
```

### 3. PyTorch Model Generator
- Converts chromosome to PyTorch model
- Uses torch.compile for optimization
- Configurable CNN architecture blocks

```python
# model_generator.py
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation_type):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             padding='same', stride=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = self._get_activation(activation_type)
        self.pool = nn.MaxPool2d(2, 2, padding=0)
    
    def _get_activation(self, activation_type):
        if activation_type == 0:
            return nn.ReLU()
        elif activation_type == 1:
            return nn.ELU()
        elif activation_type == 2:
            return nn.LeakyReLU(0.1)
        else:
            return nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.pool(x)
        return x

class CNNFromChromosome(nn.Module):
    def __init__(self, chromosome, num_classes=10):
        super().__init__()
        # Decode chromosome
        out_ch, kernel_size, activation, layer_configs = self._decode_chromosome(chromosome)
        
        # Create feature extraction layers
        self.features = self._build_feature_layers(out_ch, kernel_size, activation, layer_configs)
        
        # Create classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        
    def _decode_chromosome(self, chromosome):
        # Implementation similar to your convert_bitstring function
        out_ch = int(chromosome[0:2], 2)
        kernel_size = int(chromosome[3:5], 2)
        activation = int(chromosome[6:8], 2)
        layer_configs = chromosome[12:19]
        return out_ch, kernel_size, activation, layer_configs
    
    def _build_feature_layers(self, out_ch, kernel_size, activation, layer_configs):
        # Build the feature extraction layers based on chromosome
        # This will calculate self.feature_dim for the classifier
        # ...

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

### 4. Training Runner
- Handles single model training
- Calculates fitness scores
- Outputs results in standardized format

```python
# train_model.py
import torch
import argparse
import json
import os
from model_generator import CNNFromChromosome
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def train(args):
    # Set GPU
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    train_data, test_data = load_dataset(args.dataset)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=128)
    
    # Create model from chromosome
    model = CNNFromChromosome(args.chromosome, 
                             num_classes=100 if args.dataset == 'cifar100' else 10)
    
    # Compile model if supported
    if hasattr(torch, 'compile'):
        model = torch.compile(model)
    
    model = model.to(device)
    
    # Train for specified epochs
    results = train_epochs(model, train_loader, test_loader, 
                          epochs=args.epochs, device=device)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump(results, f)
    
    # Save model
    torch.save(model.state_dict(), f"{args.output_dir}/model.pt")
    
    # Return final accuracy as fitness
    return results["val_accuracy"][-1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chromosome", type=str, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--individual_id", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    
    train(args)
```

### 5. Genetic Algorithm Controller
- Manages population evolution
- Dispatches new generation to job queue
- Collects fitness results

```python
# evolution.py
import numpy as np
import argparse
import time
import os
from job_queue import JobQueue, ModelJob
from worker_manager import GPUManager

class GeneticAlgorithm:
    def __init__(self, pop_size=10, genome_size=20, mutation_prob=0.1):
        self.pop_size = pop_size
        self.genome_size = genome_size
        self.mutation_prob = mutation_prob
        self.population = self._init_population()
        self.fitness_scores = {}
    
    def _init_population(self):
        # Generate random chromosomes
        return [''.join(np.random.choice(['0', '1']) for _ in range(self.genome_size))
                for _ in range(self.pop_size)]
    
    def selection(self):
        # Tournament selection
        # ...
    
    def crossover(self, parent1, parent2):
        # Single-point crossover
        # ...
    
    def mutation(self, individual):
        # Bitwise mutation
        # ...
    
    def evolve(self):
        # Generate next population
        # ...
        
    def get_population_jobs(self, generation):
        # Convert population to jobs
        jobs = []
        for i, chromosome in enumerate(self.population):
            jobs.append(ModelJob(
                chromosome=chromosome,
                generation=generation,
                individual_id=i,
                config={},
                priority=0
            ))
        return jobs
    
    def update_fitness(self, results):
        # Update fitness scores from results
        self.fitness_scores.update(results)

def main(args):
    # Initialize components
    job_queue = JobQueue()
    gpu_manager = GPUManager(args.gpus)
    ga = GeneticAlgorithm(pop_size=args.population, genome_size=args.genome_size)
    
    # Main evolution loop
    for generation in range(args.generations):
        print(f"Generation {generation}")
        
        # Add current population jobs to queue
        jobs = ga.get_population_jobs(generation)
        for job in jobs:
            job_queue.add_job(job)
        
        # Process jobs until generation is complete
        results = {}
        running_processes = {}  # gpu_id -> (process, job)
        
        while len(results) < len(jobs):
            # Check completed processes
            for gpu_id, (process, job) in list(running_processes.items()):
                if process.poll() is not None:
                    # Process finished
                    # Read results from output file
                    result_file = f"results/gen{job.generation}/ind{job.individual_id}/results.json"
                    if os.path.exists(result_file):
                        with open(result_file, "r") as f:
                            import json
                            job_results = json.load(f)
                            results[job.chromosome] = job_results["val_accuracy"][-1]
                    
                    # Release GPU
                    gpu_manager.release_gpu(gpu_id)
                    del running_processes[gpu_id]
            
            # Start new jobs on available GPUs
            gpu_id = gpu_manager.get_available_gpu()
            if gpu_id is not None:
                job = job_queue.get_job()
                if job:
                    process = gpu_manager.assign_job(job, gpu_id)
                    running_processes[gpu_id] = (process, job)
            
            # Sleep briefly to prevent CPU spinning
            time.sleep(0.1)
        
        # Update fitness scores
        ga.update_fitness(results)
        
        # Evolve population for next generation
        ga.evolve()
    
    # Final generation completed
    best_chromosome = max(ga.fitness_scores.items(), key=lambda x: x[1])[0]
    print(f"Best chromosome: {best_chromosome}")
    print(f"Best fitness: {ga.fitness_scores[best_chromosome]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--genome_size", type=int, default=20)
    args = parser.parse_args()
    
    main(args)
```

## Key Features

1. **Simple Task Distribution**
   - Queue-based job management
   - Process-based worker isolation
   - GPU assignment tracking

2. **Modern PyTorch Components**
   - torch.compile for performance
   - Modern CNN architectures
   - Clean module design

3. **Resource Efficiency**
   - Dynamic GPU allocation
   - Fault tolerance via file-based results
   - No dependencies on external services

4. **Scalability**
   - Works with any number of GPUs
   - Can be extended to multiple machines
   - Dockerizable for consistent environments

## Extension Options

1. **Performance Monitoring**
   - Add TensorBoard logging for model comparison
   - Track resource utilization

2. **Early Stopping**
   - Implement early fitness prediction
   - Terminate poor performing models early

3. **Multi-Machine Support**
   - Add simple file-based or socket-based coordination
   - Share results via shared filesystem or object storage