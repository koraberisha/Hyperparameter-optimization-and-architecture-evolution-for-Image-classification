# CNN Architecture Evolution - Implementation Plan

## Task Parallelism Architecture

### Core Components

1. **Job Queue System**
   ```
   ModelTrainingQueue
   ├── Pending Jobs (chromosomes waiting to be evaluated)
   ├── Running Jobs (models currently training)
   └── Completed Jobs (evaluated models with fitness scores)
   ```

2. **GPU Resource Manager**
   ```
   GPUManager
   ├── Track available GPUs
   ├── Assign models to free GPUs
   └── Monitor GPU memory/utilization
   ```

3. **Model Training Dispatcher**
   ```
   ModelTrainer
   ├── Convert chromosome to PyTorch model
   ├── Train on assigned GPU
   └── Report validation metrics as fitness
   ```

4. **Genetic Algorithm Coordinator**
   ```
   EvolutionCoordinator
   ├── Initialize population
   ├── Queue chromosomes for evaluation
   ├── Process fitness results as they arrive
   └── Create next generation when current is complete
   ```

### Implementation Details

#### 1. Job Queue Implementation
```python
class ModelQueue:
    def __init__(self):
        self.pending = []      # Chromosomes waiting for evaluation
        self.running = {}      # {job_id: (chromosome, gpu_id, process)}
        self.completed = {}    # {job_id: (chromosome, fitness)}
        self.job_counter = 0
    
    def add_job(self, chromosome):
        job_id = self.job_counter
        self.pending.append((job_id, chromosome))
        self.job_counter += 1
        return job_id
    
    def get_next_job(self):
        if not self.pending:
            return None
        return self.pending.pop(0)
    
    def mark_job_running(self, job_id, chromosome, gpu_id, process):
        self.running[job_id] = (chromosome, gpu_id, process)
    
    def mark_job_completed(self, job_id, fitness):
        if job_id in self.running:
            chromosome, gpu_id, _ = self.running.pop(job_id)
            self.completed[job_id] = (chromosome, fitness)
            return gpu_id
        return None
```

#### 2. GPU Manager Implementation
```python
class GPUManager:
    def __init__(self, gpu_ids=None):
        # If no GPUs specified, detect all available GPUs
        self.gpu_ids = gpu_ids or self._detect_gpus()
        self.available_gpus = list(self.gpu_ids)
    
    def _detect_gpus(self):
        # Use torch.cuda.device_count() to detect available GPUs
        return list(range(torch.cuda.device_count()))
    
    def get_gpu(self):
        if not self.available_gpus:
            return None
        return self.available_gpus.pop(0)
    
    def release_gpu(self, gpu_id):
        if gpu_id not in self.available_gpus:
            self.available_gpus.append(gpu_id)
```

#### 3. Model Training Process
```python
def train_model_process(chromosome, gpu_id, result_queue):
    """Function to run in a separate process for model training"""
    try:
        # Set GPU context
        torch.cuda.set_device(gpu_id)
        
        # Convert chromosome to model architecture
        model = chromosome_to_model(chromosome)
        
        # Move model to GPU
        model = model.cuda()
        
        # Apply torch.compile if available (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            model = torch.compile(model)
        
        # Load dataset
        train_loader, val_loader = get_data_loaders()
        
        # Train for fixed number of epochs
        trainer = ModelTrainer(model)
        val_acc = trainer.train(train_loader, val_loader, epochs=5)
        
        # Return the validation accuracy as fitness
        result_queue.put((chromosome, val_acc))
    
    except Exception as e:
        # Handle exceptions gracefully
        result_queue.put((chromosome, -1.0, str(e)))
```

#### 4. Evolution Coordinator
```python
class EvolutionCoordinator:
    def __init__(self, pop_size=10, genome_size=20, generations=20):
        self.pop_size = pop_size
        self.genome_size = genome_size
        self.generations = generations
        self.current_generation = 0
        self.population = []
        self.fitness_values = {}
        self.best_chromosome = None
        self.best_fitness = -1
        
        # Setup multiprocessing components
        self.job_queue = ModelQueue()
        self.gpu_manager = GPUManager()
        self.result_queue = multiprocessing.Queue()
    
    def initialize_population(self):
        """Generate initial random population"""
        self.population = [
            ''.join(random.choice('01') for _ in range(self.genome_size))
            for _ in range(self.pop_size)
        ]
    
    def queue_population_for_evaluation(self):
        """Add all chromosomes in current population to job queue"""
        for chromosome in self.population:
            self.job_queue.add_job(chromosome)
    
    def dispatch_jobs(self):
        """Assign pending jobs to available GPUs"""
        while self.job_queue.pending and self.gpu_manager.available_gpus:
            job_id, chromosome = self.job_queue.get_next_job()
            gpu_id = self.gpu_manager.get_gpu()
            
            if gpu_id is not None:
                # Start a new process to train this model on the assigned GPU
                p = multiprocessing.Process(
                    target=train_model_process,
                    args=(chromosome, gpu_id, self.result_queue)
                )
                p.start()
                self.job_queue.mark_job_running(job_id, chromosome, gpu_id, p)
    
    def collect_results(self, timeout=0.1):
        """Check for completed model evaluations"""
        try:
            while True:
                chromosome, fitness = self.result_queue.get(block=False)
                job_id = next(
                    jid for jid, (chrom, _, _) in self.job_queue.running.items() 
                    if chrom == chromosome
                )
                gpu_id = self.job_queue.mark_job_completed(job_id, fitness)
                self.gpu_manager.release_gpu(gpu_id)
                self.fitness_values[chromosome] = fitness
                
                # Update best chromosome if needed
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_chromosome = chromosome
        except Empty:
            # No more results in queue
            pass
    
    def create_next_generation(self):
        """Apply selection, crossover, and mutation to create next generation"""
        # Selection
        selected = self.tournament_selection()
        
        # Crossover & Mutation
        new_population = []
        for i in range(0, self.pop_size, 2):
            if i+1 < len(selected):
                offspring1, offspring2 = self.crossover(selected[i], selected[i+1])
                new_population.append(self.mutation(offspring1))
                new_population.append(self.mutation(offspring2))
        
        self.population = new_population
        self.current_generation += 1
    
    def tournament_selection(self):
        # Implementation of tournament selection
        pass
    
    def crossover(self, parent1, parent2):
        # Implementation of crossover
        pass
    
    def mutation(self, chromosome):
        # Implementation of mutation
        pass
    
    def run(self):
        """Main evolution loop"""
        self.initialize_population()
        
        for gen in range(self.generations):
            self.current_generation = gen
            print(f"Generation {gen}/{self.generations}")
            
            # Queue current population for evaluation
            self.queue_population_for_evaluation()
            
            # Keep dispatching jobs and collecting results until all evaluated
            while (self.job_queue.pending or self.job_queue.running):
                self.dispatch_jobs()
                self.collect_results()
                time.sleep(0.1)  # Small delay to prevent CPU spinning
            
            # Print generation stats
            avg_fitness = sum(self.fitness_values.values()) / len(self.fitness_values)
            print(f"Generation {gen} complete: Avg fitness = {avg_fitness:.4f}, Best = {self.best_fitness:.4f}")
            
            # Create next generation
            if gen < self.generations - 1:
                self.create_next_generation()
        
        print(f"Evolution complete! Best chromosome: {self.best_chromosome}")
        print(f"Best fitness: {self.best_fitness}")
        
        return self.best_chromosome, self.best_fitness
```

## PyTorch Model Generator

For the PyTorch model generation from chromosomes, we'll implement a structure similar to your original but with modern PyTorch components:

```python
def chromosome_to_model(chromosome):
    """Convert binary chromosome to a PyTorch CNN model"""
    # Parse chromosome bits
    out_channels = int(chromosome[0:2], 2)
    kernel_size = int(chromosome[3:5], 2)
    activation_type = int(chromosome[6:8], 2)
    pool_size = int(chromosome[9:11], 2)
    layer_choice = chromosome[12:19]
    
    # Convert to actual parameter values
    output_options = [16, 32, 64, 128, 256]
    conv_options = [2, 3, 5, 7]
    activation_options = [nn.ReLU(), nn.ELU(), nn.LeakyReLU(), nn.GELU()]
    layer_levels = [1, 2, 3, 4, 5]
    
    # Create model
    model = nn.Sequential()
    
    # Input layer
    model.append(nn.Conv2d(3, output_options[out_channels], 
                          kernel_size=conv_options[kernel_size], 
                          padding='same', stride=1))
    
    # Add feature extraction blocks based on layer_choice
    for i, choice in enumerate(layer_choice):
        if i >= layer_levels[out_channels]:
            break
            
        if choice == '1':
            # Single conv block
            model.append(create_single_conv_block(
                output_options[min(out_channels+i, len(output_options)-1)],
                conv_options[kernel_size],
                activation_options[activation_type]
            ))
        else:
            # Double conv block
            model.append(create_double_conv_block(
                output_options[min(out_channels+i, len(output_options)-1)],
                conv_options[kernel_size],
                activation_options[activation_type]
            ))
        
        # Add dropout after each block
        model.append(nn.Dropout(0.25))
    
    # Classification layers
    model.append(nn.Flatten())
    model.append(nn.Linear(model.get_feature_size(), 2048))
    model.append(activation_options[0])  # ReLU
    model.append(nn.Dropout(0.5))
    model.append(nn.Linear(2048, 1024))
    model.append(activation_options[0])  # ReLU
    model.append(nn.Dropout(0.5))
    model.append(nn.Linear(1024, 100))  # 100 classes for CIFAR-100
    
    return model

def create_single_conv_block(channels, kernel_size, activation):
    """Create a single conv block with batch norm and pooling"""
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size, padding='same', stride=1),
        nn.BatchNorm2d(channels),
        activation,
        nn.MaxPool2d(2, padding=0)
    )

def create_double_conv_block(channels, kernel_size, activation):
    """Create a double conv block with batch norm and pooling"""
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size, padding='same', stride=2),
        nn.Conv2d(channels, channels, kernel_size, padding='same', stride=1),
        nn.BatchNorm2d(channels),
        activation,
        nn.MaxPool2d(2, padding=0)
    )
```

## Running the System

This simple script shows how to use our implementation:

```python
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run CNN Architecture Evolution')
    parser.add_argument('--pop_size', type=int, default=10, help='Population size')
    parser.add_argument('--generations', type=int, default=20, help='Number of generations')
    parser.add_argument('--genome_size', type=int, default=20, help='Size of chromosome in bits')
    parser.add_argument('--gpus', type=str, default=None, help='Comma-separated list of GPU IDs to use')
    
    args = parser.parse_args()
    
    # Set up GPU IDs if specified
    gpu_ids = None
    if args.gpus:
        gpu_ids = [int(x) for x in args.gpus.split(',')]
    
    # Create and run the evolution coordinator
    coordinator = EvolutionCoordinator(
        pop_size=args.pop_size,
        genome_size=args.genome_size,
        generations=args.generations,
        gpu_ids=gpu_ids
    )
    
    best_chromosome, best_fitness = coordinator.run()
    
    # Save the best model
    best_model = chromosome_to_model(best_chromosome)
    torch.save(best_model.state_dict(), 'best_model.pt')
    
    # Save chromosome and configuration
    with open('best_chromosome.json', 'w') as f:
        json.dump({
            'chromosome': best_chromosome,
            'fitness': best_fitness,
            'config': vars(args)
        }, f, indent=2)
    
    print(f"Best model saved to best_model.pt")
    print(f"Best chromosome saved to best_chromosome.json")
```