import argparse
import os
import time
import json
import matplotlib.pyplot as plt
from typing import Dict, List
import numpy as np

from genetic_algorithm import GeneticAlgorithm
from job_queue import JobQueue, ModelJob
from worker_manager import GPUManager

def setup_directories():
    """Create necessary directories for results and models"""
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="CNN Architecture Evolution with PyTorch")
    
    # Basic settings
    parser.add_argument("--generations", type=int, default=20, 
                      help="Number of generations to run")
    parser.add_argument("--population", type=int, default=10, 
                      help="Population size")
    parser.add_argument("--genome_size", type=int, default=20, 
                      help="Size of chromosome (binary string)")
    
    # GPU settings
    parser.add_argument("--gpus", type=int, nargs="+", default=[0], 
                      help="List of GPU IDs to use")
    
    # Genetic algorithm parameters
    parser.add_argument("--mutation_prob", type=float, default=0.1, 
                      help="Mutation probability")
    parser.add_argument("--tournament_size", type=int, default=3, 
                      help="Tournament selection size")
    parser.add_argument("--elitism", type=int, default=2, 
                      help="Number of top individuals to preserve unchanged")
    
    # Training parameters
    parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"], 
                      default="cifar10", help="Dataset to use")
    parser.add_argument("--epochs", type=int, default=5, 
                      help="Number of epochs for each model training")
    parser.add_argument("--batch_size", type=int, default=128, 
                      help="Batch size for training")
    
    # Checkpoint settings
    parser.add_argument("--resume", type=str, default=None, 
                      help="Path to GA state file to resume from")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", 
                      help="Directory to save checkpoints")
    parser.add_argument("--save_freq", type=int, default=1, 
                      help="Save checkpoint every N generations")
    
    return parser.parse_args()

def load_fitness_from_file(output_dir):
    """Load fitness from output file"""
    fitness_file = os.path.join(output_dir, "fitness.txt")
    if os.path.exists(fitness_file):
        with open(fitness_file, "r") as f:
            return float(f.read().strip())
    
    # Try loading from history.json as fallback
    history_file = os.path.join(output_dir, "history.json")
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)
            return history.get("best_val_accuracy", 0.0)
    
    return 0.0

def plot_generation_stats(stats, save_path=None):
    """Plot generation statistics over time"""
    # Convert stats to arrays for plotting
    generations = list(range(len(stats)))
    max_fitness = [s["max_fitness"] for s in stats]
    min_fitness = [s["min_fitness"] for s in stats]
    avg_fitness = [s["avg_fitness"] for s in stats]
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, max_fitness, 'b-', label='Max Fitness')
    plt.plot(generations, avg_fitness, 'g-', label='Avg Fitness')
    plt.plot(generations, min_fitness, 'r-', label='Min Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Validation Accuracy)')
    plt.title('Fitness Evolution Over Generations')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def run_evolution(args):
    """Run the evolutionary algorithm"""
    # Setup directories
    setup_directories()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize components
    job_queue = JobQueue()
    gpu_manager = GPUManager(args.gpus)
    
    # Initialize or load GA
    if args.resume:
        print(f"Resuming from {args.resume}")
        ga = GeneticAlgorithm.load_state(args.resume)
        # Extract the last generation from the resume file path
        last_gen = len(ga.generation_stats) - 1
        start_gen = last_gen + 1
    else:
        print("Starting new evolution run")
        ga = GeneticAlgorithm(
            pop_size=args.population, 
            genome_size=args.genome_size,
            mutation_prob=args.mutation_prob,
            tournament_size=args.tournament_size,
            elitism=args.elitism
        )
        start_gen = 0
    
    # Main evolution loop
    for generation in range(start_gen, args.generations):
        generation_start = time.time()
        print(f"\n===== Generation {generation} =====")
        
        # 1. Get jobs for current population
        jobs = []
        for i, chromosome in enumerate(ga.population):
            job = ModelJob(
                chromosome=chromosome,
                generation=generation,
                individual_id=i,
                config={
                    "dataset": args.dataset,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size
                }
            )
            jobs.append(job)
            job_queue.add_job(job)
        
        # 2. Process jobs until all complete
        results = {}  # Chromosome -> fitness
        running_processes = {}  # GPU_id -> process
        
        while len(results) < len(jobs):
            # Check for completed jobs
            finished_jobs = gpu_manager.check_finished_jobs()
            for gpu_id, job in finished_jobs:
                # Load fitness from output file
                output_dir = f"results/gen{job.generation}/ind{job.individual_id}"
                fitness = load_fitness_from_file(output_dir)
                
                # Store result
                results[job.chromosome] = fitness
                print(f"Model {job.individual_id} completed with fitness: {fitness:.4f}")
                
                # Release GPU
                gpu_manager.release_gpu(gpu_id)
            
            # Start new jobs on available GPUs
            while True:
                gpu_id = gpu_manager.get_available_gpu()
                if gpu_id is None or job_queue.is_empty():
                    break
                    
                job = job_queue.get_job()
                if job:
                    print(f"Starting model {job.individual_id} on GPU {gpu_id}")
                    process = gpu_manager.assign_job(job, gpu_id)
            
            # Sleep briefly to prevent CPU spinning
            time.sleep(1)
        
        # 3. Update GA with fitness results
        ga.update_fitness(results)
        
        # 4. Compute and print statistics
        stats = ga.compute_generation_stats()
        print(f"Generation {generation} complete in {time.time() - generation_start:.2f}s")
        print(f"Max fitness: {stats['max_fitness']:.4f}")
        print(f"Avg fitness: {stats['avg_fitness']:.4f}")
        print(f"Min fitness: {stats['min_fitness']:.4f}")
        print(f"Best chromosome: {stats['best_chromosome']}")
        
        # 5. Save checkpoint if needed
        if generation % args.save_freq == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"generation_{generation}")
            os.makedirs(checkpoint_path, exist_ok=True)
            ga.save_state(checkpoint_path)
            
            # Plot progress
            plot_generation_stats(
                ga.generation_stats,
                save_path=os.path.join(checkpoint_path, "fitness_plot.png")
            )
        
        # 6. Evolve population for next generation
        if generation < args.generations - 1:
            ga.evolve()
    
    # Save final results
    final_dir = os.path.join(args.checkpoint_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    ga.save_state(final_dir)
    
    # Plot final statistics
    plot_generation_stats(
        ga.generation_stats,
        save_path=os.path.join(final_dir, "fitness_plot.png")
    )
    
    # Save best model to a special directory
    if ga.best_chromosome:
        best_model_dir = "best_model"
        os.makedirs(best_model_dir, exist_ok=True)
        
        # Find which generation/individual was the best
        for gen_idx, gen_stats in enumerate(ga.generation_stats):
            for ind_idx, chrom in enumerate(ga.population):
                if chrom == ga.best_chromosome:
                    best_source = f"results/gen{gen_idx}/ind{ind_idx}"
                    
                    # Copy best model files
                    import shutil
                    for filename in ["best_model.pt", "history.json", "model_summary.txt"]:
                        source_file = os.path.join(best_source, filename)
                        target_file = os.path.join(best_model_dir, filename)
                        if os.path.exists(source_file):
                            shutil.copy(source_file, target_file)
                    
                    # Create info file
                    with open(os.path.join(best_model_dir, "info.json"), "w") as f:
                        info = {
                            "chromosome": ga.best_chromosome,
                            "fitness": ga.best_fitness,
                            "generation": gen_idx,
                            "individual_id": ind_idx
                        }
                        json.dump(info, f, indent=2)
                    
                    break
    
    print("\n===== Evolution Complete =====")
    print(f"Best fitness: {ga.best_fitness:.4f}")
    print(f"Best chromosome: {ga.best_chromosome}")
    print(f"Results saved to {final_dir}")
    print(f"Best model saved to {best_model_dir}")

if __name__ == "__main__":
    args = parse_args()
    run_evolution(args)