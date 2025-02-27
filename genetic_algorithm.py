import numpy as np
import random
from typing import List, Dict, Tuple
import json
import os

class GeneticAlgorithm:
    """
    Genetic Algorithm for evolving CNN architectures
    """
    def __init__(
        self, 
        pop_size: int = 10, 
        genome_size: int = 20, 
        mutation_prob: float = 0.1,
        tournament_size: int = 3,
        elitism: int = 2
    ):
        self.pop_size = pop_size
        self.genome_size = genome_size
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.elitism = elitism
        
        # Initialize population with random chromosomes
        self.population = self._init_population()
        
        # Dictionary to store fitness scores: chromosome -> fitness
        self.fitness_scores = {}
        
        # Keep track of best individual
        self.best_chromosome = None
        self.best_fitness = 0.0
        
        # Statistics for tracking progress
        self.generation_stats = []
    
    def _init_population(self) -> List[str]:
        """
        Initialize population with random binary chromosomes
        """
        return [''.join(random.choice('01') for _ in range(self.genome_size)) 
                for _ in range(self.pop_size)]
    
    def selection(self) -> str:
        """
        Tournament selection: Select the best individual from a random subset
        """
        # Randomly select tournament_size individuals
        tournament = random.sample(self.population, self.tournament_size)
        
        # Return the one with the best fitness
        return max(tournament, key=lambda x: self.fitness_scores.get(x, 0.0))
    
    def crossover(self, parent1: str, parent2: str) -> Tuple[str, str]:
        """
        Single-point crossover: Create two offspring by swapping portions of parents
        """
        # Choose crossover point
        xo_point = random.randint(1, self.genome_size - 1)
        
        # Create offspring
        offspring1 = parent1[:xo_point] + parent2[xo_point:]
        offspring2 = parent2[:xo_point] + parent1[xo_point:]
        
        return offspring1, offspring2
    
    def mutation(self, individual: str) -> str:
        """
        Bitwise mutation: Flip bits with probability mutation_prob
        """
        # Convert to list for easy modification
        result = list(individual)
        
        # Iterate through bits
        for i in range(len(result)):
            # Flip bit with probability mutation_prob
            if random.random() < self.mutation_prob:
                result[i] = '1' if result[i] == '0' else '0'
        
        return ''.join(result)
    
    def evolve(self) -> List[str]:
        """
        Evolve the population to create the next generation
        """
        # Sort population by fitness
        sorted_pop = sorted(
            self.population, 
            key=lambda x: self.fitness_scores.get(x, 0.0),
            reverse=True
        )
        
        # New population starts with elites (best individuals preserved unchanged)
        new_population = sorted_pop[:self.elitism]
        
        # Fill the rest of the population with offspring from selection and crossover
        while len(new_population) < self.pop_size:
            # Select parents
            parent1 = self.selection()
            parent2 = self.selection()
            
            # Create offspring through crossover
            offspring1, offspring2 = self.crossover(parent1, parent2)
            
            # Apply mutation
            offspring1 = self.mutation(offspring1)
            offspring2 = self.mutation(offspring2)
            
            # Add to new population
            new_population.append(offspring1)
            if len(new_population) < self.pop_size:
                new_population.append(offspring2)
        
        # Update population
        self.population = new_population
        
        return self.population
    
    def update_fitness(self, results: Dict[str, float]) -> None:
        """
        Update fitness scores with new evaluation results
        """
        # Update fitness dictionary
        self.fitness_scores.update(results)
        
        # Check for new best individual
        for chromosome, fitness in results.items():
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_chromosome = chromosome
    
    def get_population_jobs(self, generation: int) -> List[Dict]:
        """
        Convert population to job descriptions for the worker system
        """
        jobs = []
        
        for i, chromosome in enumerate(self.population):
            jobs.append({
                'chromosome': chromosome,
                'generation': generation,
                'individual_id': i,
                'config': {},
                'priority': 0
            })
        
        return jobs
    
    def compute_generation_stats(self) -> Dict:
        """
        Compute statistics for the current generation
        """
        # Get fitness values for current population
        fitness_values = [self.fitness_scores.get(ind, 0.0) for ind in self.population]
        
        stats = {
            'max_fitness': max(fitness_values) if fitness_values else 0.0,
            'min_fitness': min(fitness_values) if fitness_values else 0.0,
            'avg_fitness': sum(fitness_values) / len(fitness_values) if fitness_values else 0.0,
            'best_chromosome': self.best_chromosome,
            'best_fitness': self.best_fitness,
        }
        
        # Add to history
        self.generation_stats.append(stats)
        
        return stats
    
    def save_state(self, save_dir: str) -> None:
        """
        Save the current state of the GA to a file
        """
        os.makedirs(save_dir, exist_ok=True)
        
        state = {
            'population': self.population,
            'fitness_scores': self.fitness_scores,
            'best_chromosome': self.best_chromosome,
            'best_fitness': self.best_fitness,
            'generation_stats': self.generation_stats,
            'params': {
                'pop_size': self.pop_size,
                'genome_size': self.genome_size,
                'mutation_prob': self.mutation_prob,
                'tournament_size': self.tournament_size,
                'elitism': self.elitism
            }
        }
        
        with open(f"{save_dir}/ga_state.json", 'w') as f:
            json.dump(state, f, indent=2)
    
    @classmethod
    def load_state(cls, load_path: str) -> 'GeneticAlgorithm':
        """
        Load a GA instance from a saved state file
        """
        with open(load_path, 'r') as f:
            state = json.load(f)
        
        # Create instance with saved parameters
        ga = cls(
            pop_size=state['params']['pop_size'],
            genome_size=state['params']['genome_size'],
            mutation_prob=state['params']['mutation_prob'],
            tournament_size=state['params']['tournament_size'],
            elitism=state['params']['elitism']
        )
        
        # Restore state
        ga.population = state['population']
        ga.fitness_scores = state['fitness_scores']
        ga.best_chromosome = state['best_chromosome']
        ga.best_fitness = state['best_fitness']
        ga.generation_stats = state['generation_stats']
        
        return ga