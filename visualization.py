import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json
from typing import List, Dict, Any
import seaborn as sns
from matplotlib.figure import Figure
import time


class ExperimentTracker:
    """
    Tracks experiment progress and generates visualizations
    """
    def __init__(self, output_dir="experiment_tracking"):
        self.output_dir = output_dir
        self.generations_data = []
        self.best_chromosomes = []
        self.generation_stats = []
        self.fitness_history = pd.DataFrame()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize tracking file
        self.log_file = os.path.join(output_dir, "experiment_log.csv")
        
        # Create or clear log file
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write("timestamp,generation,individual,chromosome,fitness\n")
    
    def log_individual(self, generation: int, individual_id: int, 
                      chromosome: str, fitness: float) -> None:
        """Log data for a single individual"""
        timestamp = time.time()
        with open(self.log_file, 'a') as f:
            f.write(f"{timestamp},{generation},{individual_id},{chromosome},{fitness}\n")
    
    def log_generation(self, generation: int, population: List[str], 
                      fitness_scores: Dict[str, float], stats: Dict[str, Any]) -> None:
        """Log data for an entire generation"""
        # Store generation statistics
        stats["generation"] = generation
        self.generation_stats.append(stats)
        
        # Store best chromosome
        self.best_chromosomes.append({
            "generation": generation,
            "chromosome": stats["best_chromosome"],
            "fitness": stats["best_fitness"]
        })
        
        # Create dataframe for this generation
        gen_data = []
        for i, chrom in enumerate(population):
            fitness = fitness_scores.get(chrom, 0.0)
            gen_data.append({
                "generation": generation,
                "individual": i,
                "chromosome": chrom,
                "fitness": fitness
            })
        
        # Add to fitness history
        gen_df = pd.DataFrame(gen_data)
        if self.fitness_history.empty:
            self.fitness_history = gen_df
        else:
            self.fitness_history = pd.concat([self.fitness_history, gen_df], ignore_index=True)
        
        # Save to disk
        self.fitness_history.to_csv(os.path.join(self.output_dir, "fitness_history.csv"), index=False)
        
        # Generate and save plots
        self._generate_plots(generation)
    
    def _generate_plots(self, generation: int) -> None:
        """Generate and save visualizations"""
        # 1. Fitness progression over generations
        self._plot_fitness_progression(generation)
        
        # 2. Distribution of fitness in current generation
        self._plot_fitness_distribution(generation)
        
        # 3. Heatmap of chromosomes (if not too many)
        self._plot_chromosome_heatmap(generation)
        
        # 4. Population diversity over time
        self._plot_diversity(generation)
    
    def _plot_fitness_progression(self, generation: int) -> None:
        """Plot fitness progression over generations"""
        if len(self.generation_stats) < 2:
            return  # Need at least 2 generations for a line plot
        
        fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
        
        generations = [stat["generation"] for stat in self.generation_stats]
        max_fitness = [stat["max_fitness"] for stat in self.generation_stats]
        avg_fitness = [stat["avg_fitness"] for stat in self.generation_stats]
        min_fitness = [stat["min_fitness"] for stat in self.generation_stats]
        
        ax.plot(generations, max_fitness, 'b-', marker='o', label='Max Fitness')
        ax.plot(generations, avg_fitness, 'g-', marker='s', label='Avg Fitness')
        ax.plot(generations, min_fitness, 'r-', marker='^', label='Min Fitness')
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness (Validation Accuracy)')
        ax.set_title('Fitness Evolution Over Generations')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"fitness_progression_gen{generation}.png"))
        plt.close()
    
    def _plot_fitness_distribution(self, generation: int) -> None:
        """Plot distribution of fitness in current generation"""
        # Get data for current generation
        gen_data = self.fitness_history[self.fitness_history["generation"] == generation]
        
        fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
        
        sns.histplot(gen_data["fitness"], bins=10, kde=True, ax=ax)
        
        ax.set_xlabel('Fitness (Validation Accuracy)')
        ax.set_ylabel('Count')
        ax.set_title(f'Fitness Distribution - Generation {generation}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"fitness_distribution_gen{generation}.png"))
        plt.close()
    
    def _plot_chromosome_heatmap(self, generation: int) -> None:
        """Plot heatmap of chromosomes in the generation"""
        # Get data for current generation
        gen_data = self.fitness_history[self.fitness_history["generation"] == generation]
        
        # Convert chromosomes to binary matrix
        chromosomes = gen_data["chromosome"].tolist()
        if not chromosomes:
            return
            
        # Create binary matrix
        binary_matrix = np.zeros((len(chromosomes), len(chromosomes[0])))
        for i, chrom in enumerate(chromosomes):
            for j, bit in enumerate(chrom):
                binary_matrix[i, j] = int(bit)
        
        # Sort by fitness
        fitness = gen_data["fitness"].tolist()
        sorted_indices = np.argsort(fitness)[::-1]  # descending
        binary_matrix = binary_matrix[sorted_indices]
        sorted_fitness = [fitness[i] for i in sorted_indices]
        
        fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
        
        cax = ax.imshow(binary_matrix, cmap='Blues', aspect='auto')
        
        # Add fitness values on the left
        for i, fit in enumerate(sorted_fitness):
            ax.text(-1.5, i, f"{fit:.4f}", ha='right', va='center')
        
        ax.set_title(f'Chromosome Heatmap - Generation {generation}')
        ax.set_xlabel('Bit Position')
        ax.set_ylabel('Individual (sorted by fitness)')
        
        # Add colorbar
        plt.colorbar(cax, ax=ax)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"chromosome_heatmap_gen{generation}.png"))
        plt.close()
    
    def _plot_diversity(self, generation: int) -> None:
        """Plot population diversity over time"""
        if len(self.generation_stats) < 2:
            return  # Need at least 2 generations for a line plot
        
        # Calculate diversity for each generation
        diversity = []
        for gen in range(generation + 1):
            gen_data = self.fitness_history[self.fitness_history["generation"] == gen]
            chromosomes = gen_data["chromosome"].tolist()
            
            if not chromosomes:
                diversity.append(0)
                continue
                
            # Calculate average Hamming distance between all pairs
            total_distance = 0
            count = 0
            for i in range(len(chromosomes)):
                for j in range(i+1, len(chromosomes)):
                    # Hamming distance
                    distance = sum(c1 != c2 for c1, c2 in zip(chromosomes[i], chromosomes[j]))
                    total_distance += distance
                    count += 1
            
            avg_distance = total_distance / count if count > 0 else 0
            diversity.append(avg_distance / len(chromosomes[0]))  # Normalize by chromosome length
        
        fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
        
        ax.plot(range(generation + 1), diversity, 'g-', marker='o')
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Population Diversity')
        ax.set_title('Population Diversity Over Generations')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"diversity_gen{generation}.png"))
        plt.close()
    
    def generate_final_report(self) -> None:
        """Generate comprehensive final report"""
        # Create report directory
        report_dir = os.path.join(self.output_dir, "final_report")
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate all plots with full data
        if self.generation_stats:
            final_gen = self.generation_stats[-1]["generation"]
            self._generate_plots(final_gen)
            
            # Copy final plots to report directory
            import shutil
            for plot in [
                f"fitness_progression_gen{final_gen}.png",
                f"fitness_distribution_gen{final_gen}.png",
                f"chromosome_heatmap_gen{final_gen}.png",
                f"diversity_gen{final_gen}.png"
            ]:
                src = os.path.join(self.output_dir, plot)
                dst = os.path.join(report_dir, plot.replace(f"_gen{final_gen}", "_final"))
                if os.path.exists(src):
                    shutil.copy(src, dst)
            
            # Generate best models summary
            best_df = pd.DataFrame(self.best_chromosomes)
            best_df.to_csv(os.path.join(report_dir, "best_chromosomes.csv"), index=False)
            
            # Generate generation stats summary
            stats_df = pd.DataFrame(self.generation_stats)
            stats_df.to_csv(os.path.join(report_dir, "generation_stats.csv"), index=False)
            
            # Create HTML report
            self._generate_html_report(report_dir, final_gen)
    
    def _generate_html_report(self, report_dir: str, final_gen: int) -> None:
        """Generate HTML report with all visualizations"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CNN Evolution Experiment Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .plot-container {{ margin: 20px 0; }}
                .plot {{ max-width: 100%; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
            </style>
        </head>
        <body>
            <h1>CNN Evolution Experiment Report</h1>
            <p>Final generation: {final_gen}</p>
            
            <h2>Fitness Progression</h2>
            <div class="plot-container">
                <img class="plot" src="fitness_progression_final.png" alt="Fitness Progression">
            </div>
            
            <h2>Final Fitness Distribution</h2>
            <div class="plot-container">
                <img class="plot" src="fitness_distribution_final.png" alt="Fitness Distribution">
            </div>
            
            <h2>Population Diversity</h2>
            <div class="plot-container">
                <img class="plot" src="diversity_final.png" alt="Population Diversity">
            </div>
            
            <h2>Chromosome Heatmap</h2>
            <div class="plot-container">
                <img class="plot" src="chromosome_heatmap_final.png" alt="Chromosome Heatmap">
            </div>
            
            <h2>Best Models by Generation</h2>
            <table>
                <tr>
                    <th>Generation</th>
                    <th>Fitness</th>
                    <th>Chromosome</th>
                </tr>
        """
        
        # Add best models
        for model in self.best_chromosomes:
            html += f"""
                <tr>
                    <td>{model['generation']}</td>
                    <td>{model['fitness']:.4f}</td>
                    <td>{model['chromosome']}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        with open(os.path.join(report_dir, "report.html"), "w") as f:
            f.write(html)