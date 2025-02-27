import os
import subprocess
import time
import json
import psutil
import torch
from typing import List, Dict, Tuple, Set, Optional

class GPUMemoryTracker:
    """Track GPU memory usage and estimate available memory"""
    
    def __init__(self, gpu_ids: List[int], reserved_memory_mb: int = 1000):
        """
        Initialize memory tracker
        
        Args:
            gpu_ids: List of GPU IDs to track
            reserved_memory_mb: Amount of memory to reserve as buffer (in MB)
        """
        self.gpu_ids = gpu_ids
        self.reserved_memory_mb = reserved_memory_mb
        self.memory_per_model = {}  # Map chromosome -> estimated memory
        
        # Check if CUDA is available
        self.cuda_available = torch.cuda.is_available()
        
        # Initialize memory usage counters
        self.reset_memory_usage()
    
    def reset_memory_usage(self):
        """Reset tracked memory usage for all GPUs"""
        self.memory_usage = {gpu_id: 0 for gpu_id in self.gpu_ids}
    
    def estimate_model_memory(self, chromosome: str, num_classes: int = 10) -> int:
        """
        Estimate memory required for a model based on chromosome
        
        Args:
            chromosome: Binary chromosome string
            num_classes: Number of classes (10 for CIFAR-10, 100 for CIFAR-100)
            
        Returns:
            Estimated memory in MB
        """
        if chromosome in self.memory_per_model:
            return self.memory_per_model[chromosome]
        
        # Parse chromosome to get model complexity
        out_ch_idx = int(chromosome[0:2], 2)
        kernel_idx = int(chromosome[3:5], 2)
        num_layers = [2, 3, 4, 5][out_ch_idx]  # Estimate number of layers
        
        # Base memory estimate (MB)
        base_memory = 500  # Base model + overhead
        
        # Memory for layers (scales with number of filters and layers)
        filter_sizes = [16, 32, 64, 128]
        filter_base = filter_sizes[out_ch_idx]
        layer_memory = 0
        
        for i in range(num_layers):
            # Estimate filters at this layer
            filters = filter_base * (2 ** i)
            
            # Memory scales approximately with square of kernel size and number of filters
            kernel_sizes = [3, 5, 7, 9]
            kernel_size = kernel_sizes[kernel_idx]
            
            # Memory per layer increases with layer depth (more filters)
            layer_memory += filters * (kernel_size ** 2) * 4 / (1024 * 1024)  # Convert to MB
        
        # Additional memory for batch size, optimizer states, gradients
        batch_size_memory = 128 * 32 * 32 * 3 * 4 / (1024 * 1024)  # 128 batch size, 32x32x3 images, float32
        
        # Classifier memory increases with number of classes
        classifier_memory = num_classes * 50  # Rough estimate
        
        total_memory = base_memory + layer_memory + batch_size_memory + classifier_memory
        
        # Add buffer for unexpected memory spikes
        total_memory *= 1.2
        
        # Cache the estimate
        self.memory_per_model[chromosome] = int(total_memory)
        
        return int(total_memory)
    
    def get_available_memory(self, gpu_id: int) -> int:
        """
        Get available memory on GPU in MB
        
        Args:
            gpu_id: GPU ID to check
            
        Returns:
            Available memory in MB
        """
        if not self.cuda_available:
            # If CUDA not available, return a default value for testing
            return 8000 - self.memory_usage.get(gpu_id, 0)
        
        try:
            # Get actual free memory from CUDA
            torch.cuda.empty_cache()  # Clear cache first
            free_memory = torch.cuda.mem_get_info(gpu_id)[0] / (1024 * 1024)  # Convert to MB
            
            # Subtract tracked usage and reserved memory
            available = free_memory - self.reserved_memory_mb - self.memory_usage.get(gpu_id, 0)
            return max(0, int(available))
        except Exception as e:
            print(f"Error getting GPU memory: {e}")
            # Fallback to a default value
            return 4000 - self.memory_usage.get(gpu_id, 0)
    
    def reserve_memory(self, gpu_id: int, chromosome: str, num_classes: int = 10) -> bool:
        """
        Reserve memory for a model on GPU
        
        Args:
            gpu_id: GPU ID to reserve memory on
            chromosome: Model chromosome
            num_classes: Number of classes
            
        Returns:
            True if memory was successfully reserved, False otherwise
        """
        estimated_memory = self.estimate_model_memory(chromosome, num_classes)
        available_memory = self.get_available_memory(gpu_id)
        
        if estimated_memory <= available_memory:
            # Reserve memory
            self.memory_usage[gpu_id] = self.memory_usage.get(gpu_id, 0) + estimated_memory
            return True
        
        return False
    
    def release_memory(self, gpu_id: int, chromosome: str, num_classes: int = 10) -> None:
        """
        Release memory reserved for a model
        
        Args:
            gpu_id: GPU ID to release memory from
            chromosome: Model chromosome
            num_classes: Number of classes
        """
        estimated_memory = self.estimate_model_memory(chromosome, num_classes)
        self.memory_usage[gpu_id] = max(0, self.memory_usage.get(gpu_id, 0) - estimated_memory)


class MultiJobGPU:
    """Manages multiple jobs on a single GPU"""
    
    def __init__(self, gpu_id: int, memory_tracker: GPUMemoryTracker):
        self.gpu_id = gpu_id
        self.memory_tracker = memory_tracker
        self.running_jobs = {}  # job_id -> job
        self.processes = {}  # job_id -> process
        self.next_job_id = 0
    
    def can_fit_job(self, job) -> bool:
        """Check if a job can fit on this GPU"""
        num_classes = 100 if job.config.get("dataset") == "cifar100" else 10
        return self.memory_tracker.reserve_memory(self.gpu_id, job.chromosome, num_classes)
    
    def assign_job(self, job) -> bool:
        """
        Assign a job to this GPU and start the training process
        
        Returns:
            True if job was successfully assigned, False otherwise
        """
        # Check if we can fit the job
        num_classes = 100 if job.config.get("dataset") == "cifar100" else 10
        if not self.memory_tracker.reserve_memory(self.gpu_id, job.chromosome, num_classes):
            return False
        
        job_id = self.next_job_id
        self.next_job_id += 1
        
        # Create output directory if it doesn't exist
        output_dir = f"results/gen{job.generation}/ind{job.individual_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Launch process with specific GPU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        
        command = [
            "python", "train_model.py",
            "--chromosome", job.chromosome,
            "--gpu", "0",  # Use GPU 0 since we're setting CUDA_VISIBLE_DEVICES
            "--generation", str(job.generation),
            "--individual_id", str(job.individual_id),
            "--output_dir", output_dir
        ]
        
        # Add any additional config parameters
        for key, value in job.config.items():
            command.extend([f"--{key}", str(value)])
        
        process = subprocess.Popen(command, env=env)
        
        # Store job and process
        self.running_jobs[job_id] = job
        self.processes[job_id] = process
        
        return True
    
    def check_finished_jobs(self) -> List[object]:
        """Check for completed jobs and return finished ones"""
        finished_jobs = []
        
        for job_id, process in list(self.processes.items()):
            if process.poll() is not None:  # Process has finished
                job = self.running_jobs.get(job_id)
                if job:
                    # Release memory
                    num_classes = 100 if job.config.get("dataset") == "cifar100" else 10
                    self.memory_tracker.release_memory(self.gpu_id, job.chromosome, num_classes)
                    
                    # Add to finished jobs
                    finished_jobs.append(job)
                    
                    # Remove from tracking
                    del self.running_jobs[job_id]
                    del self.processes[job_id]
        
        return finished_jobs
    
    def get_running_job_count(self) -> int:
        """Get number of jobs running on this GPU"""
        return len(self.running_jobs)
    
    def terminate_all(self) -> None:
        """Terminate all running processes"""
        for job_id, process in self.processes.items():
            if process.poll() is None:  # If still running
                process.terminate()
        
        # Wait a bit for processes to terminate
        time.sleep(1)
        
        # Force kill any remaining processes
        for job_id, process in list(self.processes.items()):
            if process.poll() is None:
                process.kill()
        
        # Reset
        self.running_jobs = {}
        self.processes = {}
        self.memory_tracker.reset_memory_usage()


class GPUManager:
    """Manages multiple GPUs, each capable of running multiple jobs"""
    
    def __init__(self, gpu_ids: List[int], reserved_memory_mb: int = 1000):
        self.gpu_ids = gpu_ids
        self.memory_tracker = GPUMemoryTracker(gpu_ids, reserved_memory_mb)
        self.gpus = {gpu_id: MultiJobGPU(gpu_id, self.memory_tracker) for gpu_id in gpu_ids}
    
    def get_available_gpu_for_job(self, job) -> Optional[int]:
        """
        Find a GPU that can run the given job
        
        Args:
            job: Job to run
            
        Returns:
            GPU ID that can run the job, or None if no GPU has enough memory
        """
        # First try GPUs with fewest jobs
        gpus_by_load = sorted(self.gpu_ids, key=lambda gpu_id: self.gpus[gpu_id].get_running_job_count())
        
        for gpu_id in gpus_by_load:
            if self.gpus[gpu_id].can_fit_job(job):
                return gpu_id
        
        return None
    
    def assign_job(self, job) -> bool:
        """
        Assign a job to an available GPU
        
        Args:
            job: Job to assign
            
        Returns:
            True if job was successfully assigned, False otherwise
        """
        gpu_id = self.get_available_gpu_for_job(job)
        if gpu_id is None:
            return False
        
        return self.gpus[gpu_id].assign_job(job)
    
    def check_finished_jobs(self) -> List[Tuple[int, object]]:
        """Check for completed jobs across all GPUs"""
        finished = []
        
        for gpu_id, gpu in self.gpus.items():
            finished_jobs = gpu.check_finished_jobs()
            for job in finished_jobs:
                finished.append((gpu_id, job))
        
        return finished
    
    def terminate_all(self) -> None:
        """Terminate all running processes on all GPUs"""
        for gpu in self.gpus.values():
            gpu.terminate_all()
    
    def get_running_job_count(self) -> int:
        """Get total number of running jobs across all GPUs"""
        return sum(gpu.get_running_job_count() for gpu in self.gpus.values())
    
    def get_gpu_status(self) -> Dict:
        """Get status information for all GPUs"""
        status = {}
        
        for gpu_id in self.gpu_ids:
            gpu = self.gpus[gpu_id]
            available_memory = self.memory_tracker.get_available_memory(gpu_id)
            used_memory = self.memory_tracker.memory_usage.get(gpu_id, 0)
            
            status[gpu_id] = {
                "running_jobs": gpu.get_running_job_count(),
                "available_memory_mb": available_memory,
                "used_memory_mb": used_memory
            }
        
        return status