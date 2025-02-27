import os
import subprocess
from typing import List, Dict, Tuple
import time

class GPUManager:
    def __init__(self, gpu_ids: List[int]):
        self.gpu_ids = gpu_ids
        self.available_gpus = set(gpu_ids)
        self.running_jobs = {}  # gpu_id -> job
        self.processes = {}  # gpu_id -> process
    
    def get_available_gpu(self):
        """Get an available GPU ID or None if all are busy"""
        if not self.available_gpus:
            return None
        return next(iter(self.available_gpus))
    
    def assign_job(self, job, gpu_id):
        """Assign a job to a GPU and start the training process"""
        if gpu_id not in self.available_gpus:
            raise ValueError(f"GPU {gpu_id} is not available")
            
        self.available_gpus.remove(gpu_id)
        self.running_jobs[gpu_id] = job
        
        # Create output directory if it doesn't exist
        output_dir = f"results/gen{job.generation}/ind{job.individual_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Launch process with specific GPU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        command = [
            "python", "train_model.py",
            "--chromosome", job.chromosome,
            "--gpu", str(gpu_id),
            "--generation", str(job.generation),
            "--individual_id", str(job.individual_id),
            "--output_dir", output_dir
        ]
        
        # Add any additional config parameters
        for key, value in job.config.items():
            command.extend([f"--{key}", str(value)])
        
        process = subprocess.Popen(command, env=env)
        self.processes[gpu_id] = process
        
        return process
    
    def release_gpu(self, gpu_id):
        """Release a GPU after job completion"""
        if gpu_id in self.running_jobs:
            del self.running_jobs[gpu_id]
            if gpu_id in self.processes:
                del self.processes[gpu_id]
            self.available_gpus.add(gpu_id)
    
    def check_finished_jobs(self) -> List[Tuple[int, object]]:
        """Check for completed jobs and return (gpu_id, job) pairs for finished ones"""
        finished = []
        
        for gpu_id, process in list(self.processes.items()):
            if process.poll() is not None:  # Process has finished
                job = self.running_jobs.get(gpu_id)
                if job:
                    finished.append((gpu_id, job))
        
        return finished
    
    def terminate_all(self):
        """Terminate all running processes"""
        for gpu_id, process in self.processes.items():
            if process.poll() is None:  # If still running
                process.terminate()
                
        # Wait a bit for processes to terminate
        time.sleep(1)
        
        # Force kill any remaining processes
        for gpu_id, process in self.processes.items():
            if process.poll() is None:
                process.kill()
                
    def get_running_job_count(self):
        """Return the number of currently running jobs"""
        return len(self.running_jobs)