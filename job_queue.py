import queue
from dataclasses import dataclass
import time

@dataclass
class ModelJob:
    chromosome: str
    generation: int
    individual_id: int
    config: dict
    priority: int = 0
    # Add timestamp to ensure unique ordering when priorities are equal
    created_at: float = None
    
    def __post_init__(self):
        # Set creation timestamp if not provided
        if self.created_at is None:
            self.created_at = time.time()

class JobQueue:
    def __init__(self):
        self.queue = queue.PriorityQueue()
        self.job_counter = 0  # Counter to ensure FIFO ordering for equal priorities
    
    def add_job(self, job: ModelJob):
        # Use a tuple for comparison: (priority, creation_time, job)
        # This ensures jobs with equal priority are ordered by creation time (FIFO)
        self.queue.put((job.priority, job.created_at, self.job_counter, job))
        self.job_counter += 1
    
    def get_job(self):
        if self.queue.empty():
            return None
        # Return just the job object, not the priority or timestamps
        return self.queue.get()[3]
    
    def is_empty(self):
        return self.queue.empty()
    
    def queue_size(self):
        return self.queue.qsize()