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
    
    def is_empty(self):
        return self.queue.empty()
    
    def queue_size(self):
        return self.queue.qsize()