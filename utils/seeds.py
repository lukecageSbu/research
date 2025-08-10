"""
Reproducibility utilities for setting random seeds.
"""

import torch
import numpy as np
import random
import os
from typing import Optional


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducible experiments.
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms (slower but fully reproducible)
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if deterministic:
        # Use deterministic algorithms (can be slower)
        torch.use_deterministic_algorithms(True)
        
        # Additional settings for full determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set environment variable for deterministic behavior
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    else:
        # Allow non-deterministic algorithms for better performance
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_reproducible_generator(seed: Optional[int] = None) -> torch.Generator:
    """
    Create a PyTorch generator with a specific seed for reproducible random operations.
    
    Args:
        seed: Random seed (if None, uses a default seed)
        
    Returns:
        generator: PyTorch generator with set seed
    """
    if seed is None:
        seed = 42  # Default seed
    
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def create_worker_init_fn(base_seed: int = 0):
    """
    Create a worker initialization function for DataLoader reproducibility.
    
    Args:
        base_seed: Base seed value
        
    Returns:
        worker_init_fn: Function to initialize worker processes with unique seeds
    """
    def worker_init_fn(worker_id: int) -> None:
        # Each worker gets a unique seed based on base_seed and worker_id
        worker_seed = base_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    return worker_init_fn


class SeedContext:
    """
    Context manager for temporarily setting a specific seed.
    """
    
    def __init__(self, seed: int, deterministic: bool = False):
        """
        Initialize seed context.
        
        Args:
            seed: Temporary seed to use
            deterministic: Whether to use deterministic algorithms
        """
        self.seed = seed
        self.deterministic = deterministic
        
        # Store current state
        self.python_state = None
        self.numpy_state = None
        self.torch_state = None
        self.torch_cuda_state = None
        self.cudnn_deterministic = None
        self.cudnn_benchmark = None
    
    def __enter__(self):
        # Save current random states
        self.python_state = random.getstate()
        self.numpy_state = np.random.get_state()
        self.torch_state = torch.get_rng_state()
        if torch.cuda.is_available():
            self.torch_cuda_state = torch.cuda.get_rng_state_all()
        
        # Save cuDNN settings
        self.cudnn_deterministic = torch.backends.cudnn.deterministic
        self.cudnn_benchmark = torch.backends.cudnn.benchmark
        
        # Set new seed
        set_seed(self.seed, self.deterministic)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous random states
        random.setstate(self.python_state)
        np.random.set_state(self.numpy_state)
        torch.set_rng_state(self.torch_state)
        if torch.cuda.is_available() and self.torch_cuda_state is not None:
            torch.cuda.set_rng_state_all(self.torch_cuda_state)
        
        # Restore cuDNN settings
        torch.backends.cudnn.deterministic = self.cudnn_deterministic
        torch.backends.cudnn.benchmark = self.cudnn_benchmark


def check_reproducibility(model: torch.nn.Module, input_data: torch.Tensor, 
                         seed: int = 42, num_runs: int = 3) -> bool:
    """
    Check if model outputs are reproducible across multiple runs.
    
    Args:
        model: PyTorch model to test
        input_data: Sample input data
        seed: Seed to use for testing
        num_runs: Number of runs to compare
        
    Returns:
        is_reproducible: True if outputs are identical across runs
    """
    model.eval()
    outputs = []
    
    for _ in range(num_runs):
        with SeedContext(seed, deterministic=True):
            with torch.no_grad():
                output = model(input_data)
                outputs.append(output.clone())
    
    # Check if all outputs are identical
    for i in range(1, num_runs):
        if not torch.allclose(outputs[0], outputs[i], atol=1e-8):
            return False
    
    return True


# Common seeds for different components
DEFAULT_SEEDS = {
    'global': 42,
    'data_loading': 123,
    'model_init': 456,
    'training': 789,
    'evaluation': 101112
}


def get_seed(component: str) -> int:
    """
    Get a predefined seed for a specific component.
    
    Args:
        component: Component name ('global', 'data_loading', 'model_init', 'training', 'evaluation')
        
    Returns:
        seed: Seed value for the component
    """
    return DEFAULT_SEEDS.get(component, DEFAULT_SEEDS['global']) 