"""
Utils for training modules
Contains helper functions for training loop management
"""
import os
import torch
import random
import numpy as np
from typing import Optional
import logging
import json
import time
from torch.utils.data import DataLoader


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across PyTorch, NumPy, and random modules
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_training_state(model, optimizer, epoch, step, loss, filepath: str):
    """
    Save the training state including model weights, optimizer state, and epoch/step info
    """
    state = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': time.time()
    }
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)


def load_training_state(model, optimizer, filepath: str):
    """
    Load the training state from a saved checkpoint
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    state = torch.load(filepath)
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    
    return state['epoch'], state['step'], state['loss']


def setup_logging(log_dir: str = "logs", log_level: int = logging.INFO):
    """
    Setup logging configuration for training
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Create file handler
    log_file = os.path.join(log_dir, f"training_{time.strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def format_seconds(seconds: float) -> str:
    """
    Format seconds into human-readable format (HH:MM:SS)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


class EarlyStopping:
    """
    Early stopping mechanism to prevent overfitting
    """
    def __init__(self, patience: int = 7, min_delta: float = 0.0, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_loss: float, model) -> bool:
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        
        return self.early_stop
    
    def save_checkpoint(self, val_loss: float, model):
        """
        Save model checkpoint when improvement is observed
        """
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            'val_loss': val_loss,
            'model_state_dict': model.state_dict()
        }, "checkpoints/best_model.pth")


def get_gpu_memory_usage():
    """
    Get current GPU memory usage if available
    """
    if not torch.cuda.is_available():
        return {"gpu_available": False}
    
    gpu_info = {}
    for i in range(torch.cuda.device_count()):
        gpu_info[f"gpu_{i}_memory_allocated"] = torch.cuda.memory_allocated(i) / 1024**3  # GB
        gpu_info[f"gpu_{i}_memory_reserved"] = torch.cuda.memory_reserved(i) / 1024**3   # GB
        gpu_info[f"gpu_{i}_memory_max"] = torch.cuda.max_memory_allocated(i) / 1024**3   # GB
    
    return {
        "gpu_available": True,
        **gpu_info
    }


def estimate_training_time(steps_per_epoch: int, current_step: int, total_steps: int, 
                          time_per_step: float) -> dict:
    """
    Estimate remaining training time
    """
    remaining_steps = total_steps - current_step
    remaining_epochs = remaining_steps / steps_per_epoch
    
    remaining_time_seconds = remaining_steps * time_per_step
    
    return {
        "steps_remaining": remaining_steps,
        "epochs_remaining": remaining_epochs,
        "time_remaining_seconds": remaining_time_seconds,
        "time_remaining_formatted": format_seconds(remaining_time_seconds),
        "estimated_finish_time": time.time() + remaining_time_seconds
    }


class GradientAccumulator:
    """
    Helper class to manage gradient accumulation
    """
    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
    
    def should_update(self) -> bool:
        """
        Check if parameters should be updated
        """
        return (self.current_step + 1) % self.accumulation_steps == 0
    
    def increment_step(self):
        """
        Increment the internal step counter
        """
        self.current_step += 1
    
    def reset_step(self):
        """
        Reset the internal step counter
        """
        self.current_step = 0
    
    def adjust_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Adjust loss by accumulation steps
        """
        return loss / self.accumulation_steps


def count_parameters(model) -> dict:
    """
    Count the number of parameters in a model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params,
        "percentage_trainable": 100 * trainable_params / total_params if total_params > 0 else 0
    }


def print_model_summary(model):
    """
    Print a detailed summary of the model
    """
    param_counts = count_parameters(model)
    
    print("="*60)
    print("MODEL SUMMARY")
    print("="*60)
    print(f"Total Parameters:     {param_counts['total_parameters']:,}")
    print(f"Trainable Parameters: {param_counts['trainable_parameters']:,}")
    print(f"Non-trainable:        {param_counts['non_trainable_parameters']:,}")
    print(f"% Trainable:          {param_counts['percentage_trainable']:.2f}%")
    print("="*60)


def create_experiment_dir(base_dir: str = "experiments", name: str = None) -> str:
    """
    Create a unique experiment directory
    """
    if name is None:
        name = time.strftime("%Y%m%d_%H%M%S")
    
    exp_dir = os.path.join(base_dir, name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create subdirectories
    for subdir in ["checkpoints", "logs", "configs", "results"]:
        os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)
    
    return exp_dir


def log_hyperparameters(hyperparams: dict, filepath: str = "hyperparams.json"):
    """
    Save hyperparameters to a JSON file
    """
    with open(filepath, 'w') as f:
        json.dump(hyperparams, f, indent=2)