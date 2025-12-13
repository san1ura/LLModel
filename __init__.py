"""
Main initialization file for the transformer model project
Exports key classes and functions
"""
from .model.transformer import Transformer, Config
from .tokenizer.train_tokenizer import TokenizerWrapper, TokenizerTrainer
from .data.preprocessed.build_dataset import (
    PreprocessedDataset, 
    TextProcessor, 
    DataCollator, 
    build_pretrain_dataset,
    SFTDataset
)
from .training.trainer import OptimizedTrainer as Trainer, PreTrainer, SFTTrainer, RLHFTrainer, DPOTrainer
from .serving.inference_opt.generate import InferenceEngine, ModelServer
from .evaluation.benchmarks.model_eval import ModelEvaluator, run_benchmark_suite
from .optim.schedule import (
    LionOptimizer, 
    Adafactor, 
    CosineSchedulerWithWarmup, 
    LinearWithWarmupScheduler,
    get_optimizer,
    get_scheduler
)
from .evolution.evo_loop import EvolutionEngine, SelfImprovementLoop, EvolutionConfig

# Version information
__version__ = "1.0.0"
__author__ = "Transformer Model Project"

# Export main components
__all__ = [
    # Model components
    'Transformer',
    'Config',
    
    # Tokenizer components
    'TokenizerWrapper',
    'TokenizerTrainer',
    
    # Data components
    'PreprocessedDataset',
    'TextProcessor',
    'DataCollator',
    'build_pretrain_dataset',
    'SFTDataset',
    
    # Training components
    'Trainer',
    'PreTrainer',
    'SFTTrainer',
    'RLHFTrainer',
    'DPOTrainer',
    
    # Serving components
    'InferenceEngine',
    'ModelServer',
    
    # Evaluation components
    'ModelEvaluator',
    'run_benchmark_suite',
    
    # Optimization components
    'LionOptimizer',
    'Adafactor',
    'CosineSchedulerWithWarmup',
    'LinearWithWarmupScheduler',
    'get_optimizer',
    'get_scheduler',
    
    # Evolution components
    'EvolutionEngine',
    'SelfImprovementLoop',
    'EvolutionConfig',
    
    # Version
    '__version__',
    '__author__'
]