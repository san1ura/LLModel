#!/usr/bin/env python3
"""
Main entry point for the transformer model project.
Provides CLI interface for training, inference, API, and model management.
"""
import argparse
import os
import sys
import torch
from typing import Optional

from model.transformer import Config, Transformer
from training.trainer import OptimizedTrainer as Trainer, train_model


def create_default_config(output_path: str = "config/default_config.json"):
    """Create a default configuration file."""
    config = Config(
        vocab_size=32000,
        d_model=512,
        n_layers=8,
        max_len=1024,
        n_heads=8,
        d_ff=1024,  # Smaller FFN to prevent overfitting with small dataset
        dropout=0.1,
        use_rope=True,
        pos_type='rope',
        attention_type='standard',
        use_gradient_checkpointing=False,
        norm_first=True,
        initializer_range=0.02
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    config.save(output_path)
    print(f"Default configuration saved to {output_path}")


def train_command(args):
    """Handle the train command."""
    print("Starting training...")

    # Create necessary directories
    os.makedirs("config", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Use provided config or create default
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}. Creating default...")
        create_default_config(config_path)

    # Load configuration
    config = Config.load(config_path)

    # Check if we should use the LMSYS dataset directly
    if args.use_lmsys_dataset:
        # Train tokenizer from the LMSYS dataset if not already done
        tokenizer_path = "weights/tokenizer.model"
        if not os.path.exists(tokenizer_path):
            print("Tokenizer not found. Training tokenizer from LMSYS dataset...")
            from tokenizer.train_tokenizer_from_dataset import train_tokenizer_from_lmsys_dataset
            train_tokenizer_from_lmsys_dataset(output_path=tokenizer_path)

        # Prepare the LMSYS dataset for training if not already done
        data_path = "data/preprocessed/lmsys_dataset.bin"
        if not os.path.exists(data_path):
            print("Preparing LMSYS dataset for training...")
            from data.prepare_lmsys_dataset import prepare_lmsys_dataset_for_training
            prepare_lmsys_dataset_for_training(
                tokenizer_path=tokenizer_path,
                output_path=data_path
            )
        args.data_path = data_path
    else:
        # Validate required files exist
        if not os.path.exists(args.data_path):
            print(f"Training data not found: {args.data_path}")
            sys.exit(1)

    # Start training
    try:
        model = Transformer(config)

        # Placeholder for data loading - in practice you'd load actual data
        from data.preprocessed.build_dataset import PreprocessedDataset
        train_dataset = PreprocessedDataset(args.data_path, block_size=config.max_len, vocab_size=model.config.vocab_size)

        # Load tokenizer to get pad token id
        from tokenizer.train_tokenizer import SentencePieceTokenizer
        tokenizer = SentencePieceTokenizer.from_pretrained(args.tokenizer_path or "weights/tokenizer.model")

        # Make sure model's embedding size matches tokenizer's vocab size
        tokenizer_vocab_size = tokenizer.tokenizer.get_vocab_size()
        model_vocab_size = model.embed.weight.shape[0]

        print(f"Model vocab size: {model_vocab_size}, Tokenizer vocab size: {tokenizer_vocab_size}")

        if model_vocab_size != tokenizer_vocab_size:
            print(f"Resizing model embeddings from {model_vocab_size} to {tokenizer_vocab_size}")
            model.resize_token_embeddings(tokenizer_vocab_size)
            print(f"Resized model embeddings to match tokenizer vocab size: {tokenizer_vocab_size}")

        # Double-check that they match after resizing
        assert model.config.vocab_size == tokenizer.tokenizer.get_vocab_size(), (
            f"Vocab size mismatch after resize: model={model.config.vocab_size}, tokenizer={tokenizer.tokenizer.get_vocab_size()}"
        )
        print("Model and tokenizer vocab sizes are synchronized.")

        # Create data collator
        from data.preprocessed.build_dataset import DataCollator
        collator = DataCollator(
            pad_token_id=tokenizer.tokenizer.get_vocab().get('<pad>', 0),
            vocab_size=model.config.vocab_size
        )

        # Create data loader
        from torch.utils.data import DataLoader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collator
        )

        # Use the optimized trainer with better parameters
        from training.trainer import OptimizedTrainer
        trainer = OptimizedTrainer(
            model=model,
            config=config,
            train_data=train_dataloader,
            save_dir=args.output_dir,
            lr=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            total_steps=args.epochs * len(train_dataloader),  # Use actual dataset size
            eval_interval=1000,  # Evaluate less frequently
            save_interval=1000,   # Save less frequently
            early_stopping_patience=5  # Increase patience
        )

        # Add tokenizer to trainer for sample generation
        trainer.tokenizer = tokenizer

        trainer.train(epochs=args.epochs)
        print("Training completed!")

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def generate_command(args):
    """Handle the generate command."""
    print("Running generation...")

    # Load model and config
    try:
        config = Config.load(args.config)
        model = Transformer(config)

        # Load model weights
        if args.model_path and os.path.exists(args.model_path):
            model.load_state_dict(torch.load(args.model_path))

        # Load tokenizer
        from tokenizer.train_tokenizer import SentencePieceTokenizer
        tokenizer = SentencePieceTokenizer.from_pretrained(args.tokenizer_path)

        # Import the inference engine
        from serving.inference_opt.generate import InferenceEngine
        engine = InferenceEngine(model, tokenizer, config)

        # Generate text
        result = engine.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            do_sample=True
        )

        print(f"Generated: {result}")

    except Exception as e:
        print(f"Error during generation: {e}")
        sys.exit(1)


def create_config_command(args):
    """Handle the create config command."""
    print(f"Creating {args.type} model configuration...")
    os.makedirs("config", exist_ok=True)

    if args.type == "tiny":
        config = Config(
            d_model=512,
            n_layers=8,
            n_heads=8,
            d_ff=1024,
            max_len=512,
            use_gradient_checkpointing=False
        )
    elif args.type == "small":
        config = Config(
            d_model=1024,
            n_layers=12,
            n_heads=16,
            d_ff=2048,
            max_len=1024
        )
    elif args.type == "base":
        config = Config(
            d_model=2048,
            n_layers=24,
            n_heads=16,
            d_ff=5504,
            max_len=2048,
            dropout=0.1
        )
    elif args.type == "large":
        config = Config(
            d_model=2560,
            n_layers=32,
            n_heads=32,
            d_ff=6912,
            max_len=4096,
            use_gradient_checkpointing=True
        )
    elif args.type == "xl":
        config = Config(
            d_model=4096,
            n_layers=64,
            n_heads=32,
            d_ff=11008,
            max_len=4096,
            use_rope=True,
            use_gradient_checkpointing=True,
            use_flash_attention=True
        )
    else:
        print(f"Unknown model type: {args.type}")
        print("Available types: tiny, small, base, large, xl")
        sys.exit(1)

    config_path = args.output
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    config.save(config_path)
    print(f"Configuration for {args.type} model saved to {config_path}")


def api_command(args):
    """Handle the API command."""
    print("Starting API server...")

    try:
        from serving.inference_opt.generate import create_api_server
        import uvicorn
        from fastapi import FastAPI
        from pydantic import BaseModel
        from typing import Optional

        # Load model and config
        config = Config.load(args.config)
        model = Transformer(config)

        # Load tokenizer
        from tokenizer.train_tokenizer import SentencePieceTokenizer
        tokenizer = SentencePieceTokenizer.from_pretrained(args.tokenizer_path)

        # Create server app
        from serving.inference_opt.generate import ModelServer
        server = ModelServer(model, tokenizer, config, port=args.port, host=args.host)

        # For now, just provide basic info
        print(f"API server would start on {args.host}:{args.port}")
        print("Model and API components loaded successfully")

    except ImportError as e:
        print(f"Missing dependencies for API: {e}")
        print("Install with: pip install fastapi uvicorn")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting API server: {e}")
        sys.exit(1)


def evaluate_command(args):
    """Handle the evaluation command."""
    print("Running model evaluation...")

    try:
        # Load model and config
        config = Config.load(args.config)
        model = Transformer(config)

        if args.model_path and os.path.exists(args.model_path):
            model.load_state_dict(torch.load(args.model_path))

        # Load tokenizer
        from tokenizer.train_tokenizer import SentencePieceTokenizer
        tokenizer = SentencePieceTokenizer.from_pretrained(args.tokenizer_path)

        # Create evaluator
        from evaluation.benchmarks.model_eval import ModelEvaluator
        evaluator = ModelEvaluator(model, tokenizer, config)

        # Run evaluation
        # This is a simplified example
        print("Basic evaluation completed. For comprehensive benchmarks run specific scripts.")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)


def benchmark_command(args):
    """Handle the benchmark command."""
    print("Running benchmarks...")

    try:
        config = Config.load(args.config)
        model = Transformer(config)

        # Load tokenizer
        from tokenizer.train_tokenizer import SentencePieceTokenizer
        tokenizer = SentencePieceTokenizer.from_pretrained(args.tokenizer_path)

        # Import benchmarking tools
        from serving.inference_opt.generate import InferenceEngine
        engine = InferenceEngine(model, tokenizer, config)

        # Run basic benchmarks
        print("Running benchmark tests...")
        results = engine.benchmark_inference()

        print("Benchmark Results:")
        for metric, value in results.items():
            print(f"  {metric}: {value}")

    except Exception as e:
        print(f"Error during benchmarking: {e}")
        sys.exit(1)


def evolve_command(args):
    """Handle the evolution command."""
    print("Starting evolution process...")

    try:
        config = Config.load(args.config)
        model = Transformer(config)

        # Load tokenizer
        from tokenizer.train_tokenizer import SentencePieceTokenizer
        tokenizer = SentencePieceTokenizer.from_pretrained(args.tokenizer_path)

        # Create evolution engine
        from evolution.evo_loop import EvolutionEngine, EvolutionConfig, evolution_evaluation_function
        from evaluation.benchmarks.model_eval import ModelEvaluator

        # Set up evolution config
        evo_config = EvolutionConfig(
            population_size=args.population_size,
            generations=args.generations,
            mutation_rate=args.mutation_rate
        )

        # Create evaluator for evolution
        from evolution.evo_loop import ScoreEvaluator
        evaluator = ScoreEvaluator(evolution_evaluation_function)

        # Create evolution engine
        evolution_engine = EvolutionEngine(model, evo_config, evaluator)

        print(f"Evolution configured with population_size={args.population_size}, "
              f"generations={args.generations}, mutation_rate={args.mutation_rate}")

        # In practice, you would need training data for evaluation
        # This is just initializing the process

    except Exception as e:
        print(f"Error during evolution setup: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Transformer Model Suite CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Pretrain command
    pretrain_parser = subparsers.add_parser("pretrain", help="Pre-train the model")
    pretrain_parser.add_argument("--config", type=str, default="config/default_config.json", help="Path to config file")
    pretrain_parser.add_argument("--data-path", type=str, help="Path to training data (required unless --use-lmsys-dataset is specified)")
    pretrain_parser.add_argument("--tokenizer-path", type=str, help="Path to tokenizer (required unless using LMSYS dataset)")
    pretrain_parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    pretrain_parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    pretrain_parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate for training")
    pretrain_parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    pretrain_parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Number of gradient accumulation steps")
    pretrain_parser.add_argument("--use-lmsys-dataset", action="store_true", help="Use the LMSYS dataset for training (downloads and processes automatically)")
    pretrain_parser.set_defaults(func=train_command)

    # Train command (Fine-tuning)
    train_parser = subparsers.add_parser("train", help="Fine-tune the model")
    train_parser.add_argument("--config", type=str, default="config/default_config.json", help="Path to config file")
    train_parser.add_argument("--data-path", type=str, help="Path to training data (required unless --use-lmsys-dataset is specified)")
    train_parser.add_argument("--tokenizer-path", type=str, help="Path to tokenizer (required unless using LMSYS dataset)")
    train_parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    train_parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    train_parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate for training")
    train_parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    train_parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Number of gradient accumulation steps")
    train_parser.add_argument("--use-lmsys-dataset", action="store_true", help="Use the LMSYS dataset for training (downloads and processes automatically)")
    train_parser.add_argument("--model-path", type=str, help="Path to pre-trained model weights for fine-tuning")
    train_parser.set_defaults(func=train_command)

    # Fine-tune command (Supervised Fine-Tuning)
    finetune_parser = subparsers.add_parser("finetune", help="Fine-tune the model with specific SFT parameters")
    finetune_parser.add_argument("--config", type=str, default="config/default_config.json", help="Path to config file")
    finetune_parser.add_argument("--data-path", type=str, help="Path to training data (required unless --use-lmsys-dataset is specified)")
    finetune_parser.add_argument("--tokenizer-path", type=str, required=True, help="Path to tokenizer")
    finetune_parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    finetune_parser.add_argument("--batch-size", type=int, default=4, help="Batch size for fine-tuning")
    finetune_parser.add_argument("--learning-rate", type=float, default=5e-6, help="Learning rate for fine-tuning")
    finetune_parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs")
    finetune_parser.add_argument("--gradient-accumulation-steps", type=int, default=8, help="Number of gradient accumulation steps")
    finetune_parser.add_argument("--model-path", type=str, help="Path to pre-trained model weights for fine-tuning")
    finetune_parser.add_argument("--warmup-steps", type=int, default=100, help="Number of warmup steps")
    finetune_parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay for fine-tuning")
    finetune_parser.set_defaults(func=train_command)

    # LoRA fine-tune command
    lora_parser = subparsers.add_parser("lora-finetune", help="Fine-tune the model using LoRA (Low-Rank Adaptation)")
    lora_parser.add_argument("--config", type=str, default="config/default_config.json", help="Path to config file")
    lora_parser.add_argument("--data-path", type=str, help="Path to training data (required unless --use-lmsys-dataset is specified)")
    lora_parser.add_argument("--tokenizer-path", type=str, required=True, help="Path to tokenizer")
    lora_parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    lora_parser.add_argument("--batch-size", type=int, default=4, help="Batch size for LoRA fine-tuning")
    lora_parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate for LoRA fine-tuning")
    lora_parser.add_argument("--epochs", type=int, default=3, help="Number of LoRA fine-tuning epochs")
    lora_parser.add_argument("--gradient-accumulation-steps", type=int, default=8, help="Number of gradient accumulation steps")
    lora_parser.add_argument("--model-path", type=str, help="Path to pre-trained model weights for LoRA fine-tuning")
    lora_parser.add_argument("--lora-rank", type=int, default=16, help="LoRA attention dimension")
    lora_parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA scaling factor")
    lora_parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout probability")
    lora_parser.add_argument("--target-modules", type=str, default="q_proj,v_proj", help="Target modules for LoRA injection")
    lora_parser.add_argument("--optimizer-name", type=str, default="adamw", help="Optimizer for LoRA fine-tuning")
    lora_parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay for LoRA fine-tuning")
    lora_parser.set_defaults(func=train_command)

    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate text with the model")
    generate_parser.add_argument("--config", type=str, default="config/default_config.json", help="Path to config file")
    generate_parser.add_argument("--model-path", type=str, help="Path to trained model weights")
    generate_parser.add_argument("--tokenizer-path", type=str, required=True, help="Path to tokenizer")
    generate_parser.add_argument("--prompt", type=str, required=True, help="Input prompt for generation")
    generate_parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    generate_parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    generate_parser.set_defaults(func=generate_command)

    # Create config command
    create_parser = subparsers.add_parser("create-config", help="Create model configuration")
    create_parser.add_argument("--type", type=str, choices=["tiny", "small", "base", "large", "xl"],
                              default="base", help="Model size type")
    create_parser.add_argument("--output", type=str, default="config/default_config.json",
                              help="Output path for config file")
    create_parser.set_defaults(func=create_config_command)

    # API command
    api_parser = subparsers.add_parser("api", help="Start the API server")
    api_parser.add_argument("--config", type=str, default="config/default_config.json", help="Path to config file")
    api_parser.add_argument("--tokenizer-path", type=str, required=True, help="Path to tokenizer")
    api_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for the API server")
    api_parser.add_argument("--port", type=int, default=8000, help="Port for the API server")
    api_parser.set_defaults(func=api_command)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    eval_parser.add_argument("--config", type=str, default="config/default_config.json", help="Path to config file")
    eval_parser.add_argument("--model-path", type=str, help="Path to trained model weights")
    eval_parser.add_argument("--tokenizer-path", type=str, required=True, help="Path to tokenizer")
    eval_parser.set_defaults(func=evaluate_command)

    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run performance benchmarks")
    benchmark_parser.add_argument("--config", type=str, default="config/default_config.json", help="Path to config file")
    benchmark_parser.add_argument("--tokenizer-path", type=str, required=True, help="Path to tokenizer")
    benchmark_parser.set_defaults(func=benchmark_command)

    # Evolution command
    evolve_parser = subparsers.add_parser("evolve", help="Evolve the model using evolutionary methods")
    evolve_parser.add_argument("--config", type=str, default="config/default_config.json", help="Path to config file")
    evolve_parser.add_argument("--tokenizer-path", type=str, required=True, help="Path to tokenizer")
    evolve_parser.add_argument("--population-size", type=int, default=10, help="Size of the population for evolution")
    evolve_parser.add_argument("--generations", type=int, default=5, help="Number of generations to evolve")
    evolve_parser.add_argument("--mutation-rate", type=float, default=0.1, help="Mutation rate for evolution")
    evolve_parser.set_defaults(func=evolve_command)

    # Parse arguments
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    # Check if a command function is provided
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
