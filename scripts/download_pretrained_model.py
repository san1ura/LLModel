#!/usr/bin/env python3
"""
Script to download and handle pre-trained models for the transformer model suite
"""
import os
import argparse
import torch
from huggingface_hub import snapshot_download
from model.transformer import Config, Transformer


def download_pretrained_model(repo_id: str, local_dir: str):
    """
    Download a pre-trained model from Hugging Face Hub
    
    Args:
        repo_id: Repository ID in format 'username/model-name'
        local_dir: Local directory to save the model
    """
    print(f"Downloading model from {repo_id}...")
    
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir
    )
    
    print(f"Model downloaded to {local_dir}")


def load_model(model_path: str, config_path: str):
    """
    Load a model from local checkpoint
    
    Args:
        model_path: Path to the model weights file
        config_path: Path to the configuration file
    
    Returns:
        model: Loaded model
        config: Loaded configuration
    """
    print(f"Loading model from {model_path}...")
    
    # Load configuration
    config = Config.load(config_path)
    
    # Initialize model with the configuration
    model = Transformer(config)
    
    # Load the weights
    checkpoint = torch.load(model_path, map_location='cpu')  # Load to CPU by default
    
    # Handle both direct state dict and checkpoint dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"Model loaded successfully with config: {config}")
    
    return model, config


def save_model(model, config, output_path: str):
    """
    Save a model and its configuration
    
    Args:
        model: Model to save
        config: Configuration to save
        output_path: Path where to save the model
    """
    print(f"Saving model to {output_path}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save model state dict
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__
    }, output_path)
    
    # Save config separately
    config_path = os.path.join(os.path.dirname(output_path), 'config.json')
    config.save(config_path)
    
    print(f"Model saved to {output_path} and config to {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Download and manage pre-trained models")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download a pre-trained model from Hugging Face Hub")
    download_parser.add_argument("--repo-id", type=str, required=True, 
                                help="Repository ID (e.g. username/model-name)")
    download_parser.add_argument("--local-dir", type=str, default="checkpoints/downloaded",
                                help="Local directory to save the model")
    
    # Load command
    load_parser = subparsers.add_parser("load", help="Load a pre-trained model from local path")
    load_parser.add_argument("--model-path", type=str, required=True,
                            help="Path to the model weights file")
    load_parser.add_argument("--config-path", type=str, required=True,
                            help="Path to the configuration file")
    
    # Save command
    save_parser = subparsers.add_parser("save", help="Save a model to local path")
    save_parser.add_argument("--model-path", type=str, required=True,
                            help="Path to save the model weights")
    save_parser.add_argument("--config-path", type=str, required=True,
                            help="Path to save the configuration")
    
    args = parser.parse_args()
    
    if args.command == "download":
        download_pretrained_model(args.repo_id, args.local_dir)
    elif args.command == "load":
        model, config = load_model(args.model_path, args.config_path)
        print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    elif args.command == "save":
        print("This command would save a model, but requires loading a model in the script first.")
        print("Please modify the script to load your model before saving.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()