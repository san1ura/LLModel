"""
Flexible script to train the model with the LMSYS dataset
with better progress tracking and performance tuning
"""

import argparse
import os
import subprocess
import sys


def get_default_config_path():
    """Get the default config path"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "config", "small_config.json")


def confirm_params(args):
    """Display parameters and ask for confirmation"""
    print("Training parameters:")
    print(f"  Config: {args.config}")
    print(f"  Batch size: {args.bs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Gradient accumulation steps: {args.grad_accum}")

    response = input("Do you want to proceed with these parameters? (y/N): ")
    if response.lower() == "y":
        return True
    elif response == "":
        print("Selection is mandatory. Exiting.")
        sys.exit(1)
    else:
        print("Training cancelled.")
        sys.exit(0)


def train_with_lmsys_dataset(args):
    """
    Train the model using the LMSYS dataset
    """
    print("Training model with LMSYS dataset...")

    # Add the project root to Python path to ensure modules can be found
    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )  # Go up two levels to project root
    sys.path.insert(0, project_root)

    # Create the weights directory if it doesn't exist
    os.makedirs("weights", exist_ok=True)

    # Store original arguments
    original_argv = sys.argv

    try:
        # Prepare arguments as if they came from command line
        sys.argv = [
            "main.py",
            "train",
            "--use-lmsys-dataset",
            "--config",
            args.config,
            "--tokenizer-path",
            "weights/tokenizer.model",  # Add tokenizer path
            "--output-dir",
            "checkpoints",
            "--batch-size",
            str(args.bs),
            "--learning-rate",
            str(args.lr),
            "--epochs",
            str(args.epochs),
            "--gradient-accumulation-steps",
            str(args.grad_accum),
        ]

        # Import and run main function directly - but we need to reset sys.argv for the argument parser
        # Instead of calling main.main() which parses sys.argv,
        # we'll call the train command directly with prepared args
        import argparse

        import main

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")

        # Add train subparser (simplified version)
        train_parser = subparsers.add_parser("train")
        train_parser.add_argument(
            "--config", type=str, default="config/default_config.json"
        )
        train_parser.add_argument("--data-path", type=str)
        train_parser.add_argument("--tokenizer-path", type=str)
        train_parser.add_argument("--output-dir", type=str, default="checkpoints")
        train_parser.add_argument("--batch-size", type=int, default=4)
        train_parser.add_argument("--learning-rate", type=float, default=5e-5)
        train_parser.add_argument("--epochs", type=int, default=3)
        train_parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
        train_parser.add_argument("--use-lmsys-dataset", action="store_true")

        # Parse the arguments we prepared
        parsed_args = parser.parse_args(
            [
                "train",
                "--use-lmsys-dataset",
                "--config",
                args.config,
                "--tokenizer-path",
                "weights/tokenizer.model",
                "--output-dir",
                "checkpoints",
                "--batch-size",
                str(args.bs),
                "--learning-rate",
                str(args.lr),
                "--epochs",
                str(args.epochs),
                "--gradient-accumulation-steps",
                str(args.grad_accum),
            ]
        )

        # Set the func attribute to train_command
        import types

        parsed_args.func = main.train_command
        main.train_command(parsed_args)
        print("Training completed successfully!")

    except KeyboardInterrupt:
        print("Training was interrupted by user.")
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        # Restore original arguments
        sys.argv = original_argv


def train_tokenizer_first():
    """
    Train the tokenizer from the LMSYS dataset
    """
    print("Training tokenizer from LMSYS dataset...")

    import subprocess

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    cmd = [
        sys.executable,
        os.path.join(project_root, "tokenizer", "train_tokenizer_from_dataset.py"),
    ]

    print(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        if result.returncode != 0:
            raise Exception(
                f"Tokenizer training failed with exit code {result.returncode}"
            )
        print("Tokenizer training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Tokenizer training failed with error: {e}")
        sys.exit(1)


def check_config_exists(config_path):
    """Check if config file exists, if not check for alternatives"""
    if os.path.exists(config_path):
        return config_path

    # Check if small_config exists in the same directory as the default
    alt_config = os.path.join(os.path.dirname(config_path), "small_config.json")
    if os.path.exists(alt_config):
        return alt_config

    # Check if tiny_config exists
    alt_config = os.path.join(os.path.dirname(config_path), "tiny_config.json")
    if os.path.exists(alt_config):
        return alt_config

    # Check if default_config exists
    alt_config = os.path.join(os.path.dirname(config_path), "default_config.json")
    if os.path.exists(alt_config):
        return alt_config

    # If none exist, create a default config
    print(f"Config file {config_path} not found. Creating default config.")
    create_default_config = os.path.join(
        os.path.dirname(config_path), "default_config.json"
    )

    # Add project root to path to import main module
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    from main import create_default_config as create_config_func

    create_config_func(create_default_config)
    return create_default_config


def main():
    # Add the project root to Python path to ensure modules can be found
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    parser = argparse.ArgumentParser(description="Train the model with LMSYS dataset")
    parser.add_argument(
        "--config",
        type=str,
        default=get_default_config_path(),
        help="Model configuration file",
    )
    parser.add_argument(
        "--bs", "--batch-size", type=int, default=2, help="Batch size for training"
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs for training"
    )
    parser.add_argument(
        "--grad-accum",
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--tokenizer-first", action="store_true", help="Train tokenizer before training"
    )

    # Check if any arguments are provided
    args_provided = len(sys.argv) > 1

    # Parse arguments
    args = parser.parse_args()

    # If no arguments were provided, show defaults and ask for confirmation
    if not args_provided:
        args.config = check_config_exists(args.config)
        confirm_params(args)
    else:
        args.config = check_config_exists(args.config)

    print("Starting the process to train model with LMSYS dataset")
    print("Note: This process will use tqdm for progress tracking and detailed logging")

    if args.tokenizer_first:
        print("\nStep 1: Training tokenizer")
        train_tokenizer_first()
    else:
        print(
            "\nStep 1: (Skip) Tokenizer already trained, using existing weights/tokenizer.model"
        )

    print("\nStep 2: Training model")
    train_with_lmsys_dataset(args)

    print("\nTraining process completed!")
    print("You can now use the trained model for inference with:")
    print(
        "python main.py generate --config [path_to_config] --tokenizer-path weights/tokenizer.model --prompt 'Your prompt here'"
    )


if __name__ == "__main__":
    # DEBUG
    main()
