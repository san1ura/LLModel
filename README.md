# LLModel

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/github/license/san1ura/LLModel)

LLModel is a modular infrastructure for training, fine-tuning, and serving transformer-based language models from scratch. It provides a comprehensive solution for researchers and developers looking to build, experiment with, and deploy their own language models.

This project is not an alternative to HuggingFace but rather a research-focused, experimental platform for exploring transformer architectures.

## Features

- Transformer architecture (multi-head attention, RoPE, KV cache)
- Custom tokenizer (SentencePiece / BPE)
- Pretraining + fine-tuning (SFT)
- LoRA support
- Flash / fused attention options
- Checkpointing
- Evaluation & benchmarks
- Docker supported inference
- CPU / GPU compatibility
- Evolutionary algorithm for model improvement

## Architecture Overview

```
High-level data flow:

Raw Text → Tokenizer → Binary Dataset → Trainer → Checkpoint → Inference
```

Core components:
- `model/` → Model definition and architecture
- `training/` → Training loop implementation
- `tokenizer/` → Vocabulary generation and tokenization
- `data/` → Dataset preprocessing pipeline
- `serving/` → Inference and API services
- `evaluation/` → Benchmarking and evaluation tools
- `evolution/` → Evolutionary algorithms for model enhancement

## Directory Structure

```
model/
├── layers/        # Attention, FFN, RoPE implementations
├── transformer.py # Main model definition
├── __init__.py

training/
├── backup/        # Backup of training files
├── utils/         # Training utilities
├── train.py       # Main training script
├── trainer.py     # Abstract trainer implementation
├── __init__.py

tokenizer/
├── custom_sentencepiece_tokenizer.py  # Custom tokenizer implementation
├── train_tokenizer_from_dataset.py    # Train tokenizer from dataset
├── train_tokenizer.py                 # Basic tokenizer training
├── __init__.py

data/
├── preprocessed/           # Preprocessed datasets
├── prepare_lmsys_dataset.py # Script for preparing LMSYS dataset
├── __init__.py

serving/
├── inference_opt/  # Optimized inference engine
├── __init__.py

evaluation/
├── benchmarks/     # Model benchmarks

evolution/
├── evo_loop.py     # Evolutionary algorithm implementation

config/
├── config.json             # Base configuration
├── default_config.json     # Default model configuration
├── small_config.json       # Small model configuration
├── tiny_config.json        # Tiny model configuration

scripts/                    # Utility scripts

optim/                      # Optimization algorithms

tests/                      # Unit tests

checkpoints/                # Saved model checkpoints

weights/                    # Model weights and tokenizer files
```

Each directory serves a specific purpose in the overall model development pipeline:
- `model/` contains the core transformer architecture definitions
- `training/` implements the training logic and optimization procedures
- `tokenizer/` provides tokenization tools and vocabulary building
- `data/` handles data preprocessing and dataset creation
- `serving/` manages inference and API deployment
- `evaluation/` includes tools for model assessment and benchmarking
- `evolution/` contains algorithms for model evolution using genetic approaches

These directories are generated at runtime and are excluded via `.gitignore`.

## Requirements

### Hardware
- Minimum RAM: 16 GB
- GPU Recommendations: NVIDIA GPU (CUDA compatible) with at least 8GB VRAM (for training small models)
- CUDA Versions: CUDA 11.7 or higher (for PyTorch)

### Software
- Python 3.10 or higher
- PyTorch >= 2.0.0
- CUDA / ROCm (for GPU usage)

## Installation

### Local (pip)
```bash
git clone https://github.com/san1ura/LLModel.git
cd LLModel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### Editable Mode
```bash
pip install -e .
```

### Docker
```bash
docker build -t llmodel .
docker compose up
```

## Quick Start

Simple 5-minute demo:

```bash
# 1. Prepare dataset
python data/prepare_lmsys_dataset.py

# 2. Train tokenizer
python tokenizer/train_tokenizer_from_dataset.py

# 3. Create a small model configuration
python main.py create-config --type tiny --output config/tiny_config.json

# 4. Start training
python main.py pretrain --config config/tiny_config.json --use-lmsys-dataset --epochs 1

# 5. Test text generation
python main.py generate --config config/tiny_config.json --tokenizer-path weights/tokenizer.model --prompt "Hello, how are you?" --max-tokens 50
```

Expected output example:
```
Generated: Hello, how are you? I am a language model and I am functioning properly. How can I assist you today?
```

## Dataset Pipeline

### Supported Dataset Formats
- Plain text
- JSONL
- Custom binary format

### Binary Dataset Format
The binary format stores tokenized sequences as contiguous int32 arrays, indexed via a memory-mapped offset table for efficient random access.
- `.bin + .index`
- Tokenized format using int32 arrays
- Memory-mapped for efficient random access during training
- Why binary? Faster read and disk access performance

This script assumes the LMSYS dataset is already downloaded. See DATASETS.md for details.

## Tokenizer

### Type
SentencePiece BPE (Byte-Pair Encoding)

### Training
```bash
python tokenizer/train_tokenizer_from_dataset.py
```

### Tokenizer-Model Compatibility
- vocab_size: The tokenizer and model must use the same vocabulary size
- Config synchronization: The vocab_size used in the model configuration must match the tokenizer
- Potential errors: Vocabulary mismatch errors can prevent the model from functioning correctly
- **Important**: Changing the tokenizer requires regenerating the dataset and updating all configs.

## Model Configuration

### Configuration Files
Different configurations are available:
- tiny / small / base / large / xl
- Each parameter explained:

| Parameter | Description |
|----------|-------------|
| d_model | Hidden dimension of the model |
| n_layers | Number of transformer layers |
| n_heads | Number of attention heads |
| d_ff | Feed-forward layer dimension |
| max_len | Maximum sequence length |
| vocab_size | Vocabulary size |
| dropout | Dropout rate |
| use_rope | Use Rotary Position Embedding |
| attention_type | Type of attention mechanism (standard, flash, etc.) |

## Training

### Pretraining
```bash
python main.py pretrain --config config/default_config.json --data-path data/train.txt --tokenizer-path weights/tokenizer.model
```

### Gradient Accumulation
- Used to simulate larger effective batch sizes under memory constraints
- Adjustable with the `--gradient-accumulation-steps` parameter

### Mixed Precision
- Reduces training time and optimizes memory usage
- Supported via PyTorch AMP (Automatic Mixed Precision)

### Checkpointing
- Frequency: At specific steps or at the end of each epoch
- Storage location: Inside the `checkpoints/` directory

## Fine-tuning & LoRA

LoRA (Low-Rank Adaptation) enables efficient fine-tuning by applying low-rank matrix updates to model weights.

When to use:
- Fine-tuning large models with small datasets
- When memory constraints are present

Example parameters:
```bash
python main.py lora-finetune --config config/small_config.json --data-path data/finetune.txt --tokenizer-path weights/tokenizer.model --lora-rank 16 --lora-alpha 16
```

## Evaluation & Benchmark

- Metrics: Perplexity, BLEU, ROUGE, accuracy, F1 score, throughput, and memory usage
- Execution location: `evaluation/benchmarks/` directory
- Command: `python evaluation/benchmarks/model_eval.py`
- Currently implemented metrics include perplexity, generation quality (BLEU/ROUGE where dependencies available), downstream task evaluation, and efficiency metrics.
- Planned metrics include GSM8K, MMLU, and TruthfulQA.

## Inference / Serving

### CLI Inference
```bash
python main.py generate --config config/default_config.json --model-path checkpoints/final_model.pth --tokenizer-path weights/tokenizer.model --prompt "What's the weather like today?" --max-tokens 100
```

### Docker Inference
API server can be started with DockerCompose:
```bash
docker compose up transformer-api
```

### API Mode
```bash
python main.py api --config config/default_config.json --tokenizer-path weights/tokenizer.model --port 8000
```

## Testing

```bash
pytest
```

Module tests verify the correctness of model components, while integration tests check the compatibility of system components.

## Performance and Optimization

- Flash attention: Faster attention mechanism calculations
- KV cache: Prevents repeated calculations during autoregressive generation
- Batch vs streaming: Optimized approaches for different use cases

## Logging & Monitoring

- training.log: Detailed record of training processes
- TensorBoard / WandB integration: Available for experimental monitoring

## Frequently Asked Questions

- CUDA OOM: Reduce gradient accumulation amount or batch size if GPU memory is insufficient
- Vocabulary mismatch: Ensure model and tokenizer vocabulary sizes match
- Slow training: Check CUDA installation and PyTorch GPU version if GPU isn't being utilized

## Security & Disclaimer

- Model bias: May contain potential biases from training data
- Misuse warning: Should not be used for generating harmful content

## Roadmap

- Multi-GPU support
- Export to Hugging Face format
- Model quantization
- Additional evolutionary algorithms

## References

- Vaswani, A. et al. (2017). Attention Is All You Need
- Liu, L. et al. (2021). Efficient Training of Language Models to Fill in the Middle
- Turc, I. et al. (2019). Well-Read Students Learn Better: On the Importance of Pre-training Compact Models
