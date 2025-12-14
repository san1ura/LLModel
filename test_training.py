#!/usr/bin/env python3
"""
Test script to train a small model on sample data
"""
import os
import sys
import torch
from torch.utils.data import DataLoader
from model.transformer import Config, Transformer
from tokenizer.train_tokenizer import SentencePieceTokenizer, train_default_tokenizer
from data.preprocessed.build_dataset import PreprocessedDataset, DataCollator
from training.trainer import OptimizedTrainer
from data.prepare_lmsys_dataset import prepare_lmsys_dataset_for_training


def create_sample_data():
    """Create sample data for testing"""
    sample_text = """
    The Transformers library is a powerful tool for natural language processing. 
    It provides pre-trained models that can be used for various tasks. 
    These models are based on the transformer architecture introduced in the paper "Attention Is All You Need". 
    The library supports many different model architectures including BERT, GPT, T5, and more.

    Training large language models requires significant computational resources. 
    However, by using techniques such as transfer learning, we can leverage pre-trained models 
    and fine-tune them for specific tasks. This approach is more efficient than training models from scratch.

    Natural language processing has many applications including text classification, 
    sentiment analysis, and machine translation. The transformer architecture has revolutionized 
    how we approach these tasks by enabling models to train on complex patterns in text.

    The attention mechanism allows models to focus on different parts of the input sequence when making predictions. 
    This has proven to be very effective for tasks that involve sequential data like text, speech, and more.
    """
    
    # Write sample text to file
    with open("sample_data.txt", "w", encoding="utf-8") as f:
        f.write(sample_text)
    
    return "sample_data.txt"


def train_tokenizer_on_sample_data(data_path):
    """Train a tokenizer on the sample data"""
    output_path = os.path.join(".", "test_tokenizer.json")
    tokenizer = train_default_tokenizer([data_path], output_path, vocab_size=1000)
    print(f"Tokenizer trained with vocab size: {tokenizer.get_vocab_size()}")
    return tokenizer


def create_config():
    """Create a small model configuration for testing"""
    config = Config(
        vocab_size=1000,  # Match the tokenizer vocab size
        d_model=128,      # Small model size for testing
        n_layers=4,       # Few layers for quick testing
        max_len=256,      # Reasonable sequence length
        n_heads=8,        # Number of attention heads
        d_ff=256,         # Feed-forward dimension
        dropout=0.1,      # Dropout rate
        use_rope=True,    # Use RoPE
        pos_type='rope',
        attention_type='standard',
        use_gradient_checkpointing=False,
        norm_first=True,
        initializer_range=0.02
    )
    return config


def prepare_dataset_from_text(text_file, tokenizer, output_path="test_dataset.bin"):
    """Prepare a dataset from a text file"""
    # Read the text file
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Tokenize the text
    tokens = tokenizer.encode(text)

    # Break into chunks of max_length
    max_length = 128
    sequences = []
    for i in range(0, len(tokens), max_length):
        chunk = tokens[i:i + max_length]

        # Pad if necessary
        if len(chunk) < max_length:
            chunk.extend([tokenizer.pad_token_id] * (max_length - len(chunk)))

        sequences.append(chunk)

    # Write to binary file
    with open(output_path, 'wb') as out_f:
        for seq in sequences:
            # Write length as 4-byte integer
            seq_len = len(seq)
            out_f.write(seq_len.to_bytes(4, 'little'))

            # Write tokens
            tokens_tensor = torch.tensor(seq, dtype=torch.long)
            out_f.write(tokens_tensor.numpy().tobytes())

    print(f"Dataset prepared with {len(sequences)} sequences")
    return output_path


def main():
    print("Starting test training process...")
    
    # Create sample data
    sample_data_path = create_sample_data()
    
    # Train tokenizer
    print("Training tokenizer...")
    tokenizer = train_tokenizer_on_sample_data(sample_data_path)
    
    # Create dataset
    print("Preparing dataset...")
    dataset_path = prepare_dataset_from_text(sample_data_path, tokenizer)
    
    # Create model config
    config = create_config()
    
    # Create model
    print("Creating model...")
    model = Transformer(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dataset object
    dataset = PreprocessedDataset(dataset_path, block_size=config.max_len, vocab_size=config.vocab_size)
    
    # Create data collator
    collator = DataCollator(
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=config.vocab_size
    )
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=2,  # Small batch size for testing
        shuffle=True,
        collate_fn=collator,
        num_workers=0  # 0 workers for testing to avoid issues
    )
    
    # Create trainer
    trainer = OptimizedTrainer(
        model=model,
        config=config,
        train_data=dataloader,
        save_dir="./test_checkpoints",
        lr=1e-3,  # Higher learning rate for quick testing
        total_steps=20,  # Few steps for quick testing
        log_interval=5,
        save_interval=10,
        eval_interval=10,
        device='cuda' if torch.cuda.is_available() else 'cpu'  # Specify device for trainer
    )

    # Update device to what trainer actually uses
    device = trainer.device
    
    # Print initial evaluation
    print("Initial model evaluation:")
    initial_loss = trainer.evaluate()
    print(f"Initial loss: {initial_loss}")
    
    # Train the model briefly
    print("Starting training...")
    trainer.train(epochs=1)
    
    # Evaluate after training
    print("Final model evaluation:")
    final_loss = trainer.evaluate()
    print(f"Final loss: {final_loss}")
    
    # Save the model
    os.makedirs("test_models", exist_ok=True)
    model.save("test_models/test_model.pth")
    print("Model saved!")
    
    # Test generation
    print("Testing generation...")
    test_input = "The Transformers library"
    input_ids = torch.tensor([tokenizer.encode(test_input)], dtype=torch.long)

    # Move model to CPU for consistent generation (avoid device mismatch)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    input_ids = input_ids.to(device)

    model.eval()
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=20,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

        generated_text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
        print(f"Input: {test_input}")
        print(f"Generated: {generated_text}")
    
    print("Test completed successfully!")


if __name__ == "__main__":
    main()