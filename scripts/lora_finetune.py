#!/usr/bin/env python3
"""
LoRA fine-tuning script for the transformer model suite
"""
import os
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader, Dataset
from model.transformer import Config, Transformer
from tokenizer.train_tokenizer import TokenizerWrapper
from data.preprocessed.build_dataset import PreprocessedDataset, DataCollator
from peft import LoraConfig, get_peft_model, TaskType


class LoRATransformer(nn.Module):
    """
    Transformer model with LoRA support using PEFT
    """
    def __init__(self, base_model, lora_config):
        super().__init__()
        self.base_model = base_model
        self.lora_config = lora_config
        
        # Apply LoRA to the base model
        self.model = get_peft_model(self.base_model, self.lora_config)
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        # Generate using the base model, potentially with LoRA modifications
        return self.base_model.generate(*args, **kwargs)


def setup_lora_model(model, lora_rank=16, lora_alpha=16, lora_dropout=0.05, target_modules=None):
    """
    Setup a model with LoRA configuration
    
    Args:
        model: Base model to apply LoRA to
        lora_rank: LoRA attention dimension
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout probability for LoRA layers
        target_modules: List of modules to apply LoRA to
    
    Returns:
        LoRA model
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]  # Common target modules
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    
    lora_model = LoRATransformer(model, peft_config)
    return lora_model


def train_lora_model(
    model_path,
    config_path,
    tokenizer_path,
    data_path,
    output_dir,
    lora_rank=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=None,
    epochs=3,
    batch_size=4,
    learning_rate=5e-4
):
    """
    Train a model with LoRA fine-tuning
    
    Args:
        model_path: Path to the base model
        config_path: Path to the model configuration
        tokenizer_path: Path to the tokenizer
        data_path: Path to the training data
        output_dir: Output directory for fine-tuned model
        lora_rank: LoRA attention dimension
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout probability for LoRA layers
        target_modules: List of modules to apply LoRA to
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for training
    """
    print("Loading model and tokenizer...")
    
    # Load configuration
    config = Config.load(config_path)
    
    # Load base model
    model = Transformer(config)
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    # Setup LoRA
    lora_model = setup_lora_model(
        model,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules
    )
    
    print(f"LoRA model created with {sum(p.numel() for p in lora_model.parameters() if p.requires_grad):,} trainable parameters")
    
    # Load tokenizer
    tokenizer = TokenizerWrapper.from_pretrained(tokenizer_path)
    
    # Prepare dataset
    print("Loading dataset...")
    train_dataset = PreprocessedDataset(data_path, block_size=config.max_len, vocab_size=config.vocab_size)
    
    # Create data collator
    collator = DataCollator(
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=config.vocab_size
    )
    
    # Create data loader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(lora_model.parameters(), lr=learning_rate)
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lora_model.to(device)
    
    # Training loop
    print("Starting training...")
    lora_model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in train_dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device) if 'attention_mask' in batch else None
            
            # Forward pass
            outputs = lora_model(input_ids=input_ids, attention_mask=attention_mask)
            if isinstance(outputs, tuple):
                logits = outputs[0]  # The first element is usually the logits
            else:
                logits = outputs
            
            # Calculate loss (assuming next token prediction)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {num_batches}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
    
    # Save the fine-tuned LoRA model
    print("Saving fine-tuned model...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the LoRA adapter
    lora_model.model.save_pretrained(output_dir)
    
    # Also save the base config
    config.save(os.path.join(output_dir, "config.json"))
    
    print(f"Fine-tuned model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune model with LoRA")
    
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the base model")
    parser.add_argument("--config-path", type=str, required=True,
                        help="Path to the model configuration")
    parser.add_argument("--tokenizer-path", type=str, required=True,
                        help="Path to the tokenizer")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to the training data")
    parser.add_argument("--output-dir", type=str, default="checkpoints/lora_finetuned",
                        help="Output directory for fine-tuned model")
    
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA attention dimension")
    parser.add_argument("--lora-alpha", type=int, default=16,
                        help="LoRA scaling factor")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout probability")
    parser.add_argument("--target-modules", type=str, default="q_proj,v_proj",
                        help="Target modules for LoRA injection, comma-separated")
    
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-4,
                        help="Learning rate for training")
    
    args = parser.parse_args()
    
    target_modules = args.target_modules.split(',')
    target_modules = [module.strip() for module in target_modules if module.strip()]
    
    train_lora_model(
        model_path=args.model_path,
        config_path=args.config_path,
        tokenizer_path=args.tokenizer_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )


if __name__ == "__main__":
    main()