#!/usr/bin/env python3
"""
Evaluation script to measure the model's performance on test data
"""
import os
import torch
from model.transformer import Config, Transformer
from tokenizer.train_tokenizer import SentencePieceTokenizer
from torch.utils.data import DataLoader
from data.preprocessed.build_dataset import PreprocessedDataset, DataCollator
from evaluation.benchmarks.model_eval import ModelEvaluator


def create_test_data():
    """Create test data for evaluation"""
    # We'll use the same sample data as before
    test_text = """Transformers are deep learning models that have revolutionized natural language processing.
    The attention mechanism allows the model to focus on different parts of the input when making predictions.
    These models have enabled significant advances in tasks like translation, summarization, and question answering.
    """
    
    # Write to a file
    with open("evaluation_test.txt", "w", encoding="utf-8") as f:
        f.write(test_text)
    
    return "evaluation_test.txt"


def evaluate_model():
    """Evaluate the model's performance"""
    print("Starting model evaluation...")
    
    # Load the model we trained in the test
    if os.path.exists("test_models/test_model.pth"):
        checkpoint = torch.load("test_models/test_model.pth", map_location='cpu')
        config_dict = checkpoint.get('config', {})
        config = Config(**config_dict)
        model = Transformer(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded previously trained model")
    else:
        # Create a new small model for evaluation
        config = Config(
            vocab_size=1000,
            d_model=128,
            n_layers=4,
            max_len=256,
            n_heads=8,
            d_ff=256,
            dropout=0.1,
            use_rope=True,
            pos_type='rope',
            attention_type='standard',
            use_gradient_checkpointing=False,
            norm_first=True,
            initializer_range=0.02
        )
        model = Transformer(config)
        print("Created new model for evaluation")
    
    # Load tokenizer
    if os.path.exists("test_tokenizer.json"):
        tokenizer = SentencePieceTokenizer.from_pretrained("test_tokenizer.json")
    else:
        print("Tokenizer not found, skip evaluation")
        return
    
    # Create evaluator
    evaluator = ModelEvaluator(model, tokenizer, config)
    
    # Create simple test samples
    test_samples = [
        {
            'input': 'Transformers are deep learning models',
            'reference': 'that have revolutionized natural language processing.'
        },
        {
            'input': 'The attention mechanism allows',
            'reference': 'the model to focus on different parts'
        }
    ]
    
    print("Running generation quality evaluation...")
    try:
        gen_results = evaluator.evaluate_generation_quality(test_samples, max_new_tokens=10)
        print(f"Generation Quality Results: {gen_results}")
    except Exception as e:
        print(f"Generation quality evaluation failed: {e}")
    
    print("Running perplexity evaluation...")
    # For perplexity, we'd need a dataloader with test data
    # Since we don't have a proper test dataset, we'll skip this for now
    print("Perplexity evaluation: Skipped (requires proper test dataset)")
    
    print("Running efficiency evaluation...")
    try:
        eff_results = evaluator.evaluate_efficiency()
        print(f"Efficiency Results: {eff_results}")
    except Exception as e:
        print(f"Efficiency evaluation failed: {e}")
    
    print("Evaluation completed!")


def main():
    # Create test data if needed
    test_data_path = create_test_data()
    
    # Run evaluation
    evaluate_model()


if __name__ == "__main__":
    main()