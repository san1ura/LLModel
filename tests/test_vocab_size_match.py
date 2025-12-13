"""
Vocabulary size mismatch test
This test ensures that the model's embedding size matches the tokenizer's vocabulary size
to prevent CUDA assertion errors during training.
"""

import os
import sys
import torch
import pytest

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from model.transformer import Config, Transformer
from tokenizer.train_tokenizer import TokenizerWrapper


def test_model_tokenizer_vocab_size_match():
    """
    Test to ensure the model's embedding size matches the tokenizer's vocabulary size.
    This prevents the CUDA error: device-side assert triggered error related to
    'srcIndex < srcSelectDimSize' assertion failure during token generation.
    """
    # Load default config
    config_path = os.path.join(project_root, "config", "default_config.json")
    config = Config.load(config_path)
    
    # Initialize model with config
    model = Transformer(config)
    
    # Load tokenizer
    tokenizer_path = os.path.join(project_root, "weights", "tokenizer.model")
    if not os.path.exists(tokenizer_path):
        pytest.skip(f"Tokenizer not found at {tokenizer_path}, skipping test")
        
    tokenizer = TokenizerWrapper.from_pretrained(tokenizer_path)
    
    # Get vocab sizes
    model_vocab_size = model.embed.weight.shape[0]
    tokenizer_vocab_size = tokenizer.tokenizer.get_vocab_size()

    print(f"Model vocab size: {model_vocab_size}")
    print(f"Tokenizer vocab size: {tokenizer_vocab_size}")

    # Assert they match
    assert model_vocab_size == tokenizer_vocab_size, (
        f"Model vocab size ({model_vocab_size}) does not match "
        f"tokenizer vocab size ({tokenizer_vocab_size}). "
        f"This mismatch can cause CUDA assertion errors during training/generation."
    )

    print("✓ Model vocab size matches tokenizer vocab size")


def test_resize_token_embeddings():
    """
    Test that resize_token_embeddings function works properly
    """
    # Load a config with smaller vocab than tokenizer
    config_path = os.path.join(project_root, "config", "default_config.json")
    config = Config.load(config_path)
    
    # Modify config to have a smaller vocab size than tokenizer
    original_vocab_size = config.vocab_size
    config.vocab_size = 1000  # Use a smaller vocab size
    
    # Initialize model with smaller vocab
    model = Transformer(config)
    
    # Load tokenizer
    tokenizer_path = os.path.join(project_root, "weights", "tokenizer.model")
    if not os.path.exists(tokenizer_path):
        pytest.skip(f"Tokenizer not found at {tokenizer_path}, skipping test")
        
    tokenizer = TokenizerWrapper.from_pretrained(tokenizer_path)
    
    # Initially, they should not match
    model_vocab_size = model.embed.weight.shape[0]
    tokenizer_vocab_size = tokenizer.tokenizer.get_vocab_size()

    assert model_vocab_size != tokenizer_vocab_size, (
        "Model and tokenizer vocab sizes should be different initially for this test"
    )

    # Resize model embeddings to match tokenizer
    model.resize_token_embeddings(tokenizer_vocab_size)

    # Now they should match
    new_model_vocab_size = model.embed.weight.shape[0]

    assert new_model_vocab_size == tokenizer_vocab_size, (
        f"Model vocab size ({new_model_vocab_size}) does not match "
        f"tokenizer vocab size ({tokenizer_vocab_size}) after resize_token_embeddings()."
    )

    # Restore original config
    config.vocab_size = original_vocab_size

    print("✓ resize_token_embeddings() works correctly")


def test_tokenizer_generation_edge_cases():
    """
    Test tokenizer edge cases that could lead to CUDA assertion errors
    """
    # Load tokenizer
    tokenizer_path = os.path.join(project_root, "weights", "tokenizer.model")
    if not os.path.exists(tokenizer_path):
        pytest.skip(f"Tokenizer not found at {tokenizer_path}, skipping test")
    
    tokenizer = TokenizerWrapper.from_pretrained(tokenizer_path)
    
    # Test with various input types that may cause issues
    test_inputs = [
        "Hello world",
        "A very long sentence with many tokens to ensure we cover edge cases during tokenization.",
        "Special characters: !@#$%^&*()_+-=[]{}|;':\",./<>?",
        "",  # Empty string
        "Numbers: 1234567890",
        "Mixed case: HeLLo WoRLd",
    ]
    
    for test_input in test_inputs:
        # Encode and decode the text
        encoded = tokenizer.encode(test_input)
        decoded = tokenizer.decode(encoded)
        
        # Verify that the tokens are within the valid range
        tokenizer_vocab_size = tokenizer.tokenizer.get_vocab_size()
        for token_id in encoded:
            assert 0 <= token_id < tokenizer_vocab_size, (
                f"Token ID {token_id} is out of valid range [0, {tokenizer_vocab_size})"
            )
    
    print("✓ Tokenizer edge cases handled correctly")


if __name__ == "__main__":
    test_model_tokenizer_vocab_size_match()
    test_resize_token_embeddings()
    test_tokenizer_generation_edge_cases()
    print("\nAll tests passed! ✓")