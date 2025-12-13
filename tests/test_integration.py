import unittest
import torch
import tempfile
import os
from model.transformer import Config, Transformer
from tokenizer.train_tokenizer import TokenizerWrapper, TokenizerTrainer
from training.trainer import OptimizedTrainer
from serving.inference_opt.generate import InferenceEngine


class TestModelIntegration(unittest.TestCase):
    """Integration tests for the transformer model suite"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.config = Config(
            vocab_size=1000,
            d_model=128,
            n_layers=2,
            n_heads=4,
            d_ff=256,
            max_len=64,
            dropout=0.1
        )
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after each test method."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_model_creation_and_save_load(self):
        """Test that model can be created, saved, and loaded properly."""
        model = Transformer(self.config)
        
        # Test model parameters
        total_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(total_params, 0)
        
        # Save model
        model_path = os.path.join(self.temp_dir, "test_model.pt")
        torch.save(model.state_dict(), model_path)
        
        # Load model
        new_model = Transformer(self.config)
        new_model.load_state_dict(torch.load(model_path))
        
        # Check that parameters are the same
        original_params = list(model.parameters())
        loaded_params = list(new_model.parameters())
        
        for orig_param, loaded_param in zip(original_params, loaded_params):
            self.assertTrue(torch.allclose(orig_param, loaded_param))
    
    def test_tokenizer_integration(self):
        """Test tokenizer integration with the model."""
        # Create a simple tokenizer
        trainer = TokenizerTrainer(vocab_size=1000)
        # Train on some sample text
        sample_texts = ["hello world", "test tokenizer", "integration test"]
        
        # Create a temporary file for training
        train_file = os.path.join(self.temp_dir, "train.txt")
        with open(train_file, 'w') as f:
            for text in sample_texts:
                f.write(text + "\n")
        
        tokenizer_path = os.path.join(self.temp_dir, "tokenizer.json")
        tokenizer = trainer.train_from_files([train_file], tokenizer_path)
        
        # Wrap with our tokenizer wrapper
        wrapper = TokenizerWrapper(tokenizer_path=tokenizer_path)
        
        # Test encoding/decoding
        test_text = "hello world"
        encoded = wrapper.encode(test_text)
        decoded = wrapper.decode(encoded)
        
        # The decoded text might not be identical due to tokenization,
        # but the tokenizer should handle the basic functionality
        self.assertIsInstance(encoded, list)
        self.assertIsInstance(decoded, str)
        self.assertGreater(len(encoded), 0)
    
    def test_training_pipeline(self):
        """Test the training pipeline with a simple example."""
        model = Transformer(self.config)
        
        # Create dummy data for training
        batch_size = 2
        seq_len = 10
        vocab_size = self.config.vocab_size
        
        # Generate random input and target
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Create simple loss function for testing
        def simple_train_step(model, batch):
            input_ids = batch['input_ids']
            labels = batch['labels']
            
            outputs = model(input_ids)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss
        
        # Run a simple forward pass
        outputs = model(input_ids)
        self.assertEqual(outputs[0].shape, (batch_size, seq_len, vocab_size))
    
    def test_inference_engine(self):
        """Test the inference engine functionality."""
        # Create model and config
        model = Transformer(self.config)
        model.eval()  # Set to evaluation mode
        
        # Create a simple tokenizer for testing
        # For this test, we'll use a simple approach
        class SimpleTokenizer:
            def __init__(self, vocab_size):
                self.vocab_size = vocab_size
                self.pad_token_id = 0
                self.bos_token_id = 1
                self.eos_token_id = 2
                self.unk_token_id = 3
            
            def encode(self, text, add_special_tokens=True):
                # Simple tokenization for test purposes
                tokens = [abs(hash(c)) % (self.vocab_size - 10) + 10 for c in text[:10]]
                if add_special_tokens:
                    tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
                return tokens
            
            def decode(self, token_ids, skip_special_tokens=False):
                # Simple decoding for test purposes
                return f"decoded_text_{len(token_ids)}_tokens"
        
        tokenizer = SimpleTokenizer(self.config.vocab_size)
        
        # Create inference engine
        engine = InferenceEngine(model, tokenizer, self.config)
        
        # Test text generation
        result = engine.generate("test prompt", max_new_tokens=10)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
    
    def test_config_save_load(self):
        """Test saving and loading configuration."""
        # Save config
        config_path = os.path.join(self.temp_dir, "test_config.json")
        self.config.save(config_path)
        
        # Load config
        loaded_config = Config.load(config_path)
        
        # Check that all attributes match
        for attr_name, attr_value in self.config.__dict__.items():
            self.assertEqual(attr_value, getattr(loaded_config, attr_name))


class TestTokenizerAdvanced(unittest.TestCase):
    """Advanced tests for tokenizer functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_tokenizer_trainer(self):
        """Test the tokenizer trainer functionality."""
        # Create sample training data
        sample_texts = [
            "This is a sample sentence for training.",
            "Another sentence to train the tokenizer.",
            "More data for better tokenization results."
        ]
        
        # Create training file
        train_file = os.path.join(self.temp_dir, "train_data.txt")
        with open(train_file, 'w') as f:
            for text in sample_texts:
                f.write(text + "\n")
        
        # Train tokenizer
        trainer = TokenizerTrainer(vocab_size=500)
        tokenizer_path = os.path.join(self.temp_dir, "test_tokenizer.json")
        tokenizer = trainer.train_from_files([train_file], tokenizer_path)
        
        # Verify tokenizer file exists
        self.assertTrue(os.path.exists(tokenizer_path))
        
        # Test tokenizer functionality
        wrapper = TokenizerWrapper(tokenizer_path=tokenizer_path)
        text = "This is a test."
        encoded = wrapper.encode(text)
        decoded = wrapper.decode(encoded)
        
        self.assertIsInstance(encoded, list)
        self.assertIsInstance(decoded, str)
        self.assertGreater(len(encoded), 0)


if __name__ == '__main__':
    unittest.main()