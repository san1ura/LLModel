import unittest
import torch
import tempfile
import os
from model.transformer import Config, Transformer


class TestConfig(unittest.TestCase):
    def test_config_creation(self):
        """Test that config can be created and saved/loaded"""
        config = Config(vocab_size=1000, d_model=128, n_layers=4, max_len=512)
        
        # Test attributes are set correctly
        self.assertEqual(config.vocab_size, 1000)
        self.assertEqual(config.d_model, 128)
        self.assertEqual(config.n_layers, 4)
        self.assertEqual(config.max_len, 512)
        
        # Test save and load
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            config.save(temp_path)
            loaded_config = Config.load(temp_path)
            
            self.assertEqual(config.vocab_size, loaded_config.vocab_size)
            self.assertEqual(config.d_model, loaded_config.d_model)
            self.assertEqual(config.n_layers, loaded_config.n_layers)
            self.assertEqual(config.max_len, loaded_config.max_len)
        finally:
            os.unlink(temp_path)


class TestTransformer(unittest.TestCase):
    def test_model_creation(self):
        """Test that transformer model can be created"""
        config = Config(vocab_size=1000, d_model=128, n_layers=2, max_len=64, n_heads=4)
        model = Transformer(config)
        
        # Test model attributes
        self.assertEqual(model.config.vocab_size, 1000)
        self.assertEqual(model.config.d_model, 128)
        self.assertEqual(len(model.layers), 2)
        
    def test_model_forward_pass(self):
        """Test that forward pass works"""
        config = Config(vocab_size=1000, d_model=128, n_layers=2, max_len=64, n_heads=4)
        model = Transformer(config)
        
        # Create sample input
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Forward pass
        output, _ = model(input_ids)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, 1000))
        
    def test_model_save_load(self):
        """Test that model can be saved and loaded"""
        config = Config(vocab_size=1000, d_model=128, n_layers=2, max_len=64, n_heads=4)
        model = Transformer(config)
        
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.pth') as f:
            temp_path = f.name
        
        try:
            model.save(temp_path)

            # Create new model and load weights
            new_model = Transformer(config)
            checkpoint = torch.load(temp_path, map_location='cpu')
            new_model.load_state_dict(checkpoint['model_state_dict'])
            
            # Test that loaded model works
            batch_size = 1
            seq_len = 5
            input_ids = torch.randint(0, 1000, (batch_size, seq_len))
            
            original_output, _ = model(input_ids)
            loaded_output, _ = new_model(input_ids)

            # Check outputs are similar
            self.assertTrue(torch.allclose(original_output, loaded_output, atol=1e-5))
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main()