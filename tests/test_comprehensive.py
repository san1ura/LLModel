"""
Comprehensive tests for the model architecture
"""
import unittest
import torch
from model.transformer import Config, Transformer
from model.layers.attention import MHA, FlashAttention2
from model.layers.feedforward import SwiGLU
from model.layers.rotary_embedding import RotaryEmbedding
from model.layers.kv_cache import KVCacheManager


class TestComprehensive(unittest.TestCase):
    def test_config_creation(self):
        """Test Config creation with various parameters"""
        config = Config(
            vocab_size=1000,
            d_model=256,
            n_layers=2,
            n_heads=4,
            d_ff=512,
            max_len=128
        )
        self.assertIsInstance(config, Config)
        self.assertEqual(config.vocab_size, 1000)
        self.assertEqual(config.d_model, 256)

    def test_model_initialization(self):
        """Test model initialization"""
        config = Config(
            vocab_size=1000,
            d_model=256,
            n_layers=2,
            n_heads=4,
            d_ff=512,
            max_len=128
        )
        model = Transformer(config)
        self.assertIsInstance(model, Transformer)
        self.assertEqual(model.config.vocab_size, 1000)

    def test_forward_pass(self):
        """Test basic forward pass"""
        config = Config(
            vocab_size=1000,
            d_model=256,
            n_layers=2,
            n_heads=4,
            d_ff=512,
            max_len=128
        )
        model = Transformer(config)
        model.eval()

        # Create sample input
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            output, kv_caches = model(input_ids)

        expected_shape = (batch_size, seq_len, config.vocab_size)
        self.assertEqual(output.shape, expected_shape)
        self.assertEqual(len(kv_caches), config.n_layers)

    def test_generation(self):
        """Test generation functionality"""
        config = Config(
            vocab_size=1000,
            d_model=256,
            n_layers=2,
            n_heads=4,
            d_ff=512,
            max_len=128
        )
        model = Transformer(config)
        model.eval()

        # Create sample input
        batch_size, seq_len = 1, 5
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Test generation
        generated = model.generate(
            input_ids,
            max_new_tokens=3,
            do_sample=False  # Use greedy decoding for predictability
        )

        expected_len = seq_len + 3
        self.assertEqual(generated.shape[1], expected_len)

    def test_attention_components(self):
        """Test attention components individually"""
        # Test Rotary Embedding
        rot_emb = RotaryEmbedding(dim=64, max_len=128)
        pos = torch.arange(10).unsqueeze(0)  # [1, 10]
        cos, sin = rot_emb(pos)
        self.assertIsNotNone(cos)
        self.assertIsNotNone(sin)

        # Test KV Cache Manager
        cache_manager = KVCacheManager(
            max_batch_size=2,
            max_seq_len=128,
            num_heads=4,
            head_dim=64
        )
        self.assertIsInstance(cache_manager, KVCacheManager)

        # Test MHA
        mha = MHA(d_model=256, n_heads=4, use_rope=True)
        x = torch.randn(2, 10, 256)
        out, k, v = mha(x)
        self.assertEqual(out.shape, x.shape)
        self.assertIsNotNone(k)
        self.assertIsNotNone(v)

    def test_feedforward_components(self):
        """Test feedforward components"""
        swiglu = SwiGLU(hidden_dim=512, d_model=256)
        x = torch.randn(2, 10, 256)
        out = swiglu(x)
        self.assertEqual(out.shape, x.shape)


def run_comprehensive_tests():
    """Run all comprehensive tests and report results"""
    test_classes = [
        TestComprehensive
    ]

    for test_class in test_classes:
        print(f"Running tests for {test_class.__name__}")
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        if result.failures or result.errors:
            print(f"Errors or failures found in {test_class.__name__}")
            for failure in result.failures:
                print(f"FAILURE: {failure[0]} - {failure[1]}")
            for error in result.errors:
                print(f"ERROR: {error[0]} - {error[1]}")
        else:
            print(f"All tests passed for {test_class.__name__}")
        print()


if __name__ == '__main__':
    run_comprehensive_tests()
