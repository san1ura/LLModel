"""
Test for the serving/inference functionality
"""
import unittest
import torch
import tempfile
import os
from model.transformer import Config, Transformer
from tokenizer.train_tokenizer import TokenizerWrapper
from serving.inference_opt.generate import InferenceEngine, AsyncInferenceEngine, ModelServer


class TestInferenceEngine(unittest.TestCase):
    def setUp(self):
        # Create a small model for testing
        self.config = Config(
            vocab_size=100,
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=128,
            max_len=32
        )
        self.model = Transformer(self.config)
        
        # Create a temporary tokenizer file for testing
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.tokenizer_path = os.path.join(tmp_dir, "test_tokenizer.json")
            
            # Create a simple tokenizer structure
            import json
            from tokenizers import Tokenizer
            from tokenizers.models import BPE
            
            vocab = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3, "hello": 4, "world": 5, ".": 6}
            
            tokenizer_json = {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "model": {
                    "type": "BPE",
                    "dropout": None,
                    "unk_token": "<unk>",
                    "continuing_subword_prefix": None,
                    "end_of_word_suffix": None,
                    "fuse_unk": False,
                    "vocab": vocab,
                    "merges": []
                },
                "pre_tokenizer": {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": True
                },
                "post_processor": {
                    "type": "TemplateProcessing",
                    "single": [
                        {"SpecialToken": {"id": "<s>", "type_id": 0}},
                        {"Sequence": {"id": "A", "type_id": 0}},
                        {"SpecialToken": {"id": "</s>", "type_id": 0}}
                    ],
                    "pair": [
                        {"SpecialToken": {"id": "<s>", "type_id": 0}},
                        {"Sequence": {"id": "A", "type_id": 0}},
                        {"SpecialToken": {"id": "</s>", "type_id": 0}},
                        {"Sequence": {"id": "B", "type_id": 1}},
                        {"SpecialToken": {"id": "</s>", "type_id": 1}}
                    ],
                    "special_tokens": {
                        "<s>": {
                            "id": "<s>",
                            "ids": [2],
                            "tokens": ["<s>"]
                        },
                        "</s>": {
                            "id": "</s>",
                            "ids": [3],
                            "tokens": ["</s>"]
                        }
                    }
                },
                "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
                "normalizer": None
            }
            
            with open(self.tokenizer_path, 'w') as f:
                json.dump(tokenizer_json, f)
            
            self.tokenizer = TokenizerWrapper(tokenizer_path=self.tokenizer_path)

    def test_inference_engine_creation(self):
        """Test that inference engine can be created"""
        engine = InferenceEngine(self.model, self.tokenizer, self.config)
        
        self.assertEqual(engine.model, self.model)
        self.assertEqual(engine.tokenizer, self.tokenizer)
        self.assertEqual(engine.config, self.config)
        # Device can be 'cuda' or 'cpu' depending on system - check that it's set
        self.assertIsInstance(engine.device, str)

    def test_preprocess_inputs(self):
        """Test input preprocessing functionality"""
        engine = InferenceEngine(self.model, self.tokenizer, self.config)
        
        # Test single input
        inputs = "hello world"
        result = engine.preprocess_inputs(inputs)
        
        self.assertIn('input_ids', result)
        self.assertIn('attention_mask', result)
        self.assertEqual(result['input_ids'].dim(), 2)  # Should be batched
        
        # Test multiple inputs
        inputs = ["hello world", "test sentence"]
        result = engine.preprocess_inputs(inputs)
        
        self.assertEqual(result['input_ids'].size(0), 2)  # Batch size of 2

    def test_postprocess_outputs(self):
        """Test output postprocessing functionality"""
        engine = InferenceEngine(self.model, self.tokenizer, self.config)
        
        # Test with dummy token IDs
        dummy_ids = torch.tensor([[1, 2, 3, 4]])  # Batch of 1, 4 tokens
        decoded = engine.postprocess_outputs(dummy_ids)
        
        self.assertIsInstance(decoded, list)
        self.assertEqual(len(decoded), 1)
        self.assertIsInstance(decoded[0], str)

    def test_generate_functionality(self):
        """Test generation functionality"""
        engine = InferenceEngine(self.model, self.tokenizer, self.config)
        
        # Test single input generation
        prompt = "hello"
        result = engine.generate(prompt, max_new_tokens=5)
        
        self.assertIsInstance(result, str)
        
        # Test multiple input generation
        prompts = ["hello", "test"]
        results = engine.generate(prompts, max_new_tokens=5)
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)

    def test_batch_generate(self):
        """Test batch generation functionality"""
        engine = InferenceEngine(self.model, self.tokenizer, self.config)
        
        prompts = ["hello", "test", "generation"]
        results = engine.batch_generate(prompts, max_new_tokens=5)
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)
        
        for result in results:
            self.assertIsInstance(result, str)

    def test_benchmark_inference(self):
        """Test inference benchmarking functionality"""
        engine = InferenceEngine(self.model, self.tokenizer, self.config)
        
        # Run a small benchmark
        benchmark_results = engine.benchmark_inference(
            input_text="test",
            num_generations=2,
            max_new_tokens=3
        )
        
        expected_keys = [
            'avg_generation_time',
            'avg_num_tokens',
            'tokens_per_second',
            'num_generations'
        ]
        
        for key in expected_keys:
            self.assertIn(key, benchmark_results)


class TestAsyncInferenceEngine(unittest.TestCase):
    def setUp(self):
        # Create a small model for testing
        self.config = Config(
            vocab_size=100,
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=128,
            max_len=32
        )
        self.model = Transformer(self.config)
        
        # Create a temporary tokenizer file for testing
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.tokenizer_path = os.path.join(tmp_dir, "test_tokenizer.json")
            
            # Create a simple vocab for the tokenizer
            import json
            from tokenizers import Tokenizer
            from tokenizers.models import BPE
            
            vocab = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3, "hello": 4, "world": 5, ".": 6}
            
            tokenizer_json = {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "model": {
                    "type": "BPE",
                    "dropout": None,
                    "unk_token": "<unk>",
                    "continuing_subword_prefix": None,
                    "end_of_word_suffix": None,
                    "fuse_unk": False,
                    "vocab": vocab,
                    "merges": []
                },
                "pre_tokenizer": {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": True
                },
                "post_processor": {
                    "type": "TemplateProcessing",
                    "single": [
                        {"SpecialToken": {"id": "<s>", "type_id": 0}},
                        {"Sequence": {"id": "A", "type_id": 0}},
                        {"SpecialToken": {"id": "</s>", "type_id": 0}}
                    ],
                    "pair": [
                        {"SpecialToken": {"id": "<s>", "type_id": 0}},
                        {"Sequence": {"id": "A", "type_id": 0}},
                        {"SpecialToken": {"id": "</s>", "type_id": 0}},
                        {"Sequence": {"id": "B", "type_id": 1}},
                        {"SpecialToken": {"id": "</s>", "type_id": 1}}
                    ],
                    "special_tokens": {
                        "<s>": {
                            "id": "<s>",
                            "ids": [2],
                            "tokens": ["<s>"]
                        },
                        "</s>": {
                            "id": "</s>",
                            "ids": [3],
                            "tokens": ["</s>"]
                        }
                    }
                },
                "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
                "normalizer": None
            }
            
            with open(self.tokenizer_path, 'w') as f:
                json.dump(tokenizer_json, f)
            
            self.tokenizer = TokenizerWrapper(tokenizer_path=self.tokenizer_path)

    def test_async_engine_creation(self):
        """Test that async inference engine can be created"""
        async_engine = AsyncInferenceEngine(self.model, self.tokenizer, self.config)
        
        self.assertIsInstance(async_engine, AsyncInferenceEngine)
        self.assertEqual(async_engine.model, self.model)
        self.assertGreater(async_engine.max_concurrent_requests, 0)


class TestModelServer(unittest.TestCase):
    def setUp(self):
        # Create a small model for testing
        self.config = Config(
            vocab_size=100,
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=128,
            max_len=32
        )
        self.model = Transformer(self.config)
        
        # Create a temporary tokenizer file for testing
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.tokenizer_path = os.path.join(tmp_dir, "test_tokenizer.json")
            
            # Create a simple vocab for the tokenizer
            import json
            from tokenizers import Tokenizer
            from tokenizers.models import BPE
            
            vocab = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3, "hello": 4, "world": 5, ".": 6}
            
            tokenizer_json = {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "model": {
                    "type": "BPE",
                    "dropout": None,
                    "unk_token": "<unk>",
                    "continuing_subword_prefix": None,
                    "end_of_word_suffix": None,
                    "fuse_unk": False,
                    "vocab": vocab,
                    "merges": []
                },
                "pre_tokenizer": {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": True
                },
                "post_processor": {
                    "type": "TemplateProcessing",
                    "single": [
                        {"SpecialToken": {"id": "<s>", "type_id": 0}},
                        {"Sequence": {"id": "A", "type_id": 0}},
                        {"SpecialToken": {"id": "</s>", "type_id": 0}}
                    ],
                    "pair": [
                        {"SpecialToken": {"id": "<s>", "type_id": 0}},
                        {"Sequence": {"id": "A", "type_id": 0}},
                        {"SpecialToken": {"id": "</s>", "type_id": 0}},
                        {"Sequence": {"id": "B", "type_id": 1}},
                        {"SpecialToken": {"id": "</s>", "type_id": 1}}
                    ],
                    "special_tokens": {
                        "<s>": {
                            "id": "<s>",
                            "ids": [2],
                            "tokens": ["<s>"]
                        },
                        "</s>": {
                            "id": "</s>",
                            "ids": [3],
                            "tokens": ["</s>"]
                        }
                    }
                },
                "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
                "normalizer": None
            }
            
            with open(self.tokenizer_path, 'w') as f:
                json.dump(tokenizer_json, f)
            
            self.tokenizer = TokenizerWrapper(tokenizer_path=self.tokenizer_path)

    def test_server_creation(self):
        """Test that model server can be created"""
        server = ModelServer(self.model, self.tokenizer, self.config)
        
        self.assertEqual(server.model, self.model)
        self.assertEqual(server.tokenizer, self.tokenizer)
        self.assertEqual(server.config, self.config)
        self.assertEqual(server.port, 8000)

    def test_health_status(self):
        """Test health status functionality"""
        server = ModelServer(self.model, self.tokenizer, self.config)
        
        health = server.get_health_status()
        self.assertIn('status', health)
        self.assertIn('timestamp', health)

        # Perform a health check
        success = server.perform_health_check()
        self.assertIsInstance(success, bool)
        
        # Check that health status was updated
        updated_health = server.get_health_status()
        self.assertIn('status', updated_health)

    def test_server_stats(self):
        """Test server statistics functionality"""
        server = ModelServer(self.model, self.tokenizer, self.config)
        
        stats = server.get_server_stats()
        self.assertIn('requests_processed', stats)
        self.assertIn('total_processing_time', stats)
        self.assertIn('active_requests', stats)
        
        # Update stats
        server.update_stats(0.1)  # 0.1 seconds processing time
        
        updated_stats = server.get_server_stats()
        self.assertEqual(updated_stats['requests_processed'], 1)
        self.assertGreater(updated_stats['total_processing_time'], 0)

    def test_inference_engine_integration(self):
        """Test that server's inference engine works properly"""
        server = ModelServer(self.model, self.tokenizer, self.config)
        
        # Test that the server has an inference engine
        self.assertIsNotNone(server.inference_engine)
        self.assertIsInstance(server.inference_engine, InferenceEngine)


def run_serving_tests():
    """Run all serving tests and report results"""
    test_classes = [
        TestInferenceEngine,
        TestAsyncInferenceEngine,
        TestModelServer
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
    run_serving_tests()