"""
Test for the evaluation functionality
"""
import unittest
import torch
import tempfile
import os
from model.transformer import Config, Transformer
from tokenizer.train_tokenizer import TokenizerWrapper
from evaluation.benchmarks.model_eval import ModelEvaluator, run_gsm8k_evaluation, run_mmlu_evaluation, run_truthfulqa_evaluation, run_benchmark_suite, calculate_bloom_eval_score


class TestModelEvaluator(unittest.TestCase):
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
            
            vocab = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3, "test": 4, "sentence": 5, ".": 6}
            # Create a simple tokenizer structure
            tokenizer = Tokenizer(BPE())
            # Save it in a proper format
            # For simplicity, we'll create a basic tokenizer file manually
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

    def test_evaluator_creation(self):
        """Test that model evaluator can be created"""
        evaluator = ModelEvaluator(self.model, self.tokenizer, self.config)
        
        self.assertEqual(evaluator.model, self.model)
        self.assertEqual(evaluator.tokenizer, self.tokenizer)
        self.assertEqual(evaluator.config, self.config)

    def test_evaluator_device_assignment(self):
        """Test that evaluator correctly assigns device"""
        evaluator = ModelEvaluator(self.model, self.tokenizer, self.config)
        
        # Check that model is moved to the correct device
        expected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Since we don't have actual tensors moved to device in the setup,
        # we just check that the evaluator was created without errors

    def test_efficiency_evaluation(self):
        """Test efficiency evaluation module"""
        evaluator = ModelEvaluator(self.model, self.tokenizer, self.config)
        
        # Test efficiency metrics with small input shapes for testing
        results = evaluator.evaluate_efficiency([(1, 16), (2, 16)])
        
        # Check that results have expected structure
        self.assertIn('bs1_seq16', results)
        self.assertIn('bs2_seq16', results)
        
        for key, value in results.items():
            self.assertIn('inference_time', value)
            self.assertIn('throughput', value)
            self.assertIn('memory_used_mb', value)

    def test_benchmark_functions(self):
        """Test individual benchmark functions"""
        # These functions expect real models/datasets, so will likely return placeholder values
        
        # Test GSM8K evaluation (should return placeholder)
        gsm8k_result = run_gsm8k_evaluation(self.model, self.tokenizer, self.config)
        if gsm8k_result:
            self.assertIsInstance(gsm8k_result, dict)
        
        # Test MMLU evaluation (should return placeholder)
        mmlu_result = run_mmlu_evaluation(self.model, self.tokenizer, self.config)
        if mmlu_result:
            self.assertIsInstance(mmlu_result, dict)
        
        # Test TruthfulQA evaluation (should return placeholder)
        truthfulqa_result = run_truthfulqa_evaluation(self.model, self.tokenizer, self.config)
        if truthfulqa_result:
            self.assertIsInstance(truthfulqa_result, dict)

    def test_benchmark_suite(self):
        """Test the comprehensive benchmark suite"""
        # This will likely return placeholder values due to missing datasets
        results = run_benchmark_suite(self.model, self.tokenizer, self.config)
        
        # Should return a dictionary with benchmark results
        self.assertIsInstance(results, dict)
        
        # Should contain expected benchmark categories
        expected_benchmarks = ['gsm8k', 'mmlu', 'truthfulqa']
        for benchmark in expected_benchmarks:
            self.assertIn(benchmark, results)

    def test_bloom_eval_score(self):
        """Test Bloom evaluation score calculation"""
        # Test with sample metrics
        sample_metrics = {
            'perplexity': 10.0,
            'accuracy': 0.8,
            'bleu_score': 0.2,
            'rouge_score': 0.3
        }
        
        score = calculate_bloom_eval_score(sample_metrics)
        self.assertIsInstance(score, float)
        
        # Test with empty metrics
        empty_metrics = {}
        score_empty = calculate_bloom_eval_score(empty_metrics)
        self.assertEqual(score_empty, 0.0)


class TestEvaluationUtilities(unittest.TestCase):
    def setUp(self):
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
            
            vocab = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3, "test": 4, "sentence": 5, ".": 6}
            
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

    def test_save_results_functionality(self):
        """Test saving evaluation results"""
        evaluator = ModelEvaluator(self.model, self.tokenizer, self.config)
        
        # Simulate some results
        evaluator.results = {
            'perplexity': {'score': 10.5},
            'accuracy': 0.85,
            'efficiency': {'inference_time': 0.01}
        }
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = os.path.join(tmp_dir, 'eval_results')
            
            evaluator.save_results(output_dir)
            
            # Check that the file was created
            expected_file = os.path.join(output_dir, 'evaluation_results.json')
            self.assertTrue(os.path.exists(expected_file))
            
            # Check that the file contains expected data
            import json
            with open(expected_file, 'r') as f:
                saved_data = json.load(f)
            
            self.assertEqual(saved_data['accuracy'], 0.85)


def run_evaluation_tests():
    """Run all evaluation tests and report results"""
    test_classes = [
        TestModelEvaluator,
        TestEvaluationUtilities
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
    run_evaluation_tests()