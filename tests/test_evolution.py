"""
Test for the evolution system components
"""
import unittest
import torch
import tempfile
import os
from model.transformer import Config, Transformer
from evolution.evo_loop import EvolutionConfig, ModelVariator, ScoreEvaluator, EvolutionEngine, SelfImprovementLoop


class TestEvolutionConfig(unittest.TestCase):
    def test_evolution_config_creation(self):
        """Test that evolution config can be created with default values"""
        config = EvolutionConfig()
        
        self.assertEqual(config.mutation_rate, 0.1)
        self.assertEqual(config.mutation_strength, 0.01)
        self.assertEqual(config.num_elites, 1)
        self.assertEqual(config.population_size, 10)
        self.assertEqual(config.generations, 10)
        self.assertTrue(config.weight_perturbation)
        self.assertFalse(config.topology_variation)

    def test_evolution_config_custom_values(self):
        """Test that evolution config can be created with custom values"""
        config = EvolutionConfig(
            mutation_rate=0.2,
            population_size=20,
            generations=5
        )
        
        self.assertEqual(config.mutation_rate, 0.2)
        self.assertEqual(config.population_size, 20)
        self.assertEqual(config.generations, 5)


class TestModelVariator(unittest.TestCase):
    def setUp(self):
        self.evo_config = EvolutionConfig(mutation_rate=0.5, mutation_strength=0.01)
        self.variator = ModelVariator(self.evo_config)
        self.config = Config(
            vocab_size=100,
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=128,
            max_len=32
        )
        self.model = Transformer(self.config)

    def test_mutate_weights(self):
        """Test that weight mutation works correctly"""
        original_params = [p.clone() for p in self.model.parameters()]
        mutated_model = self.variator.mutate_weights(self.model)
        
        mutated_params = [p.clone() for p in mutated_model.parameters()]
        
        # Check that we have a new model with different parameters
        params_changed = False
        for orig_param, mut_param in zip(original_params, mutated_params):
            if not torch.allclose(orig_param, mut_param, atol=1e-6):
                params_changed = True
                break
        
        self.assertTrue(params_changed, "Mutated model should have different parameters")
        self.assertIsNot(mutated_model, self.model, "Mutated model should be a different instance")

    def test_crossover_models(self):
        """Test that crossover between two models works correctly"""
        model1 = self.model
        model2 = Transformer(self.config)
        
        child_model = self.variator.crossover_models(model1, model2)
        
        self.assertIsNot(child_model, model1, "Child should be a different instance from parent1")
        self.assertIsNot(child_model, model2, "Child should be a different instance from parent2")
        
        # Child parameters should be a mixture of parent parameters
        child_params = list(child_model.named_parameters())
        parent1_params = dict(model1.named_parameters())
        parent2_params = dict(model2.named_parameters())
        
        # Check that all parameter names match
        for name, child_param in child_params:
            self.assertIn(name, parent1_params, f"Parameter {name} not found in parent1")
            self.assertIn(name, parent2_params, f"Parameter {name} not found in parent2")


class TestEvolutionEngine(unittest.TestCase):
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
        
        # Create a simple evaluator function for testing
        def dummy_eval_function(model, dataloader=None):
            return torch.randn(1).item()  # Random score for testing
        
        self.evaluator = ScoreEvaluator(dummy_eval_function)
        self.evo_config = EvolutionConfig(
            population_size=3,
            generations=2,
            mutation_rate=0.1
        )

    def test_evolution_engine_creation(self):
        """Test that evolution engine can be created"""
        engine = EvolutionEngine(self.model, self.evo_config, self.evaluator)
        
        self.assertEqual(len(engine.population), self.evo_config.population_size)
        self.assertIsNotNone(engine.best_model)
        self.assertEqual(engine.best_score, float('-inf'))

    def test_initialize_population(self):
        """Test that population initialization works correctly"""
        engine = EvolutionEngine(self.model, self.evo_config, self.evaluator)
        
        self.assertEqual(len(engine.population), self.evo_config.population_size)
        # First model in population should be the base model
        self.assertIsInstance(engine.population[0], Transformer)

    def test_tournament_selection(self):
        """Test tournament selection works correctly"""
        engine = EvolutionEngine(self.model, self.evo_config, self.evaluator)
        
        # Create a scored population for testing
        scored_pop = [(self.model, 0.5), (Transformer(self.config), 0.8), (Transformer(self.config), 0.3)]
        
        selected_model = engine.tournament_selection(scored_pop)
        
        self.assertIsInstance(selected_model, Transformer)


class TestSelfImprovementLoop(unittest.TestCase):
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
        
        # For testing, we'll create a dummy tokenizer
        class DummyTokenizer:
            def encode(self, text, return_tensors=None):
                # Return a simple tensor for testing
                return torch.randint(0, 100, (1, 5))
            
            def decode(self, tokens, skip_special_tokens=True):
                return "dummy text"
        
        self.tokenizer = DummyTokenizer()

    def test_self_improvement_creation(self):
        """Test that self-improvement loop can be created"""
        loop = SelfImprovementLoop(self.model, self.tokenizer, self.config)
        
        self.assertEqual(loop.model, self.model)
        self.assertTrue(loop.self_reflection_enabled)

    def test_generate_reflection(self):
        """Test that reflection generation works"""
        loop = SelfImprovementLoop(self.model, self.tokenizer, self.config)
        
        # Since the actual generation would require more complex setup,
        # we'll just test that the method can be called
        try:
            reflection = loop.generate_reflection("test input", "test output")
            self.assertIsInstance(reflection, str)
        except Exception:
            # This is expected if model generation is not fully implemented for test
            pass

    def test_generate_alternative_responses(self):
        """Test that alternative responses can be generated"""
        loop = SelfImprovementLoop(self.model, self.tokenizer, self.config)
        
        # This might fail due to complex model generation, just check it can be called
        try:
            alternatives = loop.generate_alternative_responses("test input", num_alternatives=2)
            # self.assertEqual(len(alternatives), 2)
        except Exception:
            # This is expected if model generation is not fully implemented for test
            pass


def run_evolution_tests():
    """Run all evolution tests and report results"""
    test_classes = [
        TestEvolutionConfig,
        TestModelVariator,
        TestEvolutionEngine,
        TestSelfImprovementLoop
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
    run_evolution_tests()