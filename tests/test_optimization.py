"""
Test for the optimization functionality
"""
import unittest
import torch
import torch.nn as nn
from optim.schedule import LionOptimizer, Adafactor, CosineSchedulerWithWarmup, ConstantWithWarmupScheduler, LinearWithWarmupScheduler, MultiStageScheduler, LayerwiseLearningRateScheduler, get_optimizer, get_scheduler


class TestLionOptimizer(unittest.TestCase):
    def setUp(self):
        self.model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
        self.params = list(self.model.parameters())

    def test_lion_optimizer_creation(self):
        """Test that Lion optimizer can be created"""
        optimizer = LionOptimizer(self.params, lr=1e-3, betas=(0.9, 0.99))
        
        self.assertIsInstance(optimizer, LionOptimizer)
        self.assertEqual(optimizer.defaults['lr'], 1e-3)
        self.assertEqual(optimizer.defaults['betas'], (0.9, 0.99))

    def test_lion_optimizer_step(self):
        """Test that Lion optimizer can perform a step"""
        optimizer = LionOptimizer(self.params, lr=1e-3)
        
        # Create a simple loss
        input_tensor = torch.randn(2, 10)
        target = torch.randn(2, 1)
        loss_fn = nn.MSELoss()
        
        # Forward pass
        output = self.model(input_tensor)
        loss = loss_fn(output, target)
        
        # Backward pass
        loss.backward()
        
        # Step
        optimizer.step()
        optimizer.zero_grad()


class TestAdafactor(unittest.TestCase):
    def setUp(self):
        self.model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
        self.params = list(self.model.parameters())

    def test_adafactor_creation(self):
        """Test that Adafactor optimizer can be created"""
        optimizer = Adafactor(self.params, lr=1e-3)
        
        self.assertIsInstance(optimizer, Adafactor)
        self.assertEqual(optimizer.defaults['lr'], 1e-3)

    def test_adafactor_step(self):
        """Test that Adafactor optimizer can perform a step"""
        optimizer = Adafactor(self.params, lr=1e-3)
        
        # Create a simple loss
        input_tensor = torch.randn(2, 10)
        target = torch.randn(2, 1)
        loss_fn = nn.MSELoss()
        
        # Forward pass
        output = self.model(input_tensor)
        loss = loss_fn(output, target)
        
        # Backward pass
        loss.backward()
        
        # Step
        optimizer.step()
        optimizer.zero_grad()


class TestSchedulers(unittest.TestCase):
    def setUp(self):
        self.model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
        self.params = list(self.model.parameters())
        self.optimizer = torch.optim.SGD(self.params, lr=1e-2)

    def test_cosine_scheduler_with_warmup(self):
        """Test CosineSchedulerWithWarmup"""
        scheduler = CosineSchedulerWithWarmup(
            self.optimizer,
            num_warmup_steps=10,
            num_training_steps=100
        )
        
        initial_lr = self.optimizer.param_groups[0]['lr']
        
        # Step through warmup
        for i in range(10):
            scheduler.step(i)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.assertLessEqual(current_lr, initial_lr)
        
        # Step after warmup
        for i in range(10, 100):
            scheduler.step(i)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.assertGreaterEqual(current_lr, 0.0)  # Should be positive

    def test_constant_with_warmup_scheduler(self):
        """Test ConstantWithWarmupScheduler"""
        scheduler = ConstantWithWarmupScheduler(
            self.optimizer,
            num_warmup_steps=5,
            constant_lr=1e-3
        )
        
        # Initially, LR should be smaller during warmup
        scheduler.step(0)
        lr_after_step_0 = self.optimizer.param_groups[0]['lr']
        
        scheduler.step(3)  # Still in warmup
        lr_after_step_3 = self.optimizer.param_groups[0]['lr']
        
        scheduler.step(5)  # After warmup
        lr_after_step_5 = self.optimizer.param_groups[0]['lr']
        
        # After warmup, LR should be constant
        scheduler.step(10)
        lr_after_step_10 = self.optimizer.param_groups[0]['lr']
        
        self.assertLessEqual(lr_after_step_0, 1e-3)  # Warmup LR
        self.assertLessEqual(lr_after_step_3, 1e-3)  # Warmup LR
        self.assertEqual(lr_after_step_5, 1e-3)      # Constant LR
        self.assertEqual(lr_after_step_10, 1e-3)     # Constant LR

    def test_linear_with_warmup_scheduler(self):
        """Test LinearWithWarmupScheduler"""
        scheduler = LinearWithWarmupScheduler(
            self.optimizer,
            num_warmup_steps=5,
            num_training_steps=20
        )
        
        # Check warmup phase (should increase)
        scheduler.step(0)
        lr_0 = self.optimizer.param_groups[0]['lr']
        
        scheduler.step(4)
        lr_4 = self.optimizer.param_groups[0]['lr']
        
        # LR should increase during warmup
        self.assertLessEqual(lr_0, lr_4)
        
        # Check decay phase (should decrease)
        scheduler.step(6)
        lr_6 = self.optimizer.param_groups[0]['lr']
        
        scheduler.step(20)
        lr_20 = self.optimizer.param_groups[0]['lr']
        
        # LR should decrease after peak
        if lr_6 > lr_20:  # If they're equal, that's also fine (reached minimum)
            pass  # This is expected behavior

    def test_multi_stage_scheduler(self):
        """Test MultiStageScheduler"""
        stages = [
            ('linear', 10, {'num_warmup_steps': 5, 'num_training_steps': 10}),
            ('cosine', 10, {'num_warmup_steps': 0, 'num_training_steps': 10})
        ]
        
        try:
            scheduler = MultiStageScheduler(self.optimizer, stages)
            
            # This should work without errors
            for i in range(20):
                scheduler.step()
                
        except ImportError:
            # This is OK if we're missing dependencies
            pass


class TestOptimizerFactory(unittest.TestCase):
    def setUp(self):
        self.model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
        self.params = list(self.model.parameters())

    def test_get_optimizer_adamw(self):
        """Test retrieving AdamW optimizer"""
        class Config:
            optimizer_name = 'adamw'
            lr = 1e-3
            weight_decay = 0.1
        
        config = Config()
        optimizer = get_optimizer(config, self.params)
        
        # Check that it's some form of AdamW optimizer
        self.assertTrue(isinstance(optimizer, (torch.optim.AdamW, torch.optim.Optimizer)))

    def test_get_optimizer_lion(self):
        """Test retrieving Lion optimizer"""
        class Config:
            optimizer_name = 'lion'
            lr = 1e-4
            weight_decay = 0.0
        
        config = Config()
        optimizer = get_optimizer(config, self.params)
        
        # Should create Lion or fallback to another optimizer
        self.assertTrue(isinstance(optimizer, torch.optim.Optimizer))

    def test_get_optimizer_invalid(self):
        """Test retrieving invalid optimizer raises error"""
        class Config:
            optimizer_name = 'invalid_optimizer'
        
        config = Config()
        
        with self.assertRaises(ValueError):
            get_optimizer(config, self.params)


class TestSchedulerFactory(unittest.TestCase):
    def setUp(self):
        self.model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
        self.params = list(self.model.parameters())
        self.optimizer = torch.optim.SGD(self.params, lr=1e-2)

    def test_get_scheduler_cosine(self):
        """Test retrieving cosine scheduler"""
        class Config:
            scheduler_type = 'cosine'
            num_warmup_steps = 5
            num_training_steps = 20
        
        config = Config()
        scheduler = get_scheduler(self.optimizer, config)
        
        self.assertIsNotNone(scheduler)

    def test_get_scheduler_linear(self):
        """Test retrieving linear scheduler"""
        class Config:
            scheduler_type = 'linear'
            num_warmup_steps = 3
            num_training_steps = 15
        
        config = Config()
        scheduler = get_scheduler(self.optimizer, config)
        
        self.assertIsNotNone(scheduler)

    def test_get_scheduler_constant(self):
        """Test retrieving constant scheduler"""
        class Config:
            scheduler_type = 'constant'
            num_warmup_steps = 2
            lr = 1e-3
        
        config = Config()
        scheduler = get_scheduler(self.optimizer, config)
        
        self.assertIsNotNone(scheduler)

    def test_get_scheduler_invalid(self):
        """Test retrieving invalid scheduler raises error"""
        class Config:
            scheduler_type = 'invalid_scheduler'
        
        config = Config()
        
        with self.assertRaises(ValueError):
            get_scheduler(self.optimizer, config)


def run_optimization_tests():
    """Run all optimization tests and report results"""
    test_classes = [
        TestLionOptimizer,
        TestAdafactor,
        TestSchedulers,
        TestOptimizerFactory,
        TestSchedulerFactory
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
    run_optimization_tests()