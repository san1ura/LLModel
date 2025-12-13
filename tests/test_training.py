"""
Test for the training functionality
"""
import unittest
import torch
import tempfile
import os
import logging
from model.transformer import Config, Transformer
from training.trainer import OptimizedTrainer as Trainer, LionOptimizer
from torch.utils.data import DataLoader, TensorDataset
from tests.logging_config import setup_test_logging, log_test_start, log_test_step, log_test_success, log_test_failure


class TestTrainer(unittest.TestCase):
    def setUp(self):
        setup_test_logging()
        log_test_start("TestTrainer.setUp")

        # Create a small model for testing
        log_test_step("TestTrainer.setUp", "Creating model configuration", {"vocab_size": 100, "d_model": 64, "n_layers": 2})
        self.config = Config(
            vocab_size=100,
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=128,
            max_len=32
        )
        self.model = Transformer(self.config)
        log_test_step("TestTrainer.setUp", "Model created", {"model_type": type(self.model).__name__})

        # Ensure model parameters require gradients for training
        for param in self.model.parameters():
            param.requires_grad = True
        log_test_step("TestTrainer.setUp", "Gradients enabled for model parameters")

        # Create simple dummy dataset
        log_test_step("TestTrainer.setUp", "Creating dummy dataset", {"sample_count": 10, "seq_len": 32})
        input_ids = torch.randint(0, 100, (10, 32), dtype=torch.long)  # 10 samples, 32 tokens each
        labels = torch.randint(0, 100, (10, 32), dtype=torch.long)

        # Create a custom dataloader that returns dict format expected by trainer
        class DictDataLoader:
            def __init__(self, input_ids, labels, batch_size):
                self.input_ids = input_ids
                self.labels = labels
                self.batch_size = batch_size
                self.len = len(input_ids) // batch_size
                log_test_step("DictDataLoader.__init__", "DataLoader created", {"batch_size": batch_size, "dataset_size": len(input_ids)})

            def __iter__(self):
                log_test_step("DictDataLoader.__iter__", "Creating DataLoader iterator")
                for i in range(0, len(self.input_ids), self.batch_size):
                    input_batch = self.input_ids[i:i + self.batch_size]
                    label_batch = self.labels[i:i + self.batch_size]
                    log_test_step("DictDataLoader.__iter__", "Batch created", {"batch_idx": i//self.batch_size, "batch_size": len(input_batch)})
                    yield {"input_ids": input_batch, "labels": label_batch}

            def __len__(self):
                return self.len

        self.dataloader = DictDataLoader(input_ids, labels, batch_size=2)
        log_test_success("TestTrainer.setUp", "Setup completed")

    def test_trainer_creation(self):
        """Test that trainer can be created"""
        log_test_start("TestTrainer.test_trainer_creation")

        try:
            log_test_step("TestTrainer.test_trainer_creation", "Creating trainer",
                        {"model_type": type(self.model).__name__, "config": self.config.__dict__})
            trainer = Trainer(
                model=self.model,
                config=self.config,
                train_data=self.dataloader,
                eval_data=self.dataloader
            )
            log_test_step("TestTrainer.test_trainer_creation", "Trainer created",
                        {"trainer_type": type(trainer).__name__, "optimizer_type": type(trainer.optimizer).__name__})

            log_test_step("TestTrainer.test_trainer_creation", "Checking model")
            self.assertEqual(trainer.model, self.model)
            log_test_step("TestTrainer.test_trainer_creation", "Checking config")
            self.assertEqual(trainer.config, self.config)
            log_test_step("TestTrainer.test_trainer_creation", "Checking optimizer")
            self.assertIsNotNone(trainer.optimizer)
            log_test_step("TestTrainer.test_trainer_creation", "Checking scheduler")
            self.assertIsNotNone(trainer.scheduler)

            log_test_success("TestTrainer.test_trainer_creation", "All verifications successful")
        except Exception as e:
            log_test_failure("TestTrainer.test_trainer_creation", str(e))
            raise

    def test_trainer_optimizer(self):
        """Test trainer with different optimizers"""
        log_test_start("TestTrainer.test_trainer_optimizer")

        try:
            # Test with AdamW
            log_test_step("TestTrainer.test_trainer_optimizer", "Starting AdamW optimizer test")
            trainer_adamw = Trainer(
                model=self.model,
                config=self.config,
                train_data=self.dataloader,
                optimizer_name="adamw"
            )
            log_test_step("TestTrainer.test_trainer_optimizer", "AdamW trainer created",
                        {"optimizer_type": type(trainer_adamw.optimizer).__name__})
            self.assertIsInstance(trainer_adamw.optimizer, torch.optim.Optimizer)
            log_test_step("TestTrainer.test_trainer_optimizer", "AdamW optimizer verified")

            # Test with SGD
            log_test_step("TestTrainer.test_trainer_optimizer", "Starting SGD optimizer test")
            trainer_sgd = Trainer(
                model=self.model,
                config=self.config,
                train_data=self.dataloader,
                optimizer_name="sgd"
            )
            log_test_step("TestTrainer.test_trainer_optimizer", "SGD trainer created",
                        {"optimizer_type": type(trainer_sgd.optimizer).__name__})
            self.assertIsInstance(trainer_sgd.optimizer, torch.optim.SGD)
            log_test_step("TestTrainer.test_trainer_optimizer", "SGD optimizer verified")

            # Test with Lion
            log_test_step("TestTrainer.test_trainer_optimizer", "Starting Lion optimizer test")
            trainer_lion = Trainer(
                model=self.model,
                config=self.config,
                train_data=self.dataloader,
                optimizer_name="lion"
            )
            log_test_step("TestTrainer.test_trainer_optimizer", "Lion trainer created",
                        {"optimizer_type": type(trainer_lion.optimizer).__name__ if trainer_lion.optimizer else "None"})
            # Lion optimizer should be created
            self.assertIsNotNone(trainer_lion.optimizer)
            log_test_step("TestTrainer.test_trainer_optimizer", "Lion optimizer verified")

            log_test_success("TestTrainer.test_trainer_optimizer", "All optimizer tests successful")
        except Exception as e:
            log_test_failure("TestTrainer.test_trainer_optimizer", str(e))
            raise

    def test_compute_loss(self):
        """Test loss computation"""
        log_test_start("TestTrainer.test_compute_loss")

        try:
            log_test_step("TestTrainer.test_compute_loss", "Creating trainer")
            trainer = Trainer(
                model=self.model,
                config=self.config,
                train_data=self.dataloader
            )
            log_test_step("TestTrainer.test_compute_loss", "Trainer created")

            # Create some dummy logits and labels
            log_test_step("TestTrainer.test_compute_loss", "Creating dummy logits and labels",
                        {"batch_size": 2, "seq_len": 10, "vocab_size": 100})
            batch_size, seq_len, vocab_size = 2, 10, 100
            logits = torch.randn(batch_size, seq_len, vocab_size)
            labels = torch.randint(0, vocab_size, (batch_size, seq_len))
            log_test_step("TestTrainer.test_compute_loss", "Logits and labels created",
                        {"logits_shape": list(logits.shape), "labels_shape": list(labels.shape)})

            log_test_step("TestTrainer.test_compute_loss", "Computing loss")
            loss = trainer.compute_loss(logits, labels)
            log_test_step("TestTrainer.test_compute_loss", "Loss computed", {"loss_value": loss.item()})

            log_test_step("TestTrainer.test_compute_loss", "Checking loss type")
            self.assertIsInstance(loss, torch.Tensor)
            log_test_step("TestTrainer.test_compute_loss", "Checking loss is positive")
            self.assertGreaterEqual(loss.item(), 0)  # Loss should be non-negative

            log_test_success("TestTrainer.test_compute_loss", "Loss computation test successful")
        except Exception as e:
            log_test_failure("TestTrainer.test_compute_loss", str(e))
            raise

    def test_train_step(self):
        """Test a single training step"""
        log_test_start("TestTrainer.test_train_step")

        try:
            log_test_step("TestTrainer.test_train_step", "Creating trainer", {"device": "cpu"})
            trainer = Trainer(
                model=self.model,
                config=self.config,
                train_data=self.dataloader,
                device="cpu"  # Use CPU for testing
            )
            log_test_step("TestTrainer.test_train_step", "Trainer created",
                        {"trainer_device": str(trainer.device)})

            # Get a batch from dataloader
            log_test_step("TestTrainer.test_train_step", "Fetching batch data")
            batch_count = 0
            for batch_dict in self.dataloader:
                log_test_step("TestTrainer.test_train_step", "Batch fetched",
                            {"batch_keys": list(batch_dict.keys()),
                             "input_ids_shape": list(batch_dict["input_ids"].shape),
                             "labels_shape": list(batch_dict["labels"].shape)})

                # Perform a training step
                log_test_step("TestTrainer.test_train_step", "Starting train step")
                loss = trainer.train_step(batch_dict)
                log_test_step("TestTrainer.test_train_step", "Train step completed",
                            {"loss": loss})

                log_test_step("TestTrainer.test_train_step", "Checking loss type")
                self.assertIsInstance(loss, float)
                log_test_step("TestTrainer.test_train_step", "Checking loss is positive")
                self.assertGreaterEqual(loss, 0)

                batch_count += 1
                if batch_count >= 1:
                    log_test_step("TestTrainer.test_train_step", "Only one batch tested")
                    break  # Only test one batch

            log_test_success("TestTrainer.test_train_step", "Train step test successful")
        except Exception as e:
            log_test_failure("TestTrainer.test_train_step", str(e))
            raise

    def test_optimizer_step(self):
        """Test optimizer step functionality"""
        trainer = Trainer(
            model=self.model,
            config=self.config,
            train_data=self.dataloader,
            device="cpu"
        )

        # Perform a training step first to accumulate gradients
        for batch_dict in self.dataloader:
            # Perform training step
            _ = trainer.train_step(batch_dict)

            # Perform optimizer step
            trainer.optimizer_step()
            break

    def test_evaluate(self):
        """Test evaluation functionality"""
        trainer = Trainer(
            model=self.model,
            config=self.config,
            train_data=self.dataloader,
            eval_data=self.dataloader,
            device="cpu"
        )

        # Perform evaluation
        eval_loss = trainer.evaluate()
        
        # Should return a numeric loss value
        self.assertIsInstance(eval_loss, float)
        self.assertGreaterEqual(eval_loss, 0)  # Loss should be non-negative


class TestLionOptimizer(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        )
        self.params = list(self.model.parameters())

    def test_lion_optimizer_creation(self):
        """Test Lion optimizer creation"""
        # This is a custom Lion implementation from the training module
        optimizer = LionOptimizer(self.params, lr=1e-3)
        
        self.assertIsInstance(optimizer, LionOptimizer)
        self.assertEqual(optimizer.defaults['lr'], 1e-3)

    def test_lion_optimizer_step(self):
        """Test Lion optimizer step"""
        optimizer = LionOptimizer(self.params, lr=1e-4)
        
        # Simple forward and backward pass
        x = torch.randn(2, 10)
        y = torch.randn(2, 1)
        loss_fn = torch.nn.MSELoss()
        
        output = self.model(x)
        loss = loss_fn(output, y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


class TestTrainingUtilities(unittest.TestCase):
    def setUp(self):
        self.config = Config(
            vocab_size=100,
            d_model=32,
            n_layers=1,
            n_heads=2,
            d_ff=64,
            max_len=16
        )
        self.model = Transformer(self.config)

        # Create simple dummy dataset
        input_ids = torch.randint(0, 100, (4, 16))
        labels = torch.randint(0, 100, (4, 16))
        dataset = TensorDataset(input_ids, labels)
        self.dataloader = DataLoader(dataset, batch_size=2)

    def test_scheduler_types(self):
        """Test different scheduler types in trainer"""
        scheduler_types = ["cosine", "linear", "constant"]

        for sched_type in scheduler_types:
            with self.subTest(scheduler_type=sched_type):
                trainer = Trainer(
                    model=self.model,
                    config=self.config,
                    train_data=self.dataloader,
                    eval_data=self.dataloader,
                    lr_schedule_type=sched_type
                )
                
                # Trainer should initialize without error for each scheduler type
                self.assertIsNotNone(trainer)

    def test_save_checkpoint(self):
        """Test checkpoint saving functionality"""
        trainer = Trainer(
            model=self.model,
            config=self.config,
            train_data=self.dataloader,
            save_dir=tempfile.mkdtemp(),
            device="cpu"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, "test_checkpoint.pt")
            
            # Save checkpoint
            trainer.save_checkpoint(step=10, eval_loss=0.5)
            
            # Check that checkpoint was created
            expected_path = os.path.join(trainer.save_dir, "checkpoint_10.pt")
            self.assertTrue(os.path.exists(expected_path))

    def test_gradient_accumulation(self):
        """Test gradient accumulation functionality"""
        trainer = Trainer(
            model=self.model,
            config=self.config,
            train_data=self.dataloader,
            gradient_accumulation_steps=2,  # Accumulate over 2 steps
            device="cpu"
        )

        # Simulate training with gradient accumulation
        accumulated_loss = 0
        for i, batch_dict in enumerate(self.dataloader):

            loss = trainer.train_step(batch_dict)
            accumulated_loss += loss

            # Only update optimizer every 2 steps
            if (i + 1) % trainer.gradient_accumulation_steps == 0:
                trainer.optimizer_step()
                break  # Just test one accumulation cycle


def run_training_tests():
    """Run all training tests and report results"""
    test_classes = [
        TestTrainer,
        TestLionOptimizer,
        TestTrainingUtilities
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
    run_training_tests()