"""
Training pipeline for the transformer model
Supports pretraining, SFT, RLHF, and other training approaches
"""

import json
import logging
import math
import os
import time
from typing import Any, Callable, Dict, Optional

import torch
import torch.amp.grad_scaler as GradScaler
import torch.nn as nn
import torch.optim as optim
# from torch.amp import GradScaler, autocast
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from model.layers.kv_cache import KVCacheManager
from model.transformer import Config, Transformer

try:
    from optim.fused_ops import FusedAdamW, FusedLion
except ImportError:
    FusedLion = None
    FusedAdamW = None


class Trainer:
    """
    Main trainer class supporting various training approaches
    """

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        train_data: DataLoader,
        eval_data: Optional[DataLoader] = None,
        optimizer_name: str = "adamw",
        lr: float = 1e-5,  # Reduced learning rate to prevent overfitting
        weight_decay: float = 0.1,
        warmup_steps: int = 500,  # Reduced warmup steps for smaller datasets
        total_steps: int = 5000,  # Reduced total steps to prevent overfitting
        log_interval: int = 10,
        save_interval: int = 500,  # More frequent saving for monitoring
        eval_interval: int = 500,  # More frequent evaluation to catch overfitting
        save_dir: str = "./checkpoints",
        device: str = "cuda",
        use_amp: bool = True,
        gradient_accumulation_steps: int = 4,  # Increased for small batch sizes
        max_grad_norm: float = 1.0,
        lr_schedule_type: str = "cosine",  # 'cosine', 'linear', 'constant'
        early_stopping_patience: int = 3,  # Early stopping to prevent overfitting
        early_stopping_threshold: float = 1e-4,  # Threshold for improvement
        validation_split: float = 0.1,  # Fraction of train data to use for validation
    ):
        self.model = model
        self.config = config
        self.train_data = train_data
        self.eval_data = eval_data
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.save_dir = save_dir
        self.device = device
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.lr_schedule_type = lr_schedule_type

        # Setup device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

        # Ensure model parameters require gradients
        for param in self.model.parameters():
            param.requires_grad = True

        # Setup AMP scaler
        # Only use AMP if on CUDA
        if use_amp and str(self.device) == "cuda":
            from torch.amp import GradScaler
            self.scaler = GradScaler()
        else:
            self.scaler = None
            self.use_amp = False  # Disable AMP if not using CUDA

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Setup logging
        self._setup_logging()

        # Training state
        self.global_step = 0
        self.best_eval_loss = float("inf")
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.epochs_without_improvement = 0
        self.validation_split = validation_split

        # If eval_data is not provided but validation_split is specified, split the train data
        if eval_data is None and validation_split > 0:
            self._split_train_eval(validation_split)

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

    def _split_train_eval(self, validation_split: float):
        """Split train dataset into train and validation sets"""
        if hasattr(self.train_data, 'dataset'):
            dataset = self.train_data.dataset
            total_size = len(dataset)
            val_size = int(total_size * validation_split)
            train_size = total_size - val_size

            # Split the dataset
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            # Create new data loaders with the same parameters as the original
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.train_data.batch_size,
                shuffle=True,
                num_workers=self.train_data.num_workers if hasattr(self.train_data, 'num_workers') else 0,
                collate_fn=self.train_data.collate_fn if hasattr(self.train_data, 'collate_fn') else None,
                pin_memory=self.train_data.pin_memory if hasattr(self.train_data, 'pin_memory') else False
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.train_data.batch_size,
                shuffle=False,
                num_workers=self.train_data.num_workers if hasattr(self.train_data, 'num_workers') else 0,
                collate_fn=self.train_data.collate_fn if hasattr(self.train_data, 'collate_fn') else None,
                pin_memory=self.train_data.pin_memory if hasattr(self.train_data, 'pin_memory') else False
            )

            # Update train and eval data loaders
            self.train_data = train_loader
            self.eval_data = val_loader
            self.logger.info(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation samples")

    def _create_optimizer(self):
        """Create optimizer based on specified name"""
        if self.optimizer_name.lower() == "adamw":
            # Use fused AdamW if available, otherwise standard
            if FusedAdamW is not None:
                return FusedAdamW(
                    self.model.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                )
            try:
                from apex.optimizers import FusedAdam

                return FusedAdam(
                    self.model.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                )
            except ImportError:
                return optim.AdamW(
                    self.model.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                )
        elif self.optimizer_name.lower() == "lion":
            # Use FusedLion if available
            if FusedLion is not None:
                return FusedLion(
                    self.model.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    betas=(0.9, 0.95),
                )
            # Lion implementation
            return LionOptimizer(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.95),
            )
        elif self.optimizer_name.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

    def _create_scheduler(self):
        """Create learning rate scheduler"""
        total_steps = self.total_steps
        warmup_steps = self.warmup_steps

        if self.lr_schedule_type == "cosine":
            try:
                from transformers.optimization import \
                    get_cosine_schedule_with_warmup

                return get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps,
                )
            except ImportError:
                # Fallback to PyTorch scheduler if transformers is not available
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=total_steps - warmup_steps, eta_min=0.0
                )
                # For cosine schedule with warmup, we need to combine schedulers
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=1e-6,
                    end_factor=1.0,
                    total_iters=warmup_steps,
                )
                from torch.optim.lr_scheduler import SequentialLR

                return SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, scheduler],
                    milestones=[warmup_steps],
                )
        elif self.lr_schedule_type == "linear":
            try:
                from transformers.optimization import \
                    get_linear_schedule_with_warmup

                return get_linear_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps,
                )
            except ImportError:
                # Fallback to PyTorch scheduler if transformers is not available
                scheduler = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=1.0,
                    end_factor=0.0,
                    total_iters=total_steps - warmup_steps,
                )
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=1e-6,
                    end_factor=1.0,
                    total_iters=warmup_steps,
                )
                from torch.optim.lr_scheduler import SequentialLR

                return SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, scheduler],
                    milestones=[warmup_steps],
                )
        elif self.lr_schedule_type == "constant":
            return optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
        else:
            raise ValueError(f"Unsupported scheduler type: {self.lr_schedule_type}")

    def _setup_logging(self):
        """Setup training logging"""
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(self.save_dir, "training.log")),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def compute_loss(self, logits, labels):
        """Compute cross-entropy loss with proper masking"""
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        return loss

    def train_step(self, batch):
        """Perform a single training step"""
        self.model.train()

        # Handle both dict and tuple/list formats for batch
        if isinstance(batch, dict):
            # Dict format: {'input_ids': ..., 'labels': ...}
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            attention_mask = batch.get("attention_mask", None)
        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
            # Tuple/list format: (input_ids, labels, [attention_mask])
            input_ids = batch[0].to(self.device)
            labels = batch[1].to(self.device)
            attention_mask = batch[2].to(self.device) if len(batch) > 2 else None
        else:
            raise ValueError(f"Unsupported batch format: {type(batch)}")

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Forward pass
        if self.use_amp and str(self.device) == "cuda":
            with autocast("cuda"):
                logits, _ = self.model(input_ids, mask=attention_mask)
                loss = self.compute_loss(logits, labels)
        else:
            # When not using AMP, we still need gradients for backpropagation
            logits, _ = self.model(input_ids, mask=attention_mask)
            loss = self.compute_loss(logits, labels)

        # Scale loss and backward
        if self.use_amp and str(self.device) == "cuda":
            loss = loss / self.gradient_accumulation_steps
            self.scaler.scale(loss).backward()
        else:
            (loss / self.gradient_accumulation_steps).backward()

        return loss.item()

    def optimizer_step(self):
        """Perform optimizer step with gradient clipping"""
        if self.use_amp:
            # Gradient clipping with AMP
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

        # Update learning rate
        if self.scheduler:
            self.scheduler.step()

        # Zero gradients
        self.optimizer.zero_grad()

    def evaluate(self):
        """Evaluate the model on the eval dataset"""
        if self.eval_data is None:
            return float("inf")

        self.model.eval()
        total_loss = 0
        total_tokens = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.eval_data:
                # Handle both dict and tuple/list formats for batch
                if isinstance(batch, dict):
                    # Dict format: {'input_ids': ..., 'labels': ...}
                    input_ids = batch["input_ids"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    attention_mask = batch.get("attention_mask", None)
                elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    # Tuple/list format: (input_ids, labels, [attention_mask])
                    input_ids = batch[0].to(self.device)
                    labels = batch[1].to(self.device)
                    attention_mask = batch[2].to(self.device) if len(batch) > 2 else None
                else:
                    raise ValueError(f"Unsupported batch format: {type(batch)}")

                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                if self.use_amp and str(self.device) == "cuda":
                    with autocast("cuda"):
                        logits, _ = self.model(input_ids, mask=attention_mask)
                        loss = self.compute_loss(logits, labels)
                else:
                    # For evaluation we don't need gradients in the typical case, but in training evaluation
                    # we might need them for some computations
                    logits, _ = self.model(input_ids, mask=attention_mask)
                    loss = self.compute_loss(logits, labels)

                # Calculate metrics
                total_loss += loss.item()

                # Count non-ignored tokens for perplexity calculation
                non_ignore_mask = labels != -100
                num_non_ignore_tokens = non_ignore_mask.sum().item()
                total_tokens += num_non_ignore_tokens

                num_batches += 1

                # Clear variables to free memory
                del logits, loss

        avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")

        # Calculate perplexity
        if total_tokens > 0:
            avg_loss_per_token = total_loss / total_tokens
            perplexity = math.exp(avg_loss_per_token)
        else:
            perplexity = float("inf")

        # Log evaluation metrics
        self.logger.info(
            f"Evaluation completed. Avg Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}, Tokens: {total_tokens}"
        )

        return avg_loss

    def save_checkpoint(self, step: int, eval_loss: float = None):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_{step}.pt")

        checkpoint = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "config": self.config.to_dict(),
            "eval_loss": eval_loss,
            "global_step": self.global_step,
        }

        torch.save(checkpoint, checkpoint_path)

        # Save the best model
        if eval_loss is not None and eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            best_path = os.path.join(self.save_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with eval loss: {eval_loss:.4f}")

    def train(self, epochs: int = None):
        """Main training loop"""
        self.logger.info("Starting training...")

        # Calculate total steps based on dataset size and epochs
        if hasattr(self.train_data, "dataset") and hasattr(
            self.train_data.dataset, "__len__"
        ):
            dataset_size = len(self.train_data.dataset)
        else:
            # Estimate dataset size by iterating once (with a reasonable limit)
            dataset_size = 0
            for _ in self.train_data:
                dataset_size += 1
                if dataset_size >= 500000:  # Limit estimation to avoid long waits
                    break
            # Need to recreate the dataloader since we've exhausted the iterator
            self.logger.warning(
                f"Dataset size estimation reached limit. Using {dataset_size} as estimate."
            )

        # Calculate steps per epoch based on dataset size and batch size
        if hasattr(self.train_data, "batch_size"):
            steps_per_epoch = max(1, dataset_size // self.train_data.batch_size)
        else:
            steps_per_epoch = max(
                1, dataset_size // 4
            )  # default batch size of 4 if not set

        if epochs is not None:
            total_steps = epochs * steps_per_epoch
        else:
            # If no epochs specified, use the existing total_steps
            total_steps = self.total_steps
            epochs = total_steps // steps_per_epoch if steps_per_epoch > 0 else 1

        self.logger.info(
            f"Training for {total_steps} steps, {epochs} epochs, {steps_per_epoch} steps per epoch"
        )

        # Initialize gradient accumulation
        self.optimizer.zero_grad()

        # Create training iterator
        train_iter = iter(self.train_data)

        completed_steps = 0
        epoch_step = 0

        # Create a global progress bar
        global_pbar = tqdm(total=total_steps, desc="Training Progress", unit="step")

        for epoch in range(epochs if epochs is not None else 2):
            epoch_step = 0

            # Process one epoch
            while epoch_step < steps_per_epoch:
                try:
                    # Get batch
                    try:
                        batch = next(train_iter)
                        epoch_step += 1
                        completed_steps += 1
                    except StopIteration:
                        # Reset iterator if we reach the end of dataset
                        train_iter = iter(self.train_data)
                        try:
                            batch = next(train_iter)
                            epoch_step += 1
                            completed_steps += 1
                        except StopIteration:
                            # No more data, break
                            break

                    # Training step
                    loss = self.train_step(batch)

                    # Perform optimizer step
                    if (completed_steps + 1) % self.gradient_accumulation_steps == 0:
                        self.optimizer_step()
                        self.global_step += 1

                    # Update progress bar
                    global_pbar.set_postfix(
                        {
                            "Epoch": f"{epoch + 1}/{epochs}",
                            "Step": f"{self.global_step}",
                            "Loss": f"{loss:.4f}",
                            "LR": f"{self.scheduler.get_last_lr()[0] if self.scheduler else self.lr:.2e}",
                            "Mem": (
                                f"{torch.cuda.memory_allocated(self.device) / 1024**3:.1f}GB"
                                if torch.cuda.is_available()
                                and str(self.device) == "cuda"
                                else "N/A"
                            ),
                        }
                    )
                    global_pbar.update(1)

                    # Detailed logging
                    if self.global_step % self.log_interval == 0:
                        current_lr = (
                            self.scheduler.get_last_lr()[0]
                            if self.scheduler
                            else self.lr
                        )
                        detailed_info = (
                            f"Epoch: {epoch + 1:2d}/{epochs:2d}, "
                            f"Step: {self.global_step:5d}, "
                            f"Batch Loss: {loss:.6f}, "
                            f"LR: {current_lr:.2e}, "
                            f"Accum Steps: {self.gradient_accumulation_steps}, "
                        )

                        # Add memory usage if using CUDA
                        if torch.cuda.is_available() and str(self.device) == "cuda":
                            mem_allocated = (
                                torch.cuda.memory_allocated(self.device) / 1024**3
                            )
                            mem_reserved = (
                                torch.cuda.memory_reserved(self.device) / 1024**3
                            )
                            detailed_info += (
                                f"GPU Mem: {mem_allocated:.1f}/{mem_reserved:.1f}GB"
                            )

                        self.logger.info(detailed_info)

                    # Evaluation
                    if self.eval_data and self.global_step % self.eval_interval == 0:
                        self.logger.info(
                            f"Starting evaluation at step {self.global_step}..."
                        )
                        eval_loss = self.evaluate()
                        eval_info = (
                            f"Step {self.global_step:5d}, Eval Loss: {eval_loss:.6f}"
                        )
                        self.logger.info(eval_info)

                        # Generate sample text to evaluate quality
                        self.generate_sample_text()

                        # Early stopping check
                        if eval_loss < self.best_eval_loss - self.early_stopping_threshold:
                            self.best_eval_loss = eval_loss
                            self.epochs_without_improvement = 0
                            # Save checkpoint
                            self.save_checkpoint(self.global_step, eval_loss)
                        else:
                            self.epochs_without_improvement += 1
                            self.logger.info(
                                f"No improvement for {self.epochs_without_improvement} evaluation intervals. "
                                f"Best eval loss: {self.best_eval_loss:.6f}"
                            )

                        # Check for early stopping
                        if self.epochs_without_improvement >= self.early_stopping_patience:
                            self.logger.info(
                                f"Early stopping triggered after {self.epochs_without_improvement} "
                                f"evaluation intervals without improvement."
                            )
                            global_pbar.close()
                            return

                    # Save checkpoint periodically
                    elif self.global_step % self.save_interval == 0:
                        self.save_checkpoint(self.global_step)

                    # Break if we've reached the desired number of steps
                    if completed_steps >= total_steps:
                        self.logger.info(
                            f"Reached target {total_steps} steps. Training completed."
                        )
                        break

                except KeyboardInterrupt:
                    self.logger.info("Training interrupted by user")
                    global_pbar.close()
                    self.save_checkpoint(self.global_step)
                    return
                except Exception as e:
                    self.logger.error(
                        f"Error during training step {completed_steps}: {str(e)}"
                    )
                    global_pbar.close()
                    raise e

            if completed_steps >= total_steps:
                break

        global_pbar.close()
        self.logger.info("Training completed!")

    def generate_sample_text(self):
        """Generate sample text to evaluate model quality during training"""
        try:
            # Create a simple prompt for generation
            sample_prompt = "The future of artificial intelligence"

            # Try to tokenize the prompt if tokenizer is available
            if hasattr(self, "tokenizer") and self.tokenizer is not None:
                from tokenizer.train_tokenizer import TokenizerWrapper

                # Use the model's vocab size to create a simple prompt
                prompt_tokens = (
                    [self.config.bos_token_id]
                    if hasattr(self.config, "bos_token_id")
                    else []
                )
                prompt_tokens.extend(
                    [50256, 1000, 2000, 3000]
                )  # Use some common token IDs as example
            else:
                # Use simple token IDs based on vocab size
                prompt_tokens = [1, 2, 3, 4, 5]  # Simple example tokens

            # Convert to tensor
            input_ids = torch.tensor(
                [prompt_tokens], dtype=torch.long, device=self.device
            )

            # Generate text
            self.model.eval()  # Set to evaluation mode
            with torch.no_grad():
                # Use model's generate method if available
                if hasattr(self.model, "generate"):
                    generated = self.model.generate(
                        input_ids,
                        max_new_tokens=50,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=(
                            self.config.pad_token_id
                            if hasattr(self.config, "pad_token_id")
                            else 0
                        ),
                    )

                    # Try to decode if tokenizer is available
                    if hasattr(self, "tokenizer") and self.tokenizer is not None:
                        decoded_text = self.tokenizer.decode(generated[0].tolist())
                        self.logger.info(
                            f"Generated sample text: {decoded_text[:200]}..."
                        )
                    else:
                        self.logger.info(
                            f"Generated tokens: {generated[0].tolist()[:20]}..."
                        )
                else:
                    self.logger.info("Model does not have a generate method")

            self.model.train()  # Set back to training mode

        except Exception as e:
            self.logger.error(f"Error during sample text generation: {e}")


class LionOptimizer(optim.Optimizer):
    """
    Implementation of Lion optimizer
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super(LionOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data, dtype=torch.float)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Weight decay
                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                # Update momentum
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Compute update
                update = exp_avg.clone().sign_()

                # Apply update
                p.data.add_(update, alpha=-group["lr"])

                # Decay the momentum
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


class PreTrainer(Trainer):
    """
    Specialized trainer for pretraining
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainer_type = "pretrain"


class SFTTrainer(Trainer):
    """
    Specialized trainer for Supervised Fine-Tuning
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainer_type = "sft"

    def compute_loss(self, logits, labels):
        """Compute SFT-specific loss"""
        # Standard cross-entropy loss but only for non-masked tokens
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        return loss


class RLHFTrainer(Trainer):
    """
    Reinforcement Learning from Human Feedback trainer
    This is a simplified implementation - a full RLHF would be much more complex
    """

    def __init__(self, model, reward_model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.reward_model = reward_model
        self.trainer_type = "rlhf"

    def compute_loss(self, logits, labels):
        """
        Compute loss using policy gradient methods
        This is a simplified version - true RLHF is complex
        """
        # In true RLHF, you would use the reward model to compute rewards
        # and then use policy gradient methods like PPO
        # For this simplified version, we'll just use the standard loss
        # but in practice, this would be replaced with RL-specific computations
        return super().compute_loss(logits, labels)


class DPOTrainer(Trainer):
    """
    Direct Preference Optimization trainer
    """

    def __init__(self, beta=0.1, label_smoothing=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.trainer_type = "dpo"

    def compute_loss(self, chosen_logits, rejected_logits):
        """
        Compute DPO loss from chosen and rejected responses
        """
        # DPO loss calculation
        # This is a simplified implementation
        # In practice, you'd need to handle the batch having pairs of chosen/rejected
        logits = chosen_logits - rejected_logits
        loss = -torch.nn.functional.logsigmoid(self.beta * logits).mean()

        return loss


def create_trainer(trainer_type: str, model, config, train_data, eval_data=None, **kwargs):
    """
    Factory function to create the appropriate trainer
    """
    if trainer_type.lower() == "pretrain":
        return PreTrainer(model, config, train_data, eval_data=eval_data, **kwargs)
    elif trainer_type.lower() == "sft":
        return SFTTrainer(model, config, train_data, eval_data=eval_data, **kwargs)
    elif trainer_type.lower() == "rlhf":
        reward_model = kwargs.pop("reward_model", None)
        if reward_model is None:
            raise ValueError("RLHF trainer requires a reward model")
        return RLHFTrainer(model, reward_model, config, train_data, eval_data=eval_data, **kwargs)
    elif trainer_type.lower() == "dpo":
        return DPOTrainer(model, config, train_data, eval_data=eval_data, **kwargs)
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")


def train_model(
    config_path: str,
    train_data_path: str,
    trainer_type: str = "pretrain",
    eval_data_path: Optional[str] = None,
    output_dir: str = "./checkpoints",
    validation_split: float = 0.1,  # Add validation split parameter
    **kwargs,
):
    """
    High-level function to train a model from configuration and data

    Args:
        config_path: Path to model configuration
        train_data_path: Path to training data
        trainer_type: Type of training ('pretrain', 'sft', 'rlhf', 'dpo')
        eval_data_path: Path to evaluation data (optional)
        output_dir: Directory to save checkpoints
        validation_split: Fraction of training data to use for validation if no eval_data provided
        **kwargs: Additional arguments for trainer
    """
    # Load configuration
    config = Config.load(config_path)

    # Initialize model
    model = Transformer(config)

    # Load tokenizers and datasets would go here
    # This part would depend on your specific data loading implementation

    # Create trainer
    trainer = create_trainer(
        trainer_type,
        model,
        config,
        train_data=None,  # Replace with actual data loader
        eval_data=None,  # Will be set in trainer if needed
        save_dir=output_dir,
        validation_split=validation_split,
        **kwargs,
    )

    # Start training
    trainer.train()

    return trainer

