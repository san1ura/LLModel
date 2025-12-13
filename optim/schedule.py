"""
Optimizer and scheduler implementations for transformer models
Includes standard optimizers and advanced scheduling techniques
"""
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, List, Dict
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


class LionOptimizer(optim.Optimizer):
    """
    Implementation of Lion optimizer
    Paper: https://arxiv.org/abs/2302.06675
    """
    def __init__(
        self, 
        params, 
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99), 
        weight_decay: float = 0.0,
        use_triton: bool = False  # Future option for Triton optimization
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
        self.use_triton = use_triton
    
    def step(self, closure=None):
        """
        Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data, dtype=torch.float)
                
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Weight decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Update momentum
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
                
                # Compute update
                update = exp_avg.clone().sign_()
                
                # Apply update
                p.data.add_(update, alpha=-group['lr'])
                
                # Decay the momentum
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        
        return loss


class Adafactor(optim.Optimizer):
    """
    Adafactor optimizer implementation
    Paper: https://arxiv.org/abs/1804.04235
    """
    def __init__(
        self,
        params,
        lr: Optional[float] = None,
        eps: Tuple[float, float] = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        beta1: Optional[float] = None,
        weight_decay: float = 0.0,
        scale_parameter: bool = True,
        relative_step: bool = True,
        warmup_init: bool = False,
    ):
        if lr is not None and relative_step:
            # If lr is provided, disable relative_step
            relative_step = False
        if warmup_init and not relative_step:
            raise ValueError("warmup_init requires relative_step=True")

        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
        )
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        """
        Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                
                if grad.is_sparse:
                    raise RuntimeError('Adafactor does not support sparse gradients.')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    if grad.shape != [0]:  # Check for empty tensor
                        state['exp_avg_sq_row'] = torch.zeros(p.data.size(0)).to(grad.dtype).to(p.data.device)
                        state['exp_avg_sq_col'] = torch.zeros(p.data.size(1)).to(grad.dtype).to(p.data.device) if len(p.data.shape) > 1 else None
                    else:
                        state['exp_avg_sq_row'] = torch.zeros(1).to(grad.dtype).to(p.data.device)
                        state['exp_avg_sq_col'] = None
                    
                    if group['beta1'] is not None:
                        state['exp_avg'] = torch.zeros_like(p.data, dtype=torch.float)
                
                exp_avg_sq_row = state['exp_avg_sq_row']
                exp_avg_sq_col = state['exp_avg_sq_col']
                exp_avg = state.get('exp_avg', None)
                
                state['step'] += 1
                step_num = state['step']
                
                # Calculate lr
                if group['relative_step']:
                    lr = self._get_lr(group, state)
                else:
                    lr = group['lr']
                
                beta2t = min(1.0 - step_num ** group['decay_rate'], 0.99)
                
                # Add weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                # Precondition gradients
                grad_shape = grad.shape
                if len(grad_shape) == 2:
                    grad = grad / (grad.norm(dim=-1, keepdim=True) + group['eps'][0])
                    
                    # Update running averages
                    exp_avg_sq_row.mul_(beta2t).add_(torch.mean(grad.square(), dim=-1), alpha=1.0 - beta2t)
                    exp_avg_sq_col.mul_(beta2t).add_(torch.mean(grad.square(), dim=-2), alpha=1.0 - beta2t)
                    
                    # Calculate preconditioners
                    r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean()).rsqrt_().unsqueeze(-1)
                    c_factor = exp_avg_sq_col.rsqrt().unsqueeze(-2)
                    
                    # Apply preconditioning
                    update = grad * r_factor * c_factor
                else:
                    # For other dimensional tensors, use simpler approach
                    update = grad / grad.norm()
                
                # Apply update
                p.data.add_(update, alpha=-lr)
        
        return loss
    
    def _get_lr(self, group, state):
        """
        Get learning rate based on relative step parameters
        """
        rel_step_sz = group['lr']
        if group['relative_step']:
            min_step = 1e-6 * state['step'] if group['warmup_init'] else 1e-2
            rel_step_sz = min(min_step, 1.0 / math.sqrt(state['step']))
        return rel_step_sz


class CosineSchedulerWithWarmup:
    """
    Cosine learning rate scheduler with warmup and cooldown
    """
    def __init__(
        self,
        optimizer: optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
        min_lr_ratio: float = 0.1  # Minimum learning rate as fraction of peak
    ):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles
        self.min_lr_ratio = min_lr_ratio
        self.last_epoch = last_epoch
    
    def get_lr(self, current_step: int) -> List[float]:
        """
        Calculate learning rate for the current step
        """
        if current_step < self.num_warmup_steps:
            # Linear warmup
            if self.num_warmup_steps == 0:
                return [param_group['lr'] for param_group in self.optimizer.param_groups]
            else:
                progress = float(current_step) / float(max(1, self.num_warmup_steps))
                return [
                    param_group.get('initial_lr', param_group['lr']) * progress
                    for param_group in self.optimizer.param_groups
                ]
        else:
            # Cosine annealing
            progress = float(current_step - self.num_warmup_steps) / float(
                max(1, self.num_training_steps - self.num_warmup_steps)
            )
            cosine_factor = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress)))
            adjusted_lr = (
                self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_factor
            )
            return [
                param_group.get('initial_lr', param_group['lr']) * adjusted_lr
                for param_group in self.optimizer.param_groups
            ]
    
    def step(self, current_step: int = None):
        """
        Step the scheduler
        """
        if current_step is None:
            self.last_epoch += 1
            current_step = self.last_epoch
        
        lr_values = self.get_lr(current_step)
        for param_group, lr_value in zip(self.optimizer.param_groups, lr_values):
            param_group['lr'] = lr_value


class ConstantWithWarmupScheduler:
    """
    Constant learning rate scheduler with warmup period
    """
    def __init__(
        self,
        optimizer: optim.Optimizer,
        num_warmup_steps: int,
        constant_lr: float,
        last_epoch: int = -1
    ):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.constant_lr = constant_lr
        self.last_epoch = last_epoch
    
    def get_lr(self, current_step: int) -> List[float]:
        """
        Calculate learning rate for the current step
        """
        if current_step < self.num_warmup_steps:
            # Linear warmup to constant value
            if self.num_warmup_steps == 0:
                return [self.constant_lr] * len(self.optimizer.param_groups)
            else:
                progress = float(current_step) / float(max(1, self.num_warmup_steps))
                return [self.constant_lr * progress] * len(self.optimizer.param_groups)
        else:
            # Maintain constant learning rate
            return [self.constant_lr] * len(self.optimizer.param_groups)
    
    def step(self, current_step: int = None):
        """
        Step the scheduler
        """
        if current_step is None:
            self.last_epoch += 1
            current_step = self.last_epoch
        
        lr_values = self.get_lr(current_step)
        for param_group, lr_value in zip(self.optimizer.param_groups, lr_values):
            param_group['lr'] = lr_value


class LinearWithWarmupScheduler:
    """
    Linear decay learning rate scheduler with warmup
    """
    def __init__(
        self,
        optimizer: optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        end_lr_ratio: float = 0.0,  # Ratio of final lr to initial lr
        last_epoch: int = -1
    ):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.end_lr_ratio = end_lr_ratio
        self.last_epoch = last_epoch
    
    def get_lr(self, current_step: int) -> List[float]:
        """
        Calculate learning rate for the current step
        """
        if current_step < self.num_warmup_steps:
            # Linear warmup
            if self.num_warmup_steps == 0:
                return [param_group['lr'] for param_group in self.optimizer.param_groups]
            else:
                progress = float(current_step) / float(max(1, self.num_warmup_steps))
                return [
                    param_group.get('initial_lr', param_group['lr']) * progress
                    for param_group in self.optimizer.param_groups
                ]
        else:
            # Linear decay
            progress = float(current_step - self.num_warmup_steps) / float(
                max(1, self.num_training_steps - self.num_warmup_steps)
            )
            # Linear decay: from 1.0 to end_lr_ratio
            decay_factor = 1.0 - progress * (1.0 - self.end_lr_ratio)
            return [
                param_group.get('initial_lr', param_group['lr']) * max(0.0, decay_factor)
                for param_group in self.optimizer.param_groups
            ]
    
    def step(self, current_step: int = None):
        """
        Step the scheduler
        """
        if current_step is None:
            self.last_epoch += 1
            current_step = self.last_epoch
        
        lr_values = self.get_lr(current_step)
        for param_group, lr_value in zip(self.optimizer.param_groups, lr_values):
            param_group['lr'] = lr_value


class MultiStageScheduler:
    """
    Multi-stage learning rate scheduler for complex training schedules
    e.g., warmup -> constant -> decay
    """
    def __init__(
        self,
        optimizer: optim.Optimizer,
        stages: List[Tuple[str, int, dict]]  # List of (stage_type, duration, stage_kwargs)
    ):
        self.optimizer = optimizer
        self.stages = stages
        self.current_stage = 0
        self.stage_step = 0
        self.global_step = 0
        
        # Initialize schedulers for each stage
        self.schedulers = []
        for stage_type, duration, kwargs in self.stages:
            if stage_type == 'linear':
                scheduler_cls = LinearWithWarmupScheduler
                # Adjust duration to cover only this stage
                kwargs['num_training_steps'] = duration
            elif stage_type == 'cosine':
                scheduler_cls = CosineSchedulerWithWarmup
                kwargs['num_training_steps'] = duration
            elif stage_type == 'constant':
                scheduler_cls = ConstantWithWarmupScheduler
            else:
                raise ValueError(f"Unknown scheduler type: {stage_type}")
            
            # Create scheduler for this stage
            scheduler = scheduler_cls(optimizer, **kwargs)
            self.schedulers.append(scheduler)
    
    def step(self):
        """
        Step the scheduler for the current stage
        """
        self.global_step += 1
        
        # Check if we need to advance to the next stage
        if self.stage_step >= self.stages[self.current_stage][1]:
            if self.current_stage < len(self.stages) - 1:
                self.current_stage += 1
                self.stage_step = 0
            else:
                # Stay in the last stage
                self.stage_step += 1
        else:
            self.stage_step += 1
        
        # Step the current stage's scheduler
        current_scheduler = self.schedulers[self.current_stage]
        current_scheduler.step(self.stage_step)


class LayerwiseLearningRateScheduler:
    """
    Apply different learning rates to different layers of the model
    This is useful for fine-tuning where lower layers should be trained slower
    """
    def __init__(
        self,
        model: nn.Module,
        base_lr: float,
        lr_decay: float = 0.9,
        min_lr_ratio: float = 0.1
    ):
        self.model = model
        self.base_lr = base_lr
        self.lr_decay = lr_decay
        self.min_lr_ratio = min_lr_ratio
        
        # Create parameter groups with different learning rates
        param_groups = self._create_param_groups()
        
        # Create optimizer with the parameter groups
        self.optimizer = LionOptimizer(param_groups, lr=base_lr)
    
    def _create_param_groups(self) -> List[Dict]:
        """
        Create parameter groups with different learning rates for different layers
        """
        param_groups = []
        
        # Get all named parameters
        named_params = list(self.model.named_parameters())
        
        # Calculate number of layers (this is a simplification, refine based on your model structure)
        total_layers = self._estimate_total_layers()
        
        # Assign learning rate to each layer group
        for i, (name, param) in enumerate(named_params):
            # Calculate depth based on name hierarchy (simplified approach)
            layer_depth = self._infer_layer_depth(name, total_layers)
            
            # Calculate learning rate based on depth
            layer_lr_ratio = max(self.min_lr_ratio, self.lr_decay ** layer_depth)
            layer_lr = self.base_lr * layer_lr_ratio
            
            param_groups.append({
                'params': [param],
                'lr': layer_lr,
                'initial_lr': layer_lr
            })
        
        return param_groups
    
    def _estimate_total_layers(self) -> int:
        """
        Estimate the total number of layers in the model
        """
        # This is a simplified approach - adjust based on your model architecture
        # For transformer models, this might be the number of transformer blocks
        if hasattr(self.model, 'layers'):
            return len(self.model.layers)
        else:
            # Count transformer blocks or similar
            # This is a heuristic, you should customize based on your model
            return 32  # Default assumption
    
    def _infer_layer_depth(self, param_name: str, total_layers: int) -> int:
        """
        Infer the depth of the layer based on parameter name
        """
        # Example: layer.3.attention.wq.weight -> depth 3
        if 'layers.' in param_name:
            parts = param_name.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    try:
                        depth = int(parts[i + 1])
                        return depth
                    except ValueError:
                        pass
        
        # Default to shallow layer
        return 0
    
    def step(self):
        """
        Step the optimizer
        """
        self.optimizer.step()
    
    def zero_grad(self):
        """
        Zero gradients
        """
        self.optimizer.zero_grad()
    
    def get_param_group_lrs(self) -> List[float]:
        """
        Get the learning rates for all parameter groups
        """
        return [group['lr'] for group in self.optimizer.param_groups]


def get_optimizer(config, model_parameters):
    """
    Factory function to get the appropriate optimizer based on config
    
    Args:
        config: Configuration object containing optimizer settings
        model_parameters: Model parameters to optimize
        
    Returns:
        Optimizer object
    """
    optimizer_name = config.optimizer_name.lower() if hasattr(config, 'optimizer_name') else 'adamw'
    
    if optimizer_name == 'adamw':
        # Standard AdamW optimizer
        return optim.AdamW(
            model_parameters,
            lr=config.lr if hasattr(config, 'lr') else 3e-4,
            weight_decay=config.weight_decay if hasattr(config, 'weight_decay') else 0.1,
            betas=(0.9, 0.95),  # Standard values for transformers
            eps=1e-8
        )
    elif optimizer_name == 'lion':
        # Lion optimizer implementation
        return LionOptimizer(
            model_parameters,
            lr=config.lr if hasattr(config, 'lr') else 1e-4,
            betas=config.lion_betas if hasattr(config, 'lion_betas') else (0.9, 0.99),
            weight_decay=config.weight_decay if hasattr(config, 'weight_decay') else 0.0
        )
    elif optimizer_name == 'adafactor':
        # Adafactor optimizer
        return Adafactor(
            model_parameters,
            lr=config.lr if hasattr(config, 'lr') else 1e-3,
            weight_decay=config.weight_decay if hasattr(config, 'weight_decay') else 0.0,
            scale_parameter=config.scale_parameter if hasattr(config, 'scale_parameter') else True,
            relative_step=config.relative_step if hasattr(config, 'relative_step') else True
        )
    elif optimizer_name == 'sgd':
        # SGD with momentum
        return optim.SGD(
            model_parameters,
            lr=config.lr if hasattr(config, 'lr') else 1e-3,
            momentum=config.momentum if hasattr(config, 'momentum') else 0.9,
            weight_decay=config.weight_decay if hasattr(config, 'weight_decay') else 0.0
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(optimizer, config):
    """
    Factory function to get the appropriate scheduler based on config
    
    Args:
        optimizer: The optimizer to attach the scheduler to
        config: Configuration object containing scheduler settings
        
    Returns:
        Scheduler object
    """
    if not hasattr(config, 'scheduler_type'):
        return None  # No scheduler
    
    scheduler_type = config.scheduler_type.lower()
    
    if scheduler_type == 'cosine':
        return CosineSchedulerWithWarmup(
            optimizer,
            num_warmup_steps=config.num_warmup_steps if hasattr(config, 'num_warmup_steps') else 0,
            num_training_steps=config.num_training_steps if hasattr(config, 'num_training_steps') else 1000,
            num_cycles=config.cosine_num_cycles if hasattr(config, 'cosine_num_cycles') else 0.5,
            min_lr_ratio=config.min_lr_ratio if hasattr(config, 'min_lr_ratio') else 0.1
        )
    elif scheduler_type == 'linear':
        return LinearWithWarmupScheduler(
            optimizer,
            num_warmup_steps=config.num_warmup_steps if hasattr(config, 'num_warmup_steps') else 0,
            num_training_steps=config.num_training_steps if hasattr(config, 'num_training_steps') else 1000,
            end_lr_ratio=config.end_lr_ratio if hasattr(config, 'end_lr_ratio') else 0.0
        )
    elif scheduler_type == 'constant':
        warmup_steps = config.num_warmup_steps if hasattr(config, 'num_warmup_steps') else 0
        constant_lr = config.lr if hasattr(config, 'lr') else optimizer.defaults['lr']
        return ConstantWithWarmupScheduler(
            optimizer,
            num_warmup_steps=warmup_steps,
            constant_lr=constant_lr
        )
    elif scheduler_type == 'multistage':
        # Multi-stage scheduler
        stages = config.scheduler_stages if hasattr(config, 'scheduler_stages') else []
        return MultiStageScheduler(optimizer, stages)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")