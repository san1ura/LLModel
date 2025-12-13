"""
Fused operations for optimization
Contains optimized implementations of common operations
"""
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


class FusedLayerNorm(nn.Module):
    """
    Fused LayerNorm implementation for improved performance
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, input):
        # Use PyTorch's native layer norm which is already optimized
        return torch.nn.functional.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )


class FusedRMSNorm(nn.Module):
    """
    Fused RMSNorm implementation for improved performance
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # Compute RMS
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return self.weight * (x / rms)


class FusedLinear(nn.Module):
    """
    Wrapper for fused linear operations
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.weight.size(1)
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input):
        # Use PyTorch's optimized linear operation
        return nn.functional.linear(input, self.weight, self.bias)


class FusedSoftmax(nn.Module):
    """
    Fused Softmax for attention computation
    """
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
    
    def forward(self, input):
        return torch.softmax(input, dim=self.dim)


class FusedDropout(nn.Module):
    """
    Fused Dropout implementation
    """
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"Dropout probability must be in range [0, 1], got {p}")
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        return torch.nn.functional.dropout(input, self.p, self.training, self.inplace)


class FusedGeLU(nn.Module):
    """
    Fused GeLU activation
    """
    def __init__(self, approximate: str = "tanh"):
        super().__init__()
        self.approximate = approximate

    def forward(self, input):
        return torch.nn.functional.gelu(input, approximate=self.approximate)


class FusedSiLU(nn.Module):
    """
    Fused SiLU (Swish) activation
    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.nn.functional.silu(input)


class FusedLion(torch.optim.Optimizer):
    """
    Fused Lion optimizer implementation
    Lion is a low-memory adaptive optimizer
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super(FusedLion, self).__init__(params, defaults)

    def step(self, closure=None):
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

                # Weight decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Update momentum
                update = exp_avg.clone().sign_().mul_(beta1) + grad.mul_(1 - beta1)

                # Apply update
                p.data.add_(update.sign_(), alpha=-group['lr'])

                # Decay the momentum
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


class FusedAdamW(torch.optim.Optimizer):
    """
    Fused AdamW optimizer implementation
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(FusedAdamW, self).__init__(params, defaults)

    def step(self, closure=None):
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
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=torch.float)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute the corrected update
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1

                # Apply weight decay (decoupled from step direction)
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])

                # Apply the step update
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def fused_add_normalize_dropout_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    dropout_p: float = 0.0,
    eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused operation for residual connection + normalization + dropout
    This is a conceptual function since PyTorch doesn't have a single fused op for this
    """
    # Add residual
    x = x + residual

    # Normalize
    x = torch.nn.functional.layer_norm(x, x.shape[-1:], norm_weight, norm_bias, eps)

    # Dropout
    x = torch.nn.functional.dropout(x, dropout_p, training=True)

    return x, residual


class FusedMLP(nn.Module):
    """
    Fused MLP (Feed Forward Network) with optimized operations
    """
    def __init__(self, d_model: int, d_ff: int, activation: str = "silu", dropout: float = 0.0):
        super().__init__()
        self.w1 = FusedLinear(d_model, d_ff, bias=False)  # Up projection
        self.w2 = FusedLinear(d_model, d_ff, bias=False)  # Gate projection
        self.w3 = FusedLinear(d_ff, d_model, bias=False)  # Down projection

        if activation == "silu":
            self.activation = FusedSiLU()
        elif activation == "gelu":
            self.activation = FusedGeLU()
        else:
            raise ValueError(f"Activation {activation} not supported")

        self.dropout = FusedDropout(dropout)

    def forward(self, x):
        gate = self.w1(x)
        up = self.w2(x)

        # SwiGLU: up * SiLU(gate)
        activated_gate = self.activation(gate)
        result = up * activated_gate

        output = self.w3(result)
        output = self.dropout(output)
        return output