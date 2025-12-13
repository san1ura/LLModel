"""
FlashAttention implementation for efficient attention mechanism
Optimized for modern GPU architectures
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class FlashAttention(nn.Module):
    """
    FlashAttention implementation
    Efficient attention mechanism for long sequences
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, causal: bool = True):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.causal = causal
        
        # Ensure head_dim is compatible with FlashAttention
        if self.head_dim % 8 != 0 and self.head_dim < 32:
            raise ValueError("FlashAttention requires head_dim to be divisible by 8 or >= 32")
        
        # Linear projections
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass using FlashAttention
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (d_model)
        
        # Linear projections
        q = self.Wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = self.Wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = self.Wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        
        # Check if we can use FlashAttention
        if self._can_use_flash_attention(T):
            # Use FlashAttention via PyTorch's optimized implementation
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.causal
            )
        else:
            # Fallback to manual attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if mask is not None:
                att = att.masked_fill(mask[:,:,:T,:T] == 0, float('-inf'))
            if self.causal:
                # Causal mask to prevent attending to future tokens
                causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))
                att = att.masked_fill(~causal_mask, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.dropout_layer(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # Reassemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        
        # Output projection
        output = self.Wo(y)
        return output
    
    def _can_use_flash_attention(self, seq_len: int) -> bool:
        """
        Check if FlashAttention can be used based on sequence length and hardware
        """
        # For now, use PyTorch's built-in scaled_dot_product_attention
        # which internally uses optimized kernels when available
        return True


class FlashAttention2(nn.Module):
    """
    FlashAttention2 implementation with additional optimizations
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, 
                 causal: bool = True, use_sliding_window: bool = False, 
                 sliding_window_size: int = 1024):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.causal = causal
        self.use_sliding_window = use_sliding_window
        self.sliding_window_size = sliding_window_size
        
        # Ensure head_dim is compatible with FlashAttention
        if self.head_dim % 8 != 0 and self.head_dim < 32:
            raise ValueError("FlashAttention requires head_dim to be divisible by 8 or >= 32")
        
        # Linear projections
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass using FlashAttention2
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (d_model)
        
        # Linear projections
        q = self.Wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = self.Wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = self.Wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        
        # Create attention mask based on sliding window if needed
        if self.use_sliding_window and T > self.sliding_window_size:
            # Create sliding window mask
            window_mask = self._create_sliding_window_mask(T, self.sliding_window_size, device=q.device)
            if mask is not None:
                mask = mask & window_mask
            else:
                mask = window_mask
        
        # Use PyTorch's optimized attention
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal and not self.use_sliding_window  # Can't use causal if using sliding window
        )
        
        # Reassemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        
        # Output projection
        output = self.Wo(y)
        return output
    
    def _create_sliding_window_mask(self, seq_len: int, window_size: int, device: torch.device) -> torch.Tensor:
        """
        Create a sliding window attention mask
        """
        mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
        
        # Zero out attention beyond the window
        for i in range(seq_len):
            start_idx = max(0, i - window_size + 1)
            end_idx = min(seq_len, i + window_size)
            mask[i, :start_idx] = False
            mask[i, end_idx:] = False
        
        return mask


def flash_attention_func(q, k, v, dropout_p=0.0, causal=False, window_size=None):
    """
    Standalone FlashAttention function
    """
    B, T, H, D = q.shape
    
    # Use PyTorch's optimized attention as FlashAttention
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v,
        dropout_p=dropout_p,
        is_causal=causal
    )