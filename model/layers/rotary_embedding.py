"""
Rotary Position Embedding implementation for transformers
Includes RoPE, ALiBi, and Dynamic NTK scaling
"""
import math
import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Rotary Embedding implementation with support for variable sequence length
    Extended to support Dynamic NTK scaling
    """
    def __init__(self, dim, max_len=2048, scaling_factor=1.0, dynamic_ntk=True):
        super().__init__()
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(f"dim must be positive and even, got {dim}")
        if max_len <= 0:
            raise ValueError(f"max_len must be positive, got {max_len}")

        self.dim = dim
        self.max_len = max_len
        self.scaling_factor = scaling_factor
        self.dynamic_ntk = dynamic_ntk
        
        # Standard rotary embedding frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, positions):
        """
        Create rotary embeddings for given positions
        
        Args:
            positions: [B, S] or [S] tensor containing position indices
            
        Returns:
            tuple: (cos_freqs, sin_freqs) with shapes [B, S, 1, D//2]
        """
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)  # [1, S]
        B, S = positions.size()
        
        # Handle sequences longer than max_len with Dynamic NTK
        if S > self.max_len and self.dynamic_ntk:
            # Apply Dynamic NTK scaling
            factor = self.scaling_factor * (S / self.max_len) ** 0.5
            inv_freq = self.inv_freq / factor
        else:
            inv_freq = self.inv_freq

        # Calculate frequencies
        freqs = torch.einsum("bs,d->bsd", positions.float(), inv_freq)  # [B, S, D//2]
        cos_freqs = freqs.cos().unsqueeze(2)  # [B, S, 1, D//2]
        sin_freqs = freqs.sin().unsqueeze(2)  # [B, S, 1, D//2]
        return cos_freqs, sin_freqs


def rotate_half(x):
    """
    Rotate half the hidden dimensions of the input.
    
    Args:
        x: Input tensor of shape [..., D] where D is even
        
    Returns:
        Rotated tensor of same shape
    """
    if x.size(-1) % 2 != 0:
        raise ValueError(f"Last dimension of x must be even, got shape {x.shape}")

    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    """
    Apply rotary position embedding to input tensor.
    Fixed to properly handle KV caching scenarios where sequence lengths might differ.

    Args:
        x: Input tensor of shape [B, S, H, D] format
        cos: Cosine embeddings of shape [B, S, 1, D//2] format or [B, total_S, 1, D//2] for cached scenarios
        sin: Sine embeddings of shape [B, S, 1, D//2] format or [B, total_S, 1, D//2] for cached scenarios

    Returns:
        Tensor with rotary position embedding applied, same shape as x
    """
    if len(x.shape) != 4:
        raise ValueError(f"Expected 4D input [B, S, H, D], got {x.shape}")
    if len(cos.shape) != 4 or len(sin.shape) != 4:
        raise ValueError(
            f"Expected 4D cos/sin [B, S, 1, D//2], got cos:{cos.shape}, sin:{sin.shape}"
        )

    B, S, H, D = x.shape

    # For KV caching scenarios, the cos/sin might be computed for the total sequence length
    # But we only need the embeddings for the specific positions in x
    # For Q: we need embeddings for current positions
    # For K: we need embeddings for all positions (past + current)
    # Determine how many embeddings we actually need based on input tensor size
    required_seq_len = x.shape[1]  # This is the sequence length we need to apply RoPE to

    if cos.shape[1] >= required_seq_len and sin.shape[1] >= required_seq_len:
        # Take the required portion of cos/sin for the sequence length needed
        if x.shape[1] <= cos.shape[1]:  # If we need fewer embeddings than available
            cos = cos[:, :required_seq_len, :, :]
            sin = sin[:, :required_seq_len, :, :]
        else:  # If we need more embeddings than available
            raise ValueError(
                f"Insufficient positional embeddings: cos/sin have length {cos.shape[1]}, "
                f"but input sequence has length {required_seq_len}. Pre-compute more positional embeddings."
            )
    elif cos.shape[1] < required_seq_len:
        raise ValueError(
            f"Insufficient positional embeddings: cos/sin have length {cos.shape[1]}, "
            f"but input sequence has length {required_seq_len}. Pre-compute more positional embeddings."
        )

    # Adjust the expected shape based on the actual sequence length we're using
    actual_seq_len = cos.shape[1]  # Use the sequence length after slicing
    expected_cos_sin_shape = (B, actual_seq_len, 1, D // 2)  # Adjust to actual length used
    if cos.shape != expected_cos_sin_shape or sin.shape != expected_cos_sin_shape:
        raise ValueError(
            f"Expected cos/sin shape after slicing to be {expected_cos_sin_shape}, got cos:{cos.shape}, sin:{sin.shape}"
        )

    # Update S to reflect the actual sequence length we're working with
    S = actual_seq_len

    # Make sure D is even (required for RoPE)
    if D % 2 != 0:
        raise ValueError(f"Last dimension D={D} must be even for RoPE, got shape {x.shape}")

    # Split x tensor for rotary embedding - ensure we're using proper dimensions
    x1 = x[..., ::2]  # x1: [B, S, H, D//2] - even indices
    x2 = x[..., 1::2]  # x2: [B, S, H, D//2] - odd indices

    # Apply rotary embedding
    x1_rot = x1 * cos - x2 * sin
    x2_rot = x1 * sin + x2 * cos
    # Concatenate back
    x_rot = torch.cat([x1_rot, x2_rot], dim=-1)
    return x_rot


class ALiBi(nn.Module):
    """
    Attention with Linear Biases (ALiBi) - position-free attention bias
    """
    def __init__(self, n_heads, max_len=2048):
        super().__init__()
        self.n_heads = n_heads
        self.max_len = max_len
        
        # Learnable slopes for each head
        slopes = torch.tensor([1/(2**i) for i in range(1, n_heads + 1)])
        self.register_buffer("slopes", slopes)
        
    def forward(self, seq_len):
        """
        Generate ALiBi attention bias matrix
        
        Args:
            seq_len: Current sequence length
            
        Returns:
            Attention bias tensor of shape [n_heads, seq_len, seq_len]
        """
        # Create distance matrix
        distance = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
        distance = distance.unsqueeze(0)  # [1, seq_len, seq_len]
        
        # Apply slopes for each head
        alibi_bias = self.slopes.unsqueeze(1).unsqueeze(2) * distance  # [n_heads, seq_len, seq_len]
        return alibi_bias