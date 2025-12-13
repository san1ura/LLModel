"""
KV Cache management for efficient inference in transformer models
Optimized for both memory and compute efficiency
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import math


class KVCacheManager:
    """
    Advanced KV Cache Manager with multiple optimization strategies
    """
    def __init__(self, max_batch_size: int, max_seq_len: int, num_heads: int, head_dim: int, 
                 cache_layout: str = 'torch', use_paged_attention: bool = False,
                 block_size: int = 256):
        """
        Initialize KV cache manager with optimization options
        
        Args:
            max_batch_size: Maximum batch size supported
            max_seq_len: Maximum sequence length supported
            num_heads: Number of attention heads
            head_dim: Dimension of each head
            cache_layout: Memory layout ('torch', 'flash', etc.)
            use_paged_attention: Whether to use paged attention
            block_size: Block size for paged attention
        """
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.cache_layout = cache_layout
        self.use_paged_attention = use_paged_attention
        self.block_size = block_size
        
        if use_paged_attention:
            self.cache_k, self.cache_v = self._init_paged_cache()
        else:
            self.cache_k = torch.zeros(
                (max_batch_size, max_seq_len, num_heads, head_dim), 
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            self.cache_v = torch.zeros(
                (max_batch_size, max_seq_len, num_heads, head_dim), 
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        
        # Track sequence lengths per batch
        self.current_seq_lens = torch.zeros(max_batch_size, dtype=torch.long)
        
    def _init_paged_cache(self):
        """Initialize paged attention cache"""
        # Calculate number of blocks needed
        num_blocks = math.ceil(self.max_seq_len / self.block_size)
        
        # Initialize KV cache with blocks
        k_cache = torch.zeros(
            (self.max_batch_size, num_blocks, self.block_size, self.num_heads, self.head_dim),
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        v_cache = torch.zeros(
            (self.max_batch_size, num_blocks, self.block_size, self.num_heads, self.head_dim),
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        return k_cache, v_cache
    
    def update_cache(self, new_k: torch.Tensor, new_v: torch.Tensor, 
                     batch_idx: int, seq_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the KV cache with new keys and values
        
        Args:
            new_k: New key tensor [batch_size, seq_len, num_heads, head_dim]
            new_v: New value tensor [batch_size, seq_len, num_heads, head_dim]
            batch_idx: Index of the batch to update
            seq_pos: Starting sequence position to update
            
        Returns:
            Tuple of updated K and V caches for the current sequence
        """
        new_seq_len = new_k.size(1)
        updated_seq_len = seq_pos + new_seq_len
        
        if self.use_paged_attention:
            # Paged attention update
            start_block = seq_pos // self.block_size
            end_block = updated_seq_len // self.block_size
            
            for block_idx in range(start_block, end_block + 1):
                block_start = block_idx * self.block_size
                block_end = min((block_idx + 1) * self.block_size, updated_seq_len)
                
                if block_start < seq_pos + new_seq_len:  # Only update blocks that contain new data
                    cache_start = max(block_start - seq_pos, 0)  # Position in new_k
                    cache_end = min(block_end - seq_pos, new_seq_len)
                    
                    if cache_end > cache_start:
                        k_block_start = max(seq_pos - block_start, 0)
                        k_block_end = min(updated_seq_len - block_start, self.block_size)
                        
                        self.cache_k[batch_idx, block_idx, k_block_start:k_block_end, :, :] = \
                            new_k[0, cache_start:cache_end, :, :]
                        self.cache_v[batch_idx, block_idx, k_block_start:k_block_end, :, :] = \
                            new_v[0, cache_start:cache_end, :, :]
        else:
            # Standard cache update
            if updated_seq_len > self.max_seq_len:
                raise ValueError(f"Sequence length {updated_seq_len} exceeds maximum {self.max_seq_len}")
            
            # Update KV cache
            self.cache_k[batch_idx, seq_pos:updated_seq_len, :, :] = new_k[0]
            self.cache_v[batch_idx, seq_pos:updated_seq_len, :, :] = new_v[0]
        
        # Update sequence length
        self.current_seq_lens[batch_idx] = updated_seq_len
        
        # Return the full updated cache for this batch
        return self.get_cache(batch_idx, 0, updated_seq_len)
    
    def get_cache(self, batch_idx: int, start_pos: int, end_pos: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get KV cache for a specific batch and sequence range
        
        Args:
            batch_idx: Index of the batch
            start_pos: Start position in sequence
            end_pos: End position in sequence (if None, use current length)
            
        Returns:
            Tuple of K and V tensors for the specified range
        """
        if end_pos is None:
            end_pos = self.current_seq_lens[batch_idx]
        
        if self.use_paged_attention:
            # Collect blocks for the requested range
            start_block = start_pos // self.block_size
            end_block = end_pos // self.block_size
            
            if start_block == end_block:
                # Data is within a single block
                block_start = start_pos % self.block_size
                block_end = end_pos % self.block_size
                k_slice = self.cache_k[batch_idx, start_block, block_start:block_end, :, :]
                v_slice = self.cache_v[batch_idx, start_block, block_start:block_end, :, :]
            else:
                # Data spans multiple blocks
                k_parts = []
                v_parts = []
                
                # First block
                start_offset = start_pos % self.block_size
                k_parts.append(self.cache_k[batch_idx, start_block, start_offset:, :, :])
                v_parts.append(self.cache_v[batch_idx, start_block, start_offset:, :, :])
                
                # Middle blocks (if any)
                for b_idx in range(start_block + 1, end_block):
                    k_parts.append(self.cache_k[batch_idx, b_idx, :, :, :])
                    v_parts.append(self.cache_v[batch_idx, b_idx, :, :, :])
                
                # Last block
                end_offset = end_pos % self.block_size
                k_parts.append(self.cache_k[batch_idx, end_block, :end_offset, :, :])
                v_parts.append(self.cache_v[batch_idx, end_block, :end_offset, :, :])
                
                k_slice = torch.cat(k_parts, dim=0)
                v_slice = torch.cat(v_parts, dim=0)
            
            return k_slice, v_slice
        else:
            return (self.cache_k[batch_idx, start_pos:end_pos, :, :], 
                   self.cache_v[batch_idx, start_pos:end_pos, :, :])
    
    def clear_cache(self, batch_idx: Optional[int] = None):
        """
        Clear KV cache for specific batch or all batches
        
        Args:
            batch_idx: Index of batch to clear (if None, clear all)
        """
        if batch_idx is None:
            # Clear all cache
            if self.use_paged_attention:
                self.cache_k.zero_()
                self.cache_v.zero_()
            else:
                self.cache_k.zero_()
                self.cache_v.zero_()
            self.current_seq_lens.zero_()
        else:
            # Clear specific batch
            if self.use_paged_attention:
                self.cache_k[batch_idx].zero_()
                self.cache_v[batch_idx].zero_()
            else:
                self.cache_k[batch_idx].zero_()
                self.cache_v[batch_idx].zero_()
            self.current_seq_lens[batch_idx] = 0
    
    def get_current_length(self, batch_idx: int) -> int:
        """Get current sequence length for a batch"""
        return self.current_seq_lens[batch_idx].item()
    
    def resize_cache(self, new_max_seq_len: int):
        """
        Dynamically resize the cache to accommodate longer sequences
        
        Args:
            new_max_seq_len: New maximum sequence length
        """
        if new_max_seq_len <= self.max_seq_len:
            return  # No need to resize if new size is smaller
        
        old_max_seq_len = self.max_seq_len
        self.max_seq_len = new_max_seq_len
        
        if self.use_paged_attention:
            # For paged attention, we just need to add more blocks
            old_num_blocks = math.ceil(old_max_seq_len / self.block_size)
            new_num_blocks = math.ceil(new_max_seq_len / self.block_size)
            
            if new_num_blocks > old_num_blocks:
                # Add more blocks
                new_blocks_k = torch.zeros(
                    (self.max_batch_size, new_num_blocks - old_num_blocks, self.block_size, 
                     self.num_heads, self.head_dim),
                    dtype=self.cache_k.dtype, device=self.cache_k.device
                )
                new_blocks_v = torch.zeros(
                    (self.max_batch_size, new_num_blocks - old_num_blocks, self.block_size, 
                     self.num_heads, self.head_dim),
                    dtype=self.cache_v.dtype, device=self.cache_v.device
                )
                
                self.cache_k = torch.cat([self.cache_k, new_blocks_k], dim=1)
                self.cache_v = torch.cat([self.cache_v, new_blocks_v], dim=1)
        else:
            # For standard cache, create new larger tensors
            new_cache_k = torch.zeros(
                (self.max_batch_size, new_max_seq_len, self.num_heads, self.head_dim),
                dtype=self.cache_k.dtype, device=self.cache_k.device
            )
            new_cache_v = torch.zeros(
                (self.max_batch_size, new_max_seq_len, self.num_heads, self.head_dim),
                dtype=self.cache_v.dtype, device=self.cache_v.device
            )
            
            # Copy old data
            new_cache_k[:, :old_max_seq_len, :, :] = self.cache_k
            new_cache_v[:, :old_max_seq_len, :, :] = self.cache_v
            
            self.cache_k = new_cache_k
            self.cache_v = new_cache_v


class SlidingWindowCache:
    """
    Sliding window attention cache for long context models
    Keeps only the most recent tokens within the window
    """
    def __init__(self, max_batch_size: int, window_size: int, num_heads: int, head_dim: int):
        self.max_batch_size = max_batch_size
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Initialize cache with window size
        self.cache_k = torch.zeros(
            (max_batch_size, window_size, num_heads, head_dim),
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.cache_v = torch.zeros(
            (max_batch_size, window_size, num_heads, head_dim),
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Track the current position in the circular buffer
        self.current_positions = torch.zeros(max_batch_size, dtype=torch.long)
    
    def update(self, new_k: torch.Tensor, new_v: torch.Tensor, batch_idx: int):
        """
        Update cache with new keys/values, applying sliding window logic
        """
        seq_len = new_k.size(1)
        
        # Calculate where to place the new tokens in the circular buffer
        current_pos = self.current_positions[batch_idx]
        new_pos = (current_pos + seq_len) % self.window_size
        
        if current_pos + seq_len <= self.window_size:
            # New tokens fit in the remaining space
            self.cache_k[batch_idx, current_pos:current_pos + seq_len, :, :] = new_k[0]
            self.cache_v[batch_idx, current_pos:current_pos + seq_len, :, :] = new_v[0]
        else:
            # Need to wrap around the circular buffer
            remaining_space = self.window_size - current_pos
            self.cache_k[batch_idx, current_pos:, :, :] = new_k[0, :remaining_space]
            self.cache_v[batch_idx, current_pos:, :, :] = new_v[0, :remaining_space]
            
            if remaining_space < seq_len:
                # Place remaining tokens at the beginning
                self.cache_k[batch_idx, :seq_len - remaining_space, :, :] = new_k[0, remaining_space:]
                self.cache_v[batch_idx, :seq_len - remaining_space, :, :] = new_v[0, remaining_space:]
        
        self.current_positions[batch_idx] = new_pos
    
    def get_cache(self, batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the current cache for a batch, properly ordered"""
        current_pos = self.current_positions[batch_idx]
        
        if current_pos == 0:
            # Buffer is full and at start position
            return self.cache_k[batch_idx], self.cache_v[batch_idx]
        else:
            # Need to reorder to get the correct sequence
            ordered_k = torch.cat([
                self.cache_k[batch_idx, current_pos:, :, :],
                self.cache_k[batch_idx, :current_pos, :, :]
            ], dim=0)
            ordered_v = torch.cat([
                self.cache_v[batch_idx, current_pos:, :, :],
                self.cache_v[batch_idx, :current_pos, :, :]
            ], dim=0)
            
            return ordered_k, ordered_v


class QuantizedKVCache:
    """
    KV Cache with quantization to reduce memory usage
    """
    def __init__(self, max_batch_size: int, max_seq_len: int, num_heads: int, head_dim: int, 
                 quantization_bits: int = 8):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.quantization_bits = quantization_bits
        
        # Initialize quantized cache
        max_val = 2**(quantization_bits-1) - 1
        self.cache_k_quantized = torch.zeros(
            (max_batch_size, max_seq_len, num_heads, head_dim), 
            dtype=torch.int8 if quantization_bits <= 8 else torch.int16
        )
        self.cache_v_quantized = torch.zeros(
            (max_batch_size, max_seq_len, num_heads, head_dim), 
            dtype=torch.int8 if quantization_bits <= 8 else torch.int16
        )
        
        # Store scaling factors for dequantization
        self.k_scaling = torch.ones((max_batch_size, num_heads, head_dim))
        self.v_scaling = torch.ones((max_batch_size, num_heads, head_dim))
        
        # Track sequence lengths
        self.current_seq_lens = torch.zeros(max_batch_size, dtype=torch.long)
    
    def quantize_tensor(self, tensor: torch.Tensor, axis: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize a tensor and return quantized values with scaling factors
        """
        # Calculate scaling factor based on max absolute value
        max_vals = torch.amax(torch.abs(tensor), dim=axis, keepdim=True)
        max_vals = torch.clamp(max_vals, min=1e-5)  # Prevent division by zero
        
        scaling_factor = (2**(self.quantization_bits-1) - 1) / max_vals
        quantized = torch.clamp(
            torch.round(tensor * scaling_factor), 
            min=-(2**(self.quantization_bits-1)), 
            max=2**(self.quantization_bits-1) - 1
        )
        
        # Convert to appropriate dtype
        if self.quantization_bits <= 8:
            quantized = quantized.to(torch.int8)
        else:
            quantized = quantized.to(torch.int16)
            
        return quantized, scaling_factor.squeeze(axis)
    
    def dequantize_tensor(self, quantized: torch.Tensor, scaling: torch.Tensor, axis: int = -1) -> torch.Tensor:
        """
        Dequantize values using stored scaling factors
        """
        if len(scaling.shape) < len(quantized.shape):
            # Add dimension for broadcasting
            scaling = scaling.unsqueeze(axis)
        
        return quantized.float() / scaling
    
    def update_cache(self, new_k: torch.Tensor, new_v: torch.Tensor, 
                     batch_idx: int, seq_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache with quantized values"""
        # Quantize new values
        k_quantized, k_scaling = self.quantize_tensor(new_k)
        v_quantized, v_scaling = self.quantize_tensor(new_v)
        
        # Store quantized values
        new_seq_len = new_k.size(1)
        self.cache_k_quantized[batch_idx, seq_pos:seq_pos+new_seq_len, :, :] = k_quantized[0]
        self.cache_v_quantized[batch_idx, seq_pos:seq_pos+new_seq_len, :, :] = v_quantized[0]
        
        # Update scaling factors (for the last position, assuming all positions in the sequence have similar scaling)
        self.k_scaling[batch_idx] = k_scaling[0, -1]  # Use scaling from last token
        self.v_scaling[batch_idx] = v_scaling[0, -1]
        
        # Update sequence length
        updated_len = seq_pos + new_seq_len
        self.current_seq_lens[batch_idx] = updated_len
        
        # Return dequantized full cache for this batch
        full_k_quantized = self.cache_k_quantized[batch_idx, :updated_len]
        full_v_quantized = self.cache_v_quantized[batch_idx, :updated_len]
        
        full_k = self.dequantize_tensor(full_k_quantized, self.k_scaling[batch_idx])
        full_v = self.dequantize_tensor(full_v_quantized, self.v_scaling[batch_idx])
        
        return full_k, full_v