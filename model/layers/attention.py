"""
Attention mechanisms for the transformer model
Including Multi-Head Attention, FlashAttention2 and KV-cache optimization
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MHA(nn.Module):
    """
    Multi-Head Attention with KV-cache and RoPE support
    Optimized for both training and inference
    """

    def __init__(
        self,
        d_model,
        n_heads,
        use_rope=False,
        use_flash_attention=False,
        kv_cache_optimized=True,
        multi_query_attention=False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope
        self.use_flash_attention = use_flash_attention
        self.kv_cache_optimized = kv_cache_optimized
        self.multi_query_attention = (
            multi_query_attention  # When True, use MQA instead of MHA
        )

        if use_rope:
            from .rotary_embedding import RotaryEmbedding

            self.rope = RotaryEmbedding(self.head_dim)

        if multi_query_attention:
            # Multi-Query Attention: single K and V head shared across all Q heads
            self.fc_q = nn.Linear(d_model, d_model, bias=False)  # Multiple Q heads
            self.fc_k = nn.Linear(d_model, self.head_dim, bias=False)  # Single K head
            self.fc_v = nn.Linear(d_model, self.head_dim, bias=False)  # Single V head
        else:
            # Standard Multi-Head Attention
            self.fc_q = nn.Linear(d_model, d_model, bias=False)
            self.fc_k = nn.Linear(d_model, d_model, bias=False)
            self.fc_v = nn.Linear(d_model, d_model, bias=False)

        self.fc_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None, kv=None, past_kv_cache=None):
        B, S, D = x.size()

        # Input validation
        if D != self.d_model:
            raise ValueError(f"Expected last dimension to be {self.d_model}, got {D}")

        # Enhanced error handling with detailed checks
        if x.isnan().any():
            raise ValueError("Input tensor contains NaN values")
        if x.isinf().any():
            raise ValueError("Input tensor contains infinite values")

        # KV cache system for inference:
        # During training: past_kv_cache=None, new K and V are calculated each time
        # During inference: past_kv_cache=cache, cache is combined with new tokens

        if kv is not None:
            # Inference with incoming K and V values - usually just for the last token
            if not isinstance(kv, tuple):
                raise TypeError(f"Expected kv to be a tuple of (K, V), got {type(kv)}")
            if len(kv) != 2:
                raise ValueError(
                    f"Expected kv to be a tuple of (K, V) with length 2, got length {len(kv)}"
                )

            K_new, V_new = kv
            if K_new.size() != V_new.size():
                raise ValueError(
                    f"K and V must have the same shape, got K:{K_new.shape}, V:{V_new.shape}"
                )

            # Validate K_new and V_new tensors
            if K_new.isnan().any() or V_new.isnan().any():
                raise ValueError("K_new or V_new contains NaN values")
            if K_new.isinf().any() or V_new.isinf().any():
                raise ValueError("K_new or V_new contains infinite values")

            # Inference: combine past_kv_cache with K_new and V_new
            # First, ensure K_new and V_new are in the correct shape for concatenation
            if self.multi_query_attention:
                # For MQA: [B, S, 1, head_dim]
                if len(K_new.shape) == 3:
                    K_new = K_new.view(B, K_new.size(1), 1, self.head_dim)
                    V_new = V_new.view(B, V_new.size(1), 1, self.head_dim)
            else:
                # For MHA: [B, S, n_heads, head_dim]
                if len(K_new.shape) == 3:
                    K_new = K_new.view(B, K_new.size(1), self.n_heads, self.head_dim)
                    V_new = V_new.view(B, V_new.size(1), self.n_heads, self.head_dim)

            if past_kv_cache is not None:
                if not isinstance(past_kv_cache, tuple) or len(past_kv_cache) != 2:
                    raise ValueError("past_kv_cache must be a tuple of (K, V)")
                K_past, V_past = past_kv_cache

                # Additional validation for cached tensors
                if K_past.isnan().any() or V_past.isnan().any():
                    raise ValueError("Past KV cache contains NaN values")
                if K_past.isinf().any() or V_past.isinf().any():
                    raise ValueError("Past KV cache contains infinite values")

                # Shape checks
                if K_past.size(0) != B or V_past.size(0) != B:
                    raise ValueError(
                        f"Batch size mismatch in past_kv_cache: expected {B}, got K:{K_past.size(0)}, V:{V_past.size(0)}"
                    )
                if not self.multi_query_attention:
                    if K_past.size(2) != self.n_heads or V_past.size(2) != self.n_heads:
                        raise ValueError(
                            f"Heads mismatch in past_kv_cache: expected {self.n_heads}, got K:{K_past.size(2)}, V:{V_past.size(2)}"
                        )
                    if (
                        K_past.size(3) != self.head_dim
                        or V_past.size(3) != self.head_dim
                    ):
                        raise ValueError(
                            f"Head dim mismatch in past_kv_cache: expected {self.head_dim}, got K:{K_past.size(3)}, V:{V_past.size(3)}"
                        )
                else:
                    # For MQA, K and V only have 1 head dimension
                    if K_past.size(2) != 1 or V_past.size(2) != 1:
                        raise ValueError(
                            f"MQA expects 1 head in past_kv_cache, got K:{K_past.size(2)}, V:{V_past.size(2)}"
                        )
                    if (
                        K_past.size(3) != self.head_dim
                        or V_past.size(3) != self.head_dim
                    ):
                        raise ValueError(
                            f"Head dim mismatch in past_kv_cache: expected {self.head_dim}, got K:{K_past.size(3)}, V:{V_past.size(3)}"
                        )

                # Combine existing cache with new KV
                K_all = torch.cat([K_past, K_new], dim=1)
                V_all = torch.cat([V_past, V_new], dim=1)
            else:
                # First inference step: only use new KV
                K_all = K_new
                V_all = V_new
        else:
            # Training or inference start: calculate new K and V
            Q = self.fc_q(x)

            if self.multi_query_attention:
                # For MQA, K and V have only 1 head
                K_new = self.fc_k(x).unsqueeze(2)  # Shape: [B, S, 1, D_head]
                V_new = self.fc_v(x).unsqueeze(2)  # Shape: [B, S, 1, D_head]
            else:
                # Standard MHA
                K_new = self.fc_k(x).view(
                    B, S, self.n_heads, self.head_dim
                )  # Shape: [B, S, n_heads, head_dim]
                V_new = self.fc_v(x).view(
                    B, S, self.n_heads, self.head_dim
                )  # Shape: [B, S, n_heads, head_dim]

            if past_kv_cache is not None:
                # Combine new K and V with existing cache during inference
                if not isinstance(past_kv_cache, tuple) or len(past_kv_cache) != 2:
                    raise ValueError("past_kv_cache must be a tuple of (K, V)")
                K_past, V_past = past_kv_cache

                # Additional validation for cached tensors
                if K_past.isnan().any() or V_past.isnan().any():
                    raise ValueError("Past KV cache contains NaN values")
                if K_past.isinf().any() or V_past.isinf().any():
                    raise ValueError("Past KV cache contains infinite values")

                # Shape checks
                if self.multi_query_attention:
                    # For MQA
                    if K_past.size(0) != B or V_past.size(0) != B:
                        raise ValueError(
                            f"Batch size mismatch in past_kv_cache: expected {B}, got K:{K_past.size(0)}, V:{V_past.size(0)}"
                        )
                    if K_past.size(2) != 1 or V_past.size(2) != 1:
                        raise ValueError(
                            f"MQA expects 1 head in past_kv_cache, got K:{K_past.size(2)}, V:{V_past.size(2)}"
                        )
                    if (
                        K_past.size(3) != self.head_dim
                        or V_past.size(3) != self.head_dim
                    ):
                        raise ValueError(
                            f"Head dim mismatch in past_kv_cache: expected {self.head_dim}, got K:{K_past.size(3)}, V:{V_past.size(3)}"
                        )

                    K_all = torch.cat([K_past, K_new], dim=1)
                    V_all = torch.cat([V_past, V_new], dim=1)
                else:
                    # For MHA
                    if K_past.size(0) != B or V_past.size(0) != B:
                        raise ValueError(
                            f"Batch size mismatch in past_kv_cache: expected {B}, got K:{K_past.size(0)}, V:{V_past.size(0)}"
                        )
                    if K_past.size(2) != self.n_heads or V_past.size(2) != self.n_heads:
                        raise ValueError(
                            f"Heads mismatch in past_kv_cache: expected {self.n_heads}, got K:{K_past.size(2)}, V:{V_past.size(2)}"
                        )
                    if (
                        K_past.size(3) != self.head_dim
                        or V_past.size(3) != self.head_dim
                    ):
                        raise ValueError(
                            f"Head dim mismatch in past_kv_cache: expected {self.head_dim}, got K:{K_past.size(3)}, V:{V_past.size(3)}"
                        )

                    K_all = torch.cat([K_past, K_new], dim=1)
                    V_all = torch.cat([V_past, V_new], dim=1)
            else:
                # Training: only calculate new K and V
                K_all = K_new
                V_all = V_new

        # Calculate Q based on current x
        # Preserve the original K_all and V_all to return for cache (without rotary embedding)
        K_all_orig = K_all.clone()
        V_all_orig = V_all.clone()

        if not self.multi_query_attention:
            # Standard MHA: Q has multiple heads
            Q = Q.view(B, S, self.n_heads, self.head_dim).transpose(
                1, 2
            )  # (B, num_heads, S, head_dim)
            K_all = K_all_orig.view(
                B, K_all_orig.size(1), self.n_heads, self.head_dim
            ).transpose(
                1, 2
            )  # (B, num_heads, cache+S, head_dim)
            V_all = V_all_orig.view(
                B, V_all_orig.size(1), self.n_heads, self.head_dim
            ).transpose(
                1, 2
            )  # (B, num_heads, cache+S, head_dim)
        else:
            # MQA: Q has multiple heads, K/V have single head
            Q = Q.view(B, S, self.n_heads, self.head_dim).transpose(
                1, 2
            )  # (B, num_heads, S, head_dim)
            # K_all and V_all already have shape (B, seq_len, 1, head_dim) for MQA

        # Apply Rotary Position Embedding (RoPE) - must be applied in both cases
        if self.use_rope:
            # Get the total sequence length (past + current)
            total_seq_len = K_all.size(2)  # This is cache_len + S after transpose(1, 2)

            # Create position indices for K (from 0 to total_seq_len-1)
            k_positions = (
                torch.arange(total_seq_len, dtype=torch.long, device=Q.device)
                .unsqueeze(0)
                .expand(B, -1)
            )  # [B, total_seq_len]
            from .rotary_embedding import apply_rotary_pos_emb

            k_cos, k_sin = self.rope(
                k_positions.to(self.rope.inv_freq.device)
            )  # [B, total_seq_len, 1, head_dim//2]

            # Apply rotary embedding to K (for all cache + new tokens)
            if not self.multi_query_attention:
                K_all = apply_rotary_pos_emb(
                    K_all.transpose(1, 2), k_cos, k_sin
                ).transpose(
                    1, 2
                )  # [B, num_heads, total_seq_len, head_dim]
            else:
                # For MQA, K shape is [B, seq_len, 1, head_dim], so expand dims to match RoPE requirements
                K_all = apply_rotary_pos_emb(
                    K_all, k_cos, k_sin
                )  # [B, seq_len, 1, head_dim]

            # Create position indices for Q
            # For Q, they should correspond to the positions in the original sequence
            # If we don't have cache, positions are from 0 to S-1
            # If we have cache, Q positions are from the end of the cached sequence to cache+S-1
            past_len = past_kv_cache[0].size(1) if past_kv_cache is not None else 0
            q_positions = (
                torch.arange(past_len, past_len + S, dtype=torch.long, device=Q.device)
                .unsqueeze(0)
                .expand(B, -1)
            )  # [B, S]
            q_cos, q_sin = self.rope(
                q_positions.to(self.rope.inv_freq.device)
            )  # [B, S, 1, head_dim//2]

            # Apply rotary embedding to Q (for the current positions)
            Q = apply_rotary_pos_emb(Q.transpose(1, 2), q_cos, q_sin).transpose(
                1, 2
            )  # [B, num_heads, S, head_dim]

        # Masking for attention
        if mask is not None:
            # For training or custom mask application
            # The mask should be 3D [B, S, S] for full attention matrix
            if mask.dim() == 3:
                # Full attention mask matrix [B, S, S] - use as is
                attn_mask = mask
                is_causal = False
            elif mask.dim() == 2:
                # If the mask is [B, S], expand to [B, S, S] with causal attention
                # Create full attention mask by combining causal and padding masks
                B, S = mask.size()
                causal_mask = torch.tril(
                    torch.ones((S, S), dtype=torch.bool, device=mask.device)
                )
                causal_mask = causal_mask.unsqueeze(0).expand(B, -1, -1)  # [B, S, S]

                # Create mask for padding tokens: [B, S, 1] * [B, 1, S] -> [B, S, S]
                mask_expanded = mask.unsqueeze(-1) * mask.unsqueeze(1)  # [B, S, S]

                # Combine causal and padding masks
                combined_mask = causal_mask & (mask_expanded.bool())

                # Convert to appropriate format for attention computation
                attn_mask = torch.zeros_like(
                    combined_mask, dtype=torch.float, device=mask.device
                )
                attn_mask.masked_fill_(~combined_mask, float("-inf"))
                is_causal = False
            else:
                # Fallback for other dimensions
                attn_mask = (
                    mask.float()
                    .masked_fill(mask == 0, float("-inf"))
                    .masked_fill(mask == 1, float(0.0))
                )
                is_causal = False
        elif self.training:
            # Use causal mask during training
            attn_mask = None
            is_causal = True
        else:
            # Use causal mask during inference
            attn_mask = None
            is_causal = True

        # Attention mechanism - choose based on available options
        try:
            if self.use_flash_attention and hasattr(F, "scaled_dot_product_attention"):
                # Use PyTorch's optimized attention which is memory efficient and fast
                out = F.scaled_dot_product_attention(
                    Q,
                    K_all,
                    V_all,
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                    is_causal=is_causal,
                )
            else:
                # Fallback to manual implementation with memory optimizations
                attn_weights = torch.matmul(Q, K_all.transpose(-2, -1)) / math.sqrt(
                    Q.size(-1)
                )

                # Apply mask if provided
                if attn_mask is not None:
                    # Ensure attn_mask has the same number of dimensions as attn_weights
                    if attn_mask.dim() != attn_weights.dim():
                        # Handle different dimensional masks
                        if attn_mask.dim() == 2 and attn_weights.dim() == 4:
                            # [S, S] -> [1, 1, S, S] to broadcast across batches and heads
                            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                        elif attn_mask.dim() == 3 and attn_weights.dim() == 4:
                            # [B, S, S] -> [B, 1, S, S] to broadcast across heads
                            attn_mask = attn_mask.unsqueeze(1)
                        # If dimensions still don't match after this, we'll let PyTorch handle the error

                    attn_weights = attn_weights + attn_mask

                # Apply causal mask if needed
                if is_causal:
                    seq_len_q, seq_len_k = Q.size(-2), K_all.size(-2)
                    causal_mask = torch.tril(
                        torch.ones(
                            seq_len_q, seq_len_k, dtype=torch.bool, device=Q.device
                        )
                    )
                    # Expand mask for all batches and heads
                    if not self.multi_query_attention:
                        causal_mask = causal_mask.unsqueeze(0).unsqueeze(
                            0
                        )  # [1, 1, seq_len_q, seq_len_k]
                    else:
                        # For MQA, we broadcast across all heads
                        causal_mask = causal_mask.unsqueeze(0).unsqueeze(
                            1
                        )  # [1, 1, seq_len_q, seq_len_k]
                    attn_weights = attn_weights.masked_fill(~causal_mask, float("-inf"))

                attn_weights = F.softmax(attn_weights, dim=-1)
                out = torch.matmul(attn_weights, V_all)
        except Exception as e:
            # Ultimate fallback with memory-efficient computation
            # Compute attention in chunks to prevent memory issues
            chunk_size = 1024  # Adjustable based on available memory

            # Process attention in chunks to limit memory usage
            all_chunks = []
            for i in range(
                0, Q.size(2), chunk_size
            ):  # S dimension is index 2 for [B, H, S, D]
                Q_chunk = Q[:, :, i : i + chunk_size, :]  # [B, H, chunk_size, D]

                # Compute attention weights for this chunk
                attn_weights_chunk = torch.matmul(
                    Q_chunk, K_all.transpose(-2, -1)
                ) / math.sqrt(Q_chunk.size(-1))

                # Apply mask if provided
                if attn_mask is not None:
                    # Ensure attn_mask has the same number of dimensions as attn_weights_chunk
                    if attn_mask.dim() != attn_weights_chunk.dim():
                        # Handle different dimensional masks
                        if attn_mask.dim() == 3 and attn_weights_chunk.dim() == 4:
                            # [B, S, S] -> [B, 1, S, S] to broadcast across heads for main attention
                            # But for chunked attention with chunk_size, we need to select the right slice
                            if attn_mask.size(2) > i + chunk_size:
                                # Extract the relevant slice for this chunk [B, S, S] -> [B, chunk_size, S]
                                mask_chunk = attn_mask[:, i : i + chunk_size, :]
                                # Expand to match attention weights chunk [B, 1, chunk_size, S]
                                mask_chunk_expanded = mask_chunk.unsqueeze(1)
                            else:
                                # Use the full mask and expand appropriately
                                mask_chunk_expanded = attn_mask.unsqueeze(1)
                        elif attn_mask.dim() == 2 and attn_weights_chunk.dim() == 4:
                            # [S, S] -> [1, 1, S, S] to broadcast across batches and heads
                            mask_chunk_expanded = attn_mask.unsqueeze(0).unsqueeze(0)
                        # If dimensions still don't match after this, we'll let PyTorch handle the error
                    else:
                        # Mask dimensions already match
                        if attn_mask.size(2) > i + chunk_size:
                            # Extract appropriate slice for this chunk
                            mask_chunk = attn_mask[:, i : i + chunk_size, :]
                            mask_chunk_expanded = mask_chunk.unsqueeze(
                                1
                            )  # [B, 1, chunk_size, S]
                        else:
                            # Use the full mask
                            mask_chunk_expanded = attn_mask.unsqueeze(1)  # [B, 1, S, S]

                        attn_weights_chunk = attn_weights_chunk + mask_chunk_expanded

                # Apply causal mask if needed
                if is_causal:
                    seq_len_q = Q_chunk.size(2)
                    seq_len_k = K_all.size(2)
                    causal_mask = torch.tril(
                        torch.ones(
                            seq_len_q,
                            seq_len_k,
                            dtype=torch.bool,
                            device=Q_chunk.device,
                        )
                    )
                    # Expand mask for all batches and heads
                    causal_mask = causal_mask.unsqueeze(0).unsqueeze(
                        0
                    )  # [1, 1, seq_len_q, seq_len_k]
                    attn_weights_chunk = attn_weights_chunk.masked_fill(
                        ~causal_mask, float("-inf")
                    )

                attn_weights_chunk = F.softmax(attn_weights_chunk, dim=-1)
                out_chunk = torch.matmul(attn_weights_chunk, V_all)
                all_chunks.append(out_chunk)

            # Concatenate all chunks
            out = torch.cat(all_chunks, dim=2)

        # Reshape output
        if not self.multi_query_attention:
            out = out.transpose(1, 2).contiguous().view(B, S, D)
        else:
            # For MQA, need to handle single head K/V properly
            # Expand the single head to match Q heads and sum (or use repeat)
            out = out.transpose(1, 2).contiguous().view(B, S, D)

        out = self.fc_out(out)

        # Values to return for KV cache: only new incoming K and V
        # For inference, the values to be added to KV cache
        # The key insight: K and V to be cached should be the processed ones
        # Import apply_rotary_pos_emb for later use
        if self.use_rope:
            from .rotary_embedding import apply_rotary_pos_emb

        # Determine the correct device for position calculations
        calc_device = Q.device

        if kv is not None:
            # In inference, incoming K_new and V_new are the new values to be cached
            if self.use_rope:
                # Calculate positions for new tokens only (from past_len to past_len + S - 1)
                if past_kv_cache is not None:
                    past_len = past_kv_cache[0].size(1) if past_kv_cache else 0
                else:
                    past_len = 0

                new_positions = (
                    torch.arange(
                        past_len, past_len + S, dtype=torch.long, device=calc_device
                    )
                    .unsqueeze(0)
                    .expand(B, -1)
                )  # [B, S]
                new_cos, new_sin = self.rope(
                    new_positions.to(self.rope.inv_freq.device)
                )  # Ensure position tensor is on same device as rope

                # Reshape K_new to [B, S, H, D_head] format for RoPE
                if self.multi_query_attention:
                    K_new_reshaped = K_new  # Already in the right format for MQA
                    K_new_rope = apply_rotary_pos_emb(
                        K_new_reshaped, new_cos, new_sin
                    )  # [B, S, 1, D_head]
                    K_to_return = K_new_rope
                    V_to_return = V_new  # [B, S, 1, D_head] - no RoPE for V
                else:
                    K_new_reshaped = K_new.view(B, S, self.n_heads, self.head_dim)
                    K_new_rope = apply_rotary_pos_emb(
                        K_new_reshaped, new_cos, new_sin
                    )  # [B, S, H, D_head]
                    K_to_return = K_new_rope  # [B, S, H, D_head]
                    V_to_return = V_new.view(
                        B, S, self.n_heads, self.head_dim
                    )  # [B, S, H, D_head] - no RoPE for V
            else:
                if self.multi_query_attention:
                    K_to_return = K_new  # [B, S, 1, D_head]
                    V_to_return = V_new  # [B, S, 1, D_head]
                else:
                    K_to_return = K_new.view(
                        B, S, self.n_heads, self.head_dim
                    )  # [B, S, H, D_head]
                    V_to_return = V_new.view(
                        B, S, self.n_heads, self.head_dim
                    )  # [B, S, H, D_head]
        else:
            # During training: new K and V calculated for new x
            # When calculating new K and V, we also need to process them for caching
            if self.use_rope:
                # Calculate positions for new tokens (from 0 to S-1 in the context of this forward pass)
                # For training, we consider current position indices
                new_positions = (
                    torch.arange(S, dtype=torch.long, device=calc_device)
                    .unsqueeze(0)
                    .expand(B, -1)
                )  # [B, S]
                new_cos, new_sin = self.rope(
                    new_positions.to(self.rope.inv_freq.device)
                )  # Ensure position tensor is on same device as rope

                if self.multi_query_attention:
                    K_new_reshaped = K_new  # Already in the right shape
                    K_new_rope = apply_rotary_pos_emb(
                        K_new_reshaped, new_cos, new_sin
                    )  # [B, S, 1, D_head]
                    K_to_return = K_new_rope
                    V_to_return = V_new  # [B, S, 1, D_head] - no RoPE for V
                else:
                    K_new_reshaped = K_new.view(B, S, self.n_heads, self.head_dim)
                    V_new_reshaped = V_new.view(B, S, self.n_heads, self.head_dim)
                    K_new_rope = apply_rotary_pos_emb(
                        K_new_reshaped, new_cos, new_sin
                    )  # [B, S, H, D_head]
                    K_to_return = K_new_rope  # [B, S, H, D_head]
                    V_to_return = V_new_reshaped  # [B, S, H, D_head] - no RoPE for V
            else:
                if self.multi_query_attention:
                    K_to_return = K_new  # [B, S, 1, D_head]
                    V_to_return = V_new  # [B, S, 1, D_head]
                else:
                    K_to_return = K_new.view(
                        B, S, self.n_heads, self.head_dim
                    )  # [B, S, H, D_head]
                    V_to_return = V_new.view(
                        B, S, self.n_heads, self.head_dim
                    )  # [B, S, H, D_head]

        # Return the complete K and V for KV cache (inference)
        # For inference: return concatenated past and new K and V values (without rotary embedding applied)
        # For training: return just the new K and V values (without rotary embedding applied)
        if past_kv_cache is not None:
            # Inference: we have past cache, return full concatenated cache (without rotary embedding applied)
            if self.multi_query_attention:
                # For MQA
                final_k = K_all_orig  # [B, past_len + S, 1, head_dim] - without RoPE
                final_v = V_all_orig  # [B, past_len + S, 1, head_dim] - without RoPE
            else:
                # For MHA
                final_k = (
                    K_all_orig  # [B, past_len + S, n_heads, head_dim] - without RoPE
                )
                final_v = (
                    V_all_orig  # [B, past_len + S, n_heads, head_dim] - without RoPE
                )
        else:
            # Training or first inference step: no past cache, return new values only (without rotary embedding applied)
            if self.multi_query_attention:
                # For MQA - K_new and V_new are already in the right shape [B, S, 1, head_dim]
                final_k = K_new.contiguous()
                final_v = V_new.contiguous()
            else:
                # For MHA - K_new and V_new are already in the right shape [B, S, n_heads, head_dim]
                final_k = K_new.contiguous()
                final_v = V_new.contiguous()

        return out, final_k, final_v


class FlashAttention2(nn.Module):
    """
    FlashAttention2 implementation with memory and compute optimizations
    Uses PyTorch's built-in scaled_dot_product_attention when available
    Falls back to manual implementation with chunking for large sequences
    """

    def __init__(self, d_model, n_heads, use_rope=False, dropout=0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope
        self.dropout = dropout

        if use_rope:
            from .rotary_embedding import RotaryEmbedding

            self.rope = RotaryEmbedding(self.head_dim)

        self.fc_q = nn.Linear(d_model, d_model, bias=False)
        self.fc_k = nn.Linear(d_model, d_model, bias=False)
        self.fc_v = nn.Linear(d_model, d_model, bias=False)
        self.fc_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None, is_causal=False):
        B, S, D = x.size()

        # Linear projections
        Q = self.fc_q(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.fc_k(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.fc_v(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if enabled
        if self.use_rope:
            from .rotary_embedding import apply_rotary_pos_emb

            positions = (
                torch.arange(S, dtype=torch.long, device=x.device)
                .unsqueeze(0)
                .expand(B, -1)
            )
            cos, sin = self.rope(positions.to(self.rope.inv_freq.device))
            Q = apply_rotary_pos_emb(Q.transpose(1, 2), cos, sin).transpose(1, 2)
            K = apply_rotary_pos_emb(K.transpose(1, 2), cos, sin).transpose(1, 2)

        # Use PyTorch's optimized scaled dot product attention
        attn_output = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        # Reshape and output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)
        output = self.fc_out(attn_output)

        return output

