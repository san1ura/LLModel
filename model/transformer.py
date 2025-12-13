"""
Main transformer model implementation
This contains the core Transformer architecture with all components integrated
"""

import json
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class TransformerBlock(nn.Module):
    """
    Single transformer block with attention and feedforward layers
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        use_gradient_checkpointing: bool = False,
        use_rope: bool = False,
        attention_type: str = "standard",  # 'standard', 'flash2'
        use_parallel_ffn: bool = False,
        norm_first: bool = True,  # Pre-norm vs post-norm
    ):
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.norm_first = norm_first

        # Choose attention mechanism
        if attention_type == "flash2":
            from model.layers.attention import FlashAttention2

            self.attention = FlashAttention2(d_model, n_heads, use_rope=use_rope)
        else:
            from model.layers.attention import MHA

            self.attention = MHA(d_model, n_heads, use_rope=use_rope)

        # Feedforward network
        from model.layers.feedforward import SwiGLU

        self.ff = SwiGLU(d_ff, d_model)

        # Normalization layers
        from model.layers.feedforward import RMSNorm

        self.norm_attn = RMSNorm(d_model)
        self.norm_ff = RMSNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, past_kv_cache=None):
        # Pre-normalization (more stable)
        if self.norm_first:
            # Attention branch
            attn_input = self.norm_attn(x)
            attn_output, new_k, new_v = self.attention(
                attn_input, mask, past_kv_cache=past_kv_cache
            )
            x = x + self.dropout(attn_output)

            # FFN branch
            ffn_input = self.norm_ff(x)
            ffn_output = self.ff(ffn_input)
            x = x + self.dropout(ffn_output)
        else:
            # Post-normalization
            attn_output, new_k, new_v = self.attention(
                x, mask, past_kv_cache=past_kv_cache
            )
            x = self.norm_attn(x + self.dropout(attn_output))

            ffn_output = self.ff(x)
            x = self.norm_ff(x + self.dropout(ffn_output))

        return x, new_k, new_v


class Transformer(nn.Module):
    """
    Main transformer implementation with modern architecture features
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.max_len = config.max_len

        # Token embeddings
        self.embed = nn.Embedding(config.vocab_size, config.d_model)

        # Positional embeddings - support multiple types
        self.pos_type = getattr(config, "pos_type", "rope")  # Default to RoPE
        if self.pos_type == "absolute":
            self.pos_embed = nn.Embedding(config.max_len, config.d_model)
        elif self.pos_type == "sinusoidal":
            # Create sinusoidal positional embeddings
            self.register_buffer(
                "pos_embed",
                self._create_sinusoidal_pos_embed(config.max_len, config.d_model),
            )
        elif self.pos_type == "alibi":
            from model.layers.rotary_embedding import ALiBi

            self.alibi = ALiBi(config.n_heads, config.max_len)
        # RoPE is handled in attention layers

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    config.d_model,
                    config.n_heads,
                    config.d_ff,
                    dropout=config.dropout,
                    use_gradient_checkpointing=config.use_gradient_checkpointing,
                    use_rope=config.use_rope,
                    attention_type=getattr(config, "attention_type", "standard"),
                    norm_first=getattr(config, "norm_first", True),
                )
                for _ in range(config.n_layers)
            ]
        )

        # Final normalization
        from model.layers.feedforward import RMSNorm

        self.final_norm = RMSNorm(config.d_model)

        # Output projection
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights
        self._init_weights()

    def _create_sinusoidal_pos_embed(self, max_len, d_model):
        """Create sinusoidal positional embeddings"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _init_weights(self):
        """Initialize model weights following modern best practices"""
        with torch.no_grad():
            # Embedding layer
            nn.init.normal_(
                self.embed.weight, mean=0.0, std=self.config.initializer_range
            )

            # Linear layers with specific initialization for different types
            for name, module in self.named_modules():
                if isinstance(module, nn.Linear):
                    # Use different initialization for different types of linear layers
                    std = self.config.initializer_range

                    # For attention projections, we may want to scale differently
                    if (
                        "attention" in name
                        or "attn" in name
                        or "q_proj" in name
                        or "k_proj" in name
                        or "v_proj" in name
                    ):
                        # For attention projections, potentially use a different scaling
                        # This can help with gradient stability
                        std = self.config.initializer_range / math.sqrt(
                            2 * self.config.n_layers
                        )

                    # For output projection of attention
                    if "fc_out" in name or "o_proj" in name:
                        # Output projection of attention layers often uses a smaller std
                        std = self.config.initializer_range / math.sqrt(
                            self.config.n_layers
                        )

                    # For feedforward layers
                    if (
                        "ff" in name
                        or "mlp" in name
                        or "up_proj" in name
                        or "gate_proj" in name
                    ):
                        # Standard init for feedforward layers
                        std = self.config.initializer_range

                    # For final output layer
                    if "output" in name:
                        # Standard init for final output layer
                        std = self.config.initializer_range

                    nn.init.normal_(module.weight, mean=0.0, std=std)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

            # Special initialization for QKV matrices if they exist
            # Note: In current implementation, attention layers handle their own initialization
            # but we can ensure proper initialization here if needed

            # Initialize layer norms
            for module in self.modules():
                if hasattr(module, "weight") and isinstance(
                    module, (nn.LayerNorm, getattr(torch, "nn", object()).RMSNorm)
                ):
                    nn.init.ones_(module.weight)
                if hasattr(module, "bias") and isinstance(
                    module, (nn.LayerNorm, getattr(torch, "nn", object()).RMSNorm)
                ):
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, mask=None, past_kv_caches=None):
        """Forward pass through the transformer"""
        batch_size, seq_len = input_ids.shape

        # Input validation
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum {self.max_len}"
            )

        # Token embeddings
        x = self.embed(input_ids)

        # Add positional embeddings based on type
        if self.pos_type == "absolute":
            pos_ids = (
                torch.arange(seq_len, device=input_ids.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            pos_embeds = self.pos_embed(pos_ids)
            x = x + pos_embeds
        elif self.pos_type == "sinusoidal":
            # Use precomputed sinusoidal embeddings for the sequence length
            pos_embeds = self.pos_embed[:seq_len, :].unsqueeze(
                0
            )  # [1, seq_len, d_model]
            x = x + pos_embeds

        # Process through transformer layers
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            past_cache = past_kv_caches[i] if past_kv_caches is not None else None

            if self.config.use_gradient_checkpointing and self.training:
                # Use gradient checkpointing to reduce memory usage
                def create_custom_forward(layer_module):
                    def custom_forward(*inputs):
                        return layer_module(*inputs)

                    return custom_forward

                layer_inputs = (x, mask, past_cache)
                custom_forward_func = create_custom_forward(layer)
                x, new_k, new_v = torch.utils.checkpoint(
                    custom_forward_func, *layer_inputs, use_reentrant=False
                )
            else:
                x, new_k, new_v = layer(x, mask, past_cache)

            new_kv_caches.append((new_k, new_v))

        # Final normalization and output
        x = self.final_norm(x)
        logits = self.output(x)

        # Ensure logits are the same length as input_ids for compatibility with labels
        if logits.size(1) != seq_len:
            # Adjust logits to match input length if needed
            min_len = min(logits.size(1), seq_len)
            logits = logits[:, :min_len, :]

        return logits, new_kv_caches

    def save(self, filepath: str):
        """
        Save the model state dict to a file
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config.to_dict()
        }, filepath)

    @classmethod
    def load(cls, filepath: str, config=None):
        """
        Load a model from a file
        """
        checkpoint = torch.load(filepath, map_location='cpu')

        # If config is not provided, reconstruct it from saved config
        if config is None:
            saved_config = checkpoint.get('config', {})
            # Import here to avoid circular import
            from model.transformer import Config
            config = Config(**saved_config)

        # Create model instance
        model = cls(config)

        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])

        return model

    def generate(
        self,
        input_ids,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: int = 0,
        use_cache: bool = True,
        repetition_penalty: float = 1.0,
    ):
        """
        Generate tokens autoregressively
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Initialize tracking variables
        generated = input_ids
        past_kv_caches = None if use_cache else []

        for _ in range(max_new_tokens):
            # Prepare input - only use the last token if using cache
            if use_cache and past_kv_caches is not None and generated.shape[1] > 1:
                input_segment = generated[:, -1:]
            else:
                input_segment = generated

            # Forward pass
            logits, past_kv_caches = self(input_segment, past_kv_caches=past_kv_caches)

            # Extract logits for the last token
            next_token_logits = logits[:, -1, :] / temperature

            # Apply repetition penalty
            if repetition_penalty != 1.0 and generated.size(1) > 0:
                # Apply penalty to tokens that have already appeared
                for batch_idx in range(batch_size):
                    for token_id in set(generated[batch_idx].tolist()):
                        next_token_logits[batch_idx, token_id] /= repetition_penalty

            # Apply sampling strategies
            if do_sample:
                if top_k is not None and top_k > 0:
                    # Top-k sampling
                    top_k_logits, top_k_indices = torch.topk(
                        next_token_logits, min(top_k, next_token_logits.size(-1))
                    )
                    mask = torch.full_like(next_token_logits, float("-inf"))
                    # Ensure indices and values have matching dimensions for scatter
                    mask.scatter_(1, top_k_indices.long(), top_k_logits)
                    next_token_logits = mask

                if top_p is not None and 0 < top_p < 1.0:
                    # Top-p (nucleus) sampling
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True, dim=-1
                    )
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )

                    # Remove tokens with cumulative probability above top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = False

                    # Create a mask for all logits to set unwanted indices to -inf
                    # Create a boolean mask for the entire logits tensor
                    top_p_mask = torch.zeros_like(next_token_logits, dtype=torch.bool)
                    # For each batch, we need to mark the indices to remove
                    for batch_idx in range(next_token_logits.size(0)):
                        # Get the indices to remove for this batch
                        indices_to_remove = sorted_indices[batch_idx][
                            sorted_indices_to_remove[batch_idx]
                        ]
                        # Mark these positions in the mask
                        top_p_mask[batch_idx].scatter_(
                            0, indices_to_remove.long(), True
                        )

                    next_token_logits.masked_fill_(top_p_mask, float("-inf"))

                # Sample from the distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding (take the token with highest probability)
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Check for end-of-sequence tokens
            if eos_token_id is not None:
                if (next_tokens == eos_token_id).all():
                    break

            # Append the generated tokens
            generated = torch.cat([generated, next_tokens], dim=-1)

            # Stop early if all sequences have hit the EOS token
            if eos_token_id is not None:
                # Check if all newly generated tokens are EOS
                if (next_tokens == eos_token_id).all():
                    break

        return generated

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:
        """
        Resizes input token embeddings if new_num_tokens != config.vocab_size
        """
        if new_num_tokens is None:
            return self.embed

        # Resize the embedding layer
        old_num_tokens, old_embedding_dim = self.embed.weight.size()
        if new_num_tokens == old_num_tokens:
            return self.embed

        # Create a new embedding layer with the new size
        new_embed = nn.Embedding(new_num_tokens, old_embedding_dim)
        new_embed.to(self.embed.weight.device)

        # Copy the weights from the old embedding layer to the new one
        # Only copy up to the minimum of old and new sizes to handle both expansion and shrinkage
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embed.weight.data[:num_tokens_to_copy, :] = self.embed.weight.data[:num_tokens_to_copy, :]

        # Initialize any new embeddings
        if new_num_tokens > old_num_tokens:
            # Initialize new embeddings with the same distribution as existing ones
            nn.init.normal_(
                new_embed.weight.data[old_num_tokens:new_num_tokens, :],
                mean=0.0,
                std=self.config.initializer_range
            )

        # Replace the old embedding layer with the new one
        self.embed = new_embed

        # Also resize the output layer to match new vocab size
        old_output_dim = self.output.out_features
        if new_num_tokens != old_output_dim:
            # Create a new output layer with the new size
            new_output = nn.Linear(self.output.in_features, new_num_tokens, bias=False)
            new_output.to(self.output.weight.device)

            # Copy the weights from the old output layer to the new one
            num_tokens_to_copy = min(old_output_dim, new_num_tokens)
            new_output.weight.data[:num_tokens_to_copy, :] = self.output.weight.data[:num_tokens_to_copy, :]

            # Replace the old output layer with the new one
            self.output = new_output

        # Update vocab size in config
        self.config.vocab_size = new_num_tokens
        self.vocab_size = new_num_tokens

        return self.embed


class Config:
    """
    Model configuration with modern defaults
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 4096,
        n_layers: int = 32,
        n_heads: int = 32,
        d_ff: int = 11008,  # Standard FF size for SwiGLU
        max_len: int = 4096,
        dropout: float = 0.0,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        use_rope: bool = True,
        pos_type: str = "rope",  # 'rope', 'alibi', 'absolute'
        use_gradient_checkpointing: bool = False,
        attention_type: str = "standard",  # 'standard', 'flash2'
        norm_first: bool = True,  # Pre-norm vs post-norm
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_parallel_ffn: bool = False,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_len = max_len
        self.dropout = dropout
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.use_rope = use_rope
        self.pos_type = pos_type
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.attention_type = attention_type
        self.norm_first = norm_first
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_parallel_ffn = use_parallel_ffn
        self.tie_word_embeddings = tie_word_embeddings

        # Store any additional config values
        for key, value in kwargs.items():
            setattr(self, key, value)

    def save(self, file_path: str):
        """Save configuration to file"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, file_path: str):
        """Load configuration from file"""
        with open(file_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def to_dict(self):
        """Convert to dictionary"""
        return self.__dict__.copy()

