"""
Feed-Forward Network implementations for transformers
Includes SwiGLU, GeGLU, and other variants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit implementation
    Standard in modern LLMs like LLaMA, Mistral
    """

    def __init__(self, hidden_dim, d_model, multiple_of=256, ffn_dim_multiplier=None):
        super().__init__()
        self.multiple_of = multiple_of

        # Calculate intermediate dimension based on SwiGLU requirements
        # Standard SwiGLU has intermediate size of 2/3 * 4 * d_model = 8/3 * d_model
        # But we'll allow for flexibility with multiplier
        intermediate_size = int(2 * d_model * 4 / 3)
        if ffn_dim_multiplier is not None:
            intermediate_size = int(ffn_dim_multiplier * intermediate_size)
        intermediate_size = (
            (intermediate_size + self.multiple_of - 1)
            // self.multiple_of
            * self.multiple_of
        )

        self.w1 = nn.Linear(d_model, intermediate_size, bias=False)  # gate projection
        self.w2 = nn.Linear(d_model, intermediate_size, bias=False)  # up projection
        self.w3 = nn.Linear(intermediate_size, d_model, bias=False)  # down projection

    def forward(self, x):
        gate = self.w1(x)
        up = self.w2(x)
        # Swish Gate: x * SiLU(gate) = up * SiLU(gate)
        result = up * F.silu(gate)
        return self.w3(result)


class GeGLU(nn.Module):
    """
    Gated Linear Unit with GELU activation
    Alternative to SwiGLU
    """

    def __init__(self, hidden_dim, d_model, multiple_of=256, ffn_dim_multiplier=None):
        super().__init__()
        self.multiple_of = multiple_of

        intermediate_size = int(2 * d_model * 4 / 3)
        if ffn_dim_multiplier is not None:
            intermediate_size = int(ffn_dim_multiplier * intermediate_size)
        intermediate_size = (
            (intermediate_size + self.multiple_of - 1)
            // self.multiple_of
            * self.multiple_of
        )

        self.w1 = nn.Linear(d_model, intermediate_size, bias=False)  # gate projection
        self.w2 = nn.Linear(d_model, intermediate_size, bias=False)  # up projection
        self.w3 = nn.Linear(intermediate_size, d_model, bias=False)  # down projection

    def forward(self, x):
        gate = self.w1(x)
        up = self.w2(x)
        # GELU Gate: up * GELU(gate)
        result = up * F.gelu(gate)
        return self.w3(result)


class FeedForward(nn.Module):
    """
    Standard 2-layer MLP with configurable activation
    """

    def __init__(self, d_model, hidden_dim, activation="silu", dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        if activation == "relu":
            self.act_fn = F.relu
        elif activation == "gelu":
            self.act_fn = F.gelu
        elif activation == "silu":  # swish
            self.act_fn = F.silu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class MoEFeedForward(nn.Module):
    """
    Mixture of Experts implementation
    For efficient scaling with large models
    """

    def __init__(self, d_model, num_experts=8, top_k=2, expert_hidden_dim=None):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_hidden_dim = expert_hidden_dim or d_model * 4

        # Initialize experts
        self.experts = nn.ModuleList(
            [
                FeedForward(d_model, self.expert_hidden_dim, activation="silu")
                for _ in range(num_experts)
            ]
        )

        # Router network
        self.router = nn.Linear(d_model, num_experts, bias=False)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        original_shape = x.shape
        x = x.view(-1, x.size(-1))  # [batch_size * seq_len, d_model]

        # Router logits
        router_logits = self.router(x)  # [batch_size * seq_len, num_experts]

        # Get top-k experts for each token
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)

        # Apply softmax to get routing weights (only for selected experts)
        routing_weights = F.softmax(
            top_k_logits, dim=-1
        )  # [batch_size * seq_len, top_k]

        # Process all tokens through selected experts efficiently
        final_output = torch.zeros_like(x)

        # For each expert, process all tokens assigned to it
        for expert_idx in range(self.num_experts):
            # Create mask for tokens that have this expert in their top-k
            expert_mask = (top_k_indices == expert_idx).any(
                dim=-1
            )  # [batch_size * seq_len]

            if expert_mask.any():
                # Get input tokens for this expert
                expert_input = x[expert_mask]  # [num_tokens_for_expert, d_model]

                # Get positions of this expert in top-k for these tokens
                token_positions = torch.where(top_k_indices[expert_mask] == expert_idx)[
                    1
                ]  # [num_tokens_for_expert]

                # Get corresponding routing weights
                expert_weights = routing_weights[
                    expert_mask, token_positions
                ].unsqueeze(
                    -1
                )  # [num_tokens_for_expert, 1]

                # Process through the expert
                expert_output = self.experts[expert_idx](expert_input)

                # Apply routing weights
                weighted_output = expert_output * expert_weights

                # Add to final output
                final_output[expert_mask] += weighted_output

        final_output = self.dropout(final_output)
        return final_output.view(original_shape)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    More stable than Layer Norm in deep models
    """

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # Compute RMS
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return self.weight * (x / rms)


class ParallelTransformerFFN(nn.Module):
    """
    Parallel computation of attention and FFN
    Used in some modern architectures for efficiency
    """

    def __init__(self, d_model, hidden_dim, activation="silu", dropout=0.0):
        super().__init__()
        self.attention_proj = nn.Linear(d_model, d_model, bias=False)
        self.ffn = FeedForward(d_model, hidden_dim, activation, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_output):
        # Parallel computation
        ffn_output = self.ffn(x)
        combined_output = attention_output + ffn_output
        return self.dropout(combined_output)

