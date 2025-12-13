"""
LoRA (Low-Rank Adaptation) implementation for parameter-efficient fine-tuning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
import math


class LoRALinear(nn.Module):
    """
    LoRA implementation for linear layers
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: int = 16,
        dropout: float = 0.0,
        merge_weights: bool = True,
        bias: bool = False
    ):
        super(LoRALinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.merge_weights = merge_weights

        # Standard linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # LoRA parameters
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        if dropout > 0:
            self.lora_dropout = nn.Dropout(dropout)
        else:
            self.lora_dropout = lambda x: x

        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters"""
        if hasattr(self.linear, 'reset_parameters'):
            self.linear.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation
        """
        # Standard linear transformation
        out = self.linear(x)

        # Add LoRA adaptation
        lora_input = self.lora_dropout(x)
        lora_out = torch.matmul(
            torch.matmul(lora_input, self.lora_A.transpose(0, 1)),
            self.lora_B.transpose(0, 1)
        )
        lora_out = lora_out * self.scaling

        return out + lora_out

    def merge_weights(self):
        """Merge LoRA weights with base weights"""
        if not self.merge_weights:
            # Calculate delta weight
            delta_weight = torch.matmul(self.lora_B, self.lora_A) * self.scaling
            self.linear.weight.data += delta_weight
            self.merge_weights = True

    def unmerge_weights(self):
        """Unmerge LoRA weights from base weights"""
        if self.merge_weights:
            # Calculate delta weight
            delta_weight = torch.matmul(self.lora_B, self.lora_A) * self.scaling
            self.linear.weight.data -= delta_weight
            self.merge_weights = False


class LoRAConfig:
    """
    Configuration for LoRA adaptation
    """
    def __init__(
        self,
        r: int = 16,
        alpha: int = 16,
        dropout: float = 0.0,
        target_modules: List[str] = ["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj"],
        merge_weights: bool = False
    ):
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules
        self.merge_weights = merge_weights


def apply_lora_to_model(model: nn.Module, lora_config: LoRAConfig):
    """
    Apply LoRA to a model by replacing target modules with LoRA modules
    """
    for name, module in model.named_modules():
        # Check if this module is a target for LoRA
        if any(target in name for target in lora_config.target_modules):
            if isinstance(module, nn.Linear):
                # Replace the linear layer with a LoRA layer
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]

                # Get parent module
                parent_module = model
                if parent_name:
                    for p_name in parent_name.split("."):
                        parent_module = getattr(parent_module, p_name)

                # Create LoRA layer
                lora_layer = LoRALinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    rank=lora_config.r,
                    alpha=lora_config.alpha,
                    dropout=lora_config.dropout,
                    merge_weights=lora_config.merge_weights
                )

                # Copy original weights
                lora_layer.linear.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    lora_layer.linear.bias.data = module.bias.data.clone()

                # Replace the module
                setattr(parent_module, child_name, lora_layer)

    return model


class LoRATrainer:
    """
    Trainer class that supports LoRA fine-tuning
    """
    def __init__(self, model: nn.Module, config: LoRAConfig, **trainer_kwargs):
        self.config = config
        
        # Apply LoRA to the model
        self.model = apply_lora_to_model(model, config)
        
        # Freeze base parameters (only train LoRA parameters)
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Get LoRA parameters for training
        self.lora_params = [p for n, p in self.model.named_parameters() if "lora_" in n]
        
        print(f"LoRA applied. Training {len(self.lora_params)} LoRA parameters out of {sum(p.numel() for p in self.model.parameters())} total parameters.")

    def get_optimizer(self, lr: float = 1e-4, **kwargs):
        """
        Get optimizer for LoRA training
        """
        return torch.optim.AdamW(self.lora_params, lr=lr, **kwargs)

    def save_lora_weights(self, save_path: str):
        """
        Save only LoRA weights
        """
        lora_state_dict = {}
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                lora_state_dict[name] = param.data
        
        torch.save(lora_state_dict, save_path)

    def load_lora_weights(self, load_path: str):
        """
        Load LoRA weights
        """
        lora_state_dict = torch.load(load_path, map_location='cpu')
        self.model.load_state_dict(lora_state_dict, strict=False)

    def merge_and_save_full_model(self, model, save_path: str):
        """
        Merge LoRA weights into base model and save full model
        """
        # Temporarily merge weights
        for name, module in self.model.named_modules():
            if isinstance(module, LoRALinear):
                module.merge_weights()
        
        # Save the model with merged weights
        torch.save(model.state_dict(), save_path)
        
        # Unmerge if needed
        for name, module in self.model.named_modules():
            if isinstance(module, LoRALinear):
                module.unmerge_weights()


def create_lora_finetune_command(args):
    """
    Create a fine-tuning command using LoRA
    """
    from model.transformer import Config, Transformer
    from training.pretrain import SFTTrainer
    
    # Load model and config
    config = Config.load(args.config)
    model = Transformer(config)
    
    if args.model_path and hasattr(args, 'model_path') and args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    
    # Set up LoRA configuration
    lora_config = LoRAConfig(
        r=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=args.target_modules.split(",") if hasattr(args, 'target_modules') and args.target_modules else ["q_proj", "v_proj"]
    )
    
    # Apply LoRA to model
    lora_trainer = LoRATrainer(model, lora_config)
    
    # Create a fine-tuning trainer with LoRA settings
    trainer = SFTTrainer(
        model=lora_trainer.model,
        config=config,
        train_data=None,  # Will be provided by main function
        eval_data=None,   # Will be provided by main function
        optimizer_name=args.optimizer_name,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        save_dir=args.output_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    return trainer