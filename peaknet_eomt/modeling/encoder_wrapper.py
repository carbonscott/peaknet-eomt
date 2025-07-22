# Copyright (c) 2025 PeakNet EoMT Implementation
# Licensed under the MIT License.
#
# EoMT-compatible wrapper for standalone DINOv2 models

from typing import Optional, Union, Tuple, Dict, Any
import torch
import torch.nn as nn
from types import SimpleNamespace

from .dinov2.vision_transformer import DinoVisionTransformer, MemEffAttention, Block
from .dinov2.layers import PatchEmbed
from functools import partial


def get_standard_vit_configs() -> Dict[str, Dict[str, Any]]:
    """Get standard ViT configuration presets."""
    return {
        "vit_small": {
            "embed_dim": 384,
            "depth": 12,
            "num_heads": 6,
            "mlp_ratio": 4.0,
        },
        "vit_base": {
            "embed_dim": 768, 
            "depth": 12,
            "num_heads": 12,
            "mlp_ratio": 4.0,
        },
        "vit_large": {
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
            "mlp_ratio": 4.0,
        },
        "vit_giant": {
            "embed_dim": 1536,
            "depth": 40, 
            "num_heads": 24,
            "mlp_ratio": 4.0,
            "ffn_layer": "swiglufused",
        }
    }


def config_to_pretrained_name(
    embed_dim: int,
    depth: int, 
    num_heads: int,
    patch_size: int = 14,
    num_register_tokens: int = 0,
    ffn_layer: str = "mlp"
) -> Optional[str]:
    """
    Map ViT configuration to pretrained model name.
    
    Returns the DINOv2 pretrained model name that matches the given configuration,
    or None if no standard pretrained model matches.
    """
    
    # Standard DINOv2 configurations
    configs = {
        # ViT-Small
        (384, 12, 6, 14, 0, "mlp"): "dinov2_vits14",
        (384, 12, 6, 14, 4, "mlp"): "dinov2_vits14_reg",
        
        # ViT-Base  
        (768, 12, 12, 14, 0, "mlp"): "dinov2_vitb14",
        (768, 12, 12, 14, 4, "mlp"): "dinov2_vitb14_reg",
        
        # ViT-Large
        (1024, 24, 16, 14, 0, "mlp"): "dinov2_vitl14", 
        (1024, 24, 16, 14, 4, "mlp"): "dinov2_vitl14_reg",
        
        # ViT-Giant
        (1536, 40, 24, 14, 0, "swiglufused"): "dinov2_vitg14",
        (1536, 40, 24, 14, 4, "swiglufused"): "dinov2_vitg14_reg",
    }
    
    config_key = (embed_dim, depth, num_heads, patch_size, num_register_tokens, ffn_layer)
    return configs.get(config_key)


class EoMTDinoV2Encoder(nn.Module):
    """
    EoMT-compatible wrapper for standalone DINOv2 models.
    
    This wrapper provides the same interface that EoMT expects from TIMM models,
    allowing seamless replacement without changing EoMT code.
    """
    
    def __init__(
        self,
        # Core ViT architecture parameters
        patch_size: int = 14,
        embed_dim: int = 1024,
        depth: int = 24, 
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        
        # DINOv2 specific parameters
        num_register_tokens: int = 4,
        ffn_layer: str = "mlp",
        init_values: Optional[float] = 1.0,
        
        # Training parameters  
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        qkv_bias: bool = True,
        ffn_bias: bool = True,
        proj_bias: bool = True,
        
        # Input parameters
        img_size: Union[int, Tuple[int, int]] = 518,
        in_chans: int = 3,
        
        # Pretrained weights
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
        
        **kwargs
    ):
        """
        Initialize EoMT-compatible DINOv2 encoder from configuration.
        
        Args:
            patch_size: Patch size for patch embedding
            embed_dim: Embedding dimension (hidden size)
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            num_register_tokens: Number of register tokens (0 for none)
            ffn_layer: FFN layer type ("mlp" or "swiglufused")
            init_values: LayerScale initialization values
            drop_path_rate: Stochastic depth rate
            drop_path_uniform: Whether to use uniform drop rate
            img_size: Input image size
            pretrained: Whether to load pretrained weights
            pretrained_path: Explicit path to pretrained weights
        """
        super().__init__()
        
        # Store configuration
        self.config = {
            "patch_size": patch_size,
            "embed_dim": embed_dim, 
            "depth": depth,
            "num_heads": num_heads,
            "mlp_ratio": mlp_ratio,
            "num_register_tokens": num_register_tokens,
            "ffn_layer": ffn_layer,
            "init_values": init_values,
            "drop_path_rate": drop_path_rate,
            "drop_path_uniform": drop_path_uniform,
            "qkv_bias": qkv_bias,
            "ffn_bias": ffn_bias,
            "proj_bias": proj_bias,
            "img_size": img_size,
            "in_chans": in_chans,
        }
        
        # Create DINOv2 model directly
        self.backbone = DinoVisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans, 
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            ffn_bias=ffn_bias,
            proj_bias=proj_bias,
            drop_path_rate=drop_path_rate,
            drop_path_uniform=drop_path_uniform,
            init_values=init_values,
            embed_layer=PatchEmbed,
            act_layer=nn.GELU,
            block_fn=partial(Block, attn_class=MemEffAttention),
            ffn_layer=ffn_layer,
            block_chunks=0,  # Don't chunk blocks - EoMT needs individual block access
            num_register_tokens=num_register_tokens,
            interpolate_antialias=num_register_tokens > 0,  # Use antialias with register tokens
            interpolate_offset=0.0 if num_register_tokens > 0 else 0.1,
        )
        
        # Load pretrained weights if requested
        if pretrained or pretrained_path:
            self._load_pretrained_weights(pretrained_path)
        
        # Setup EoMT-compatible interface
        self._setup_eomt_interface()
        
        # Setup pixel normalization (using DINOv2 defaults)
        pixel_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)
        
        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)
        
    def _load_pretrained_weights(self, pretrained_path: Optional[str] = None):
        """Load pretrained weights either from explicit path or auto-detected model."""
        
        if pretrained_path:
            # Load from explicit path
            state_dict = torch.load(pretrained_path, map_location="cpu")
            self.backbone.load_state_dict(state_dict, strict=True)
            print(f"Loaded pretrained weights from {pretrained_path}")
        else:
            # Auto-detect pretrained model from configuration
            pretrained_name = config_to_pretrained_name(
                embed_dim=self.config["embed_dim"],
                depth=self.config["depth"],
                num_heads=self.config["num_heads"],
                patch_size=self.config["patch_size"],
                num_register_tokens=self.config["num_register_tokens"],
                ffn_layer=self.config["ffn_layer"]
            )
            
            if pretrained_name:
                # Load from DINOv2 hub
                from .dinov2.hub.utils import _DINOV2_BASE_URL, _make_dinov2_model_name
                
                # Map our name to hub naming convention
                arch_name_map = {
                    "dinov2_vits14": ("vit_small", 14),
                    "dinov2_vitb14": ("vit_base", 14),
                    "dinov2_vitl14": ("vit_large", 14),
                    "dinov2_vitg14": ("vit_giant2", 14),
                    "dinov2_vits14_reg": ("vit_small", 14),
                    "dinov2_vitb14_reg": ("vit_base", 14),
                    "dinov2_vitl14_reg": ("vit_large", 14),
                    "dinov2_vitg14_reg": ("vit_giant2", 14),
                }
                
                if pretrained_name in arch_name_map:
                    arch_name, patch_size = arch_name_map[pretrained_name]
                    model_base_name = _make_dinov2_model_name(arch_name, patch_size)
                    model_full_name = _make_dinov2_model_name(arch_name, patch_size, self.config["num_register_tokens"])
                    url = f"{_DINOV2_BASE_URL}/{model_base_name}/{model_full_name}_pretrain.pth"
                    
                    try:
                        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
                        self.backbone.load_state_dict(state_dict, strict=True)
                        print(f"Loaded pretrained DINOv2 weights: {pretrained_name}")
                    except Exception as e:
                        print(f"Warning: Could not load pretrained weights for {pretrained_name}: {e}")
                        print("Continuing with random initialization.")
                else:
                    print(f"Warning: Unknown pretrained model {pretrained_name}")
            else:
                print("Warning: No matching pretrained model found for this configuration.")
                print("Available configurations:")
                configs = get_standard_vit_configs()
                for name, config in configs.items():
                    print(f"  {name}: {config}")
                print("Continuing with random initialization.")
        
    def _setup_eomt_interface(self):
        """Setup the interface attributes that EoMT expects."""
        
        # Core attributes
        self.backbone.embed_dim = self.backbone.embed_dim
        self.backbone.num_prefix_tokens = 1 + self.backbone.num_register_tokens  # CLS + register tokens
        
        # Store original patch embed and add the expected attributes directly to it
        self.backbone.patch_embed.grid_size = self.backbone.patch_embed.patches_resolution
        
        # Add missing methods that EoMT expects
        self.backbone._pos_embed = self._pos_embed
        self.backbone.patch_drop = nn.Identity()  # DINOv2 doesn't use patch dropout
        self.backbone.norm_pre = nn.Identity()   # DINOv2 doesn't use pre-norm
        
    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Position embedding method that EoMT expects.
        
        Args:
            x: Patch tokens [B, N, D]
            
        Returns:
            Position-embedded tokens [B, N+num_prefix_tokens, D]
        """
        B, N, D = x.shape
        
        # Add CLS token
        cls_tokens = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embeddings (with interpolation support)
        # We need to extract image size from somewhere - let's assume square patches
        patch_size = self.backbone.patch_embed.patch_size[0]
        img_h = img_w = int((N ** 0.5) * patch_size)
        pos_embed = self.backbone.interpolate_pos_encoding(x, img_w, img_h)
        x = x + pos_embed
        
        # Add register tokens if present
        if self.backbone.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],  # CLS token
                    self.backbone.register_tokens.expand(B, -1, -1),  # Register tokens
                    x[:, 1:],  # Patch tokens  
                ),
                dim=1,
            )
            
        return x
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - just return the backbone output.
        EoMT will handle the specific processing it needs.
        """
        return self.backbone.forward_features(x)


# Convenience functions for creating encoders
def create_dinov2_encoder(
    # Core ViT architecture parameters
    patch_size: int = 14,
    embed_dim: int = 1024,
    depth: int = 24,
    num_heads: int = 16,
    mlp_ratio: float = 4.0,
    
    # DINOv2 specific parameters  
    num_register_tokens: int = 4,
    ffn_layer: str = "mlp",
    init_values: Optional[float] = 1.0,
    
    # Training parameters
    drop_path_rate: float = 0.0,
    
    # Input parameters
    img_size: Union[int, Tuple[int, int]] = 518,
    
    # Pretrained weights
    pretrained: bool = True,
    **kwargs
) -> EoMTDinoV2Encoder:
    """
    Create a DINOv2 encoder from configuration parameters.
    
    Args:
        patch_size: Patch size for patch embedding
        embed_dim: Embedding dimension (hidden size)
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        num_register_tokens: Number of register tokens (0 for none)
        ffn_layer: FFN layer type ("mlp" or "swiglufused")
        init_values: LayerScale initialization values
        drop_path_rate: Stochastic depth rate
        img_size: Input image size
        pretrained: Whether to load pretrained weights
        **kwargs: Additional model arguments
        
    Returns:
        EoMT-compatible DINOv2 encoder
    """
    
    return EoMTDinoV2Encoder(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        num_register_tokens=num_register_tokens,
        ffn_layer=ffn_layer,
        init_values=init_values,
        drop_path_rate=drop_path_rate,
        img_size=img_size,
        pretrained=pretrained,
        **kwargs
    )


# Helper functions for standard configurations
def vit_small_config(**overrides) -> Dict[str, Any]:
    """ViT-Small configuration."""
    config = {
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "mlp_ratio": 4.0,
        "ffn_layer": "mlp",
    }
    config.update(overrides)
    return config


def vit_base_config(**overrides) -> Dict[str, Any]:
    """ViT-Base configuration."""
    config = {
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "ffn_layer": "mlp",
    }
    config.update(overrides)
    return config


def vit_large_config(**overrides) -> Dict[str, Any]:
    """ViT-Large configuration."""
    config = {
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "ffn_layer": "mlp",
    }
    config.update(overrides)
    return config


def vit_giant_config(**overrides) -> Dict[str, Any]:
    """ViT-Giant configuration."""
    config = {
        "embed_dim": 1536,
        "depth": 40,
        "num_heads": 24,
        "mlp_ratio": 4.0,
        "ffn_layer": "swiglufused",
    }
    config.update(overrides)
    return config


# Legacy support function (deprecated)
def create_dinov2_encoder_legacy(
    model_name: str = "vit_large_patch14_reg4_dinov2",
    img_size: Union[int, Tuple[int, int]] = 518,
    pretrained: bool = True,
    **kwargs
) -> EoMTDinoV2Encoder:
    """
    Legacy function for TIMM-style model naming (deprecated).
    
    Use create_dinov2_encoder() with explicit config parameters instead.
    """
    import warnings
    warnings.warn(
        "create_dinov2_encoder_legacy is deprecated. "
        "Use create_dinov2_encoder() with explicit configuration parameters.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Map TIMM-style names to configurations
    name_to_config = {
        "vit_small_patch14_dinov2": vit_small_config(num_register_tokens=0),
        "vit_base_patch14_dinov2": vit_base_config(num_register_tokens=0),
        "vit_large_patch14_dinov2": vit_large_config(num_register_tokens=0),
        "vit_giant_patch14_dinov2": vit_giant_config(num_register_tokens=0),
        "vit_small_patch14_reg4_dinov2": vit_small_config(num_register_tokens=4),
        "vit_base_patch14_reg4_dinov2": vit_base_config(num_register_tokens=4),
        "vit_large_patch14_reg4_dinov2": vit_large_config(num_register_tokens=4),
        "vit_giant_patch14_reg4_dinov2": vit_giant_config(num_register_tokens=4),
    }
    
    if model_name not in name_to_config:
        raise ValueError(f"Unknown model: {model_name}. Use explicit configuration instead.")
    
    config = name_to_config[model_name]
    config.update(kwargs)
    
    return create_dinov2_encoder(
        img_size=img_size,
        pretrained=pretrained,
        **config
    )