# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# EoMT implementation with standalone DINOv2 encoder
# ---------------------------------------------------------------

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .scale_block import ScaleBlock
from .encoder_wrapper import create_dinov2_encoder


class EoMT(nn.Module):
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
        
        # Input parameters
        img_size: tuple = (518, 518),
        
        # EoMT specific parameters
        num_classes: int = 150,
        num_q: int = 100,
        num_blocks: int = 4,
        masked_attn_enabled: bool = True,
        
        # Pretrained weights
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        
        # Create the DINOv2 encoder with direct configuration
        self.encoder = create_dinov2_encoder(
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
            pretrained_path=pretrained_path,
        )
        
        self.num_q = num_q
        self.num_blocks = num_blocks
        self.masked_attn_enabled = masked_attn_enabled

        self.register_buffer("attn_mask_probs", torch.ones(num_blocks))

        self.q = nn.Embedding(num_q, self.encoder.backbone.embed_dim)

        self.class_head = nn.Linear(self.encoder.backbone.embed_dim, num_classes + 1)

        self.mask_head = nn.Sequential(
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
            nn.GELU(),
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
            nn.GELU(),
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
        )

        patch_size = self.encoder.backbone.patch_embed.patch_size
        max_patch_size = max(patch_size[0], patch_size[1])
        num_upscale = max(1, int(math.log2(max_patch_size)) - 2)

        self.upscale = nn.Sequential(
            *[ScaleBlock(self.encoder.backbone.embed_dim) for _ in range(num_upscale)],
        )

    def _predict(self, x: torch.Tensor):
        q = x[:, : self.num_q, :]

        class_logits = self.class_head(q)

        # Extract patch tokens (skip queries and prefix tokens)
        patch_tokens = x[:, self.num_q + self.encoder.backbone.num_prefix_tokens :, :]
        
        # Debug: print shapes to understand the issue
        # print(f"Debug: x.shape = {x.shape}")
        # print(f"Debug: patch_tokens.shape = {patch_tokens.shape}")
        # print(f"Debug: grid_size = {self.encoder.backbone.patch_embed.grid_size}")
        # print(f"Debug: num_q = {self.num_q}, num_prefix_tokens = {self.encoder.backbone.num_prefix_tokens}")
        
        # Reshape patch tokens to spatial format [B, D, H, W]
        B, N_patches, D = patch_tokens.shape
        H, W = self.encoder.backbone.patch_embed.grid_size
        
        # Verify the dimensions match
        expected_patches = H * W
        if N_patches != expected_patches:
            raise ValueError(
                f"Patch token count mismatch: got {N_patches} tokens, expected {expected_patches} "
                f"(grid_size={H}x{W}). Check if num_q={self.num_q} and "
                f"num_prefix_tokens={self.encoder.backbone.num_prefix_tokens} are correct."
            )
        
        x_spatial = patch_tokens.transpose(1, 2).reshape(B, D, H, W)

        mask_logits = torch.einsum(
            "bqc, bchw -> bqhw", self.mask_head(q), self.upscale(x_spatial)
        )

        return mask_logits, class_logits

    @torch.compiler.disable
    def _disable_attn_mask(self, attn_mask, prob):
        if prob < 1:
            random_queries = (
                torch.rand(attn_mask.shape[0], self.num_q, device=attn_mask.device)
                > prob
            )
            attn_mask[
                :, : self.num_q, self.num_q + self.encoder.backbone.num_prefix_tokens :
            ][random_queries] = True

        return attn_mask

    def _attn(self, module: nn.Module, x: torch.Tensor, mask: Optional[torch.Tensor]):
        B, N, C = x.shape
        
        # Handle DINOv2 attention interface
        head_dim = C // module.num_heads
        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        
        # DINOv2 doesn't have q_norm, k_norm - skip if not available
        if hasattr(module, 'q_norm') and hasattr(module, 'k_norm'):
            q, k = module.q_norm(q), module.k_norm(k)

        if mask is not None:
            mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)

        # Handle dropout - DINOv2 uses different interface
        if hasattr(module, 'attn_drop') and hasattr(module.attn_drop, 'p'):
            dropout_p = module.attn_drop.p if self.training else 0.0
        elif hasattr(module, 'attn_drop'):
            dropout_p = module.attn_drop if self.training else 0.0
        else:
            dropout_p = 0.0

        # Use scaled_dot_product_attention if available, otherwise manual attention
        if hasattr(F, 'scaled_dot_product_attention'):
            # For masked attention, we need to handle the mask format
            attn_mask = None
            if mask is not None:
                # Convert boolean mask to additive mask for scaled_dot_product_attention
                attn_mask = torch.where(mask, 0.0, float('-inf'))
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
        else:
            # Fallback to manual attention computation
            attn = (q @ k.transpose(-2, -1)) * module.scale
            if mask is not None:
                attn = attn.masked_fill(~mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            if hasattr(module, 'attn_drop') and callable(module.attn_drop):
                attn = module.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = module.proj(x)
        if hasattr(module, 'proj_drop') and callable(module.proj_drop):
            x = module.proj_drop(x)

        return x

    def forward(self, x: torch.Tensor):
        x = (x - self.encoder.pixel_mean) / self.encoder.pixel_std

        # Use our DINOv2 encoder's patch embedding
        x = self.encoder.backbone.patch_embed(x)
        x = self.encoder.backbone._pos_embed(x)
        x = self.encoder.backbone.patch_drop(x)
        x = self.encoder.backbone.norm_pre(x)

        attn_mask = None
        mask_logits_per_layer, class_logits_per_layer = [], []
        
        for i, block in enumerate(self.encoder.backbone.blocks):
            if i == len(self.encoder.backbone.blocks) - self.num_blocks:
                x = torch.cat(
                    (self.q.weight[None, :, :].expand(x.shape[0], -1, -1), x), dim=1
                )

            if (
                self.masked_attn_enabled
                and i >= len(self.encoder.backbone.blocks) - self.num_blocks
            ):
                mask_logits, class_logits = self._predict(self.encoder.backbone.norm(x))
                mask_logits_per_layer.append(mask_logits)
                class_logits_per_layer.append(class_logits)

                attn_mask = torch.ones(
                    x.shape[0],
                    x.shape[1],
                    x.shape[1],
                    dtype=torch.bool,
                    device=x.device,
                )
                interpolated = F.interpolate(
                    mask_logits,
                    self.encoder.backbone.patch_embed.grid_size,
                    mode="bilinear",
                )
                interpolated = interpolated.view(
                    interpolated.size(0), interpolated.size(1), -1
                )
                attn_mask[
                    :,
                    : self.num_q,
                    self.num_q + self.encoder.backbone.num_prefix_tokens :,
                ] = (
                    interpolated > 0
                )
                attn_mask = self._disable_attn_mask(
                    attn_mask,
                    self.attn_mask_probs[
                        i - len(self.encoder.backbone.blocks) + self.num_blocks
                    ],
                )

            # Handle DINOv2 attention differently since it may not have the same interface
            if hasattr(block, 'attn') and hasattr(block.attn, 'qkv'):
                # Standard ViT-like attention
                x = x + block.drop_path1(
                    block.ls1(self._attn(block.attn, block.norm1(x), attn_mask))
                )
                x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
            else:
                # Fallback to regular block forward (DINOv2 blocks may not support masking)
                x = block(x)

        mask_logits, class_logits = self._predict(self.encoder.backbone.norm(x))
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)

        return (
            mask_logits_per_layer,
            class_logits_per_layer,
        )