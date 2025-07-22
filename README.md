# PeakNet EoMT - Production-Ready Implementation

A standalone implementation of EoMT (Encoder of Multi-Task) using DINOv2 without TIMM dependencies, designed for production deployment.

## 🚀 Key Features

- **No TIMM Dependency**: Uses standalone DINOv2 implementation copied from Meta's official codebase
- **Production Ready**: Clean, minimal dependencies for reliable deployment
- **Full EoMT Compatibility**: Drop-in replacement for original TIMM-based implementation
- **DINOv2 Features**: Position embedding interpolation, register tokens, memory-efficient attention
- **Multiple Model Sizes**: Supports ViT-S/B/L/G variants with and without register tokens

## 📁 Project Structure

```
peaknet_eomt/
├── modeling/
│   ├── __init__.py              # Main imports
│   ├── eomt.py                  # EoMT model implementation
│   ├── encoder_wrapper.py       # EoMT-compatible DINOv2 wrapper
│   ├── scale_block.py          # Upsampling blocks (no TIMM dependency)
│   └── dinov2/                 # Standalone DINOv2 implementation
│       ├── vision_transformer.py
│       ├── layers/              # All DINOv2 layer implementations
│       └── hub/                 # Model factory and loading utilities
└── test_eomt.py                # Test script
```

## 🔧 Usage

### Basic Usage (New Configuration-Based Interface)

```python
from peaknet_eomt.modeling import EoMT

# Create EoMT model with direct ViT configuration
model = EoMT(
    # Core ViT architecture parameters
    patch_size=14,
    embed_dim=1024,        # ViT-Large
    depth=24,              # ViT-Large
    num_heads=16,          # ViT-Large
    mlp_ratio=4.0,
    
    # DINOv2 specific parameters
    num_register_tokens=4, # Register tokens for training stability
    ffn_layer="mlp",       # or "swiglufused" for ViT-Giant
    
    # EoMT parameters
    img_size=(518, 518),
    num_classes=150,
    num_q=100,
    
    # Pretrained weights (auto-detected from config)
    pretrained=True
)

# Forward pass
import torch
x = torch.randn(2, 3, 518, 518)
mask_logits_per_layer, class_logits_per_layer = model(x)
```

### Using Configuration Helpers

```python
from peaknet_eomt.modeling import EoMT, vit_large_config, vit_giant_config

# Use predefined configurations
config = vit_large_config(num_register_tokens=4)
model = EoMT(**config, img_size=(518, 518), num_classes=150, num_q=100)

# ViT-Giant with SwiGLU FFN
giant_config = vit_giant_config(num_register_tokens=4)
model_giant = EoMT(**giant_config, img_size=(518, 518), num_classes=150, num_q=100)
```

### Available Standard Configurations

```python
from peaknet_eomt.modeling import vit_small_config, vit_base_config, vit_large_config, vit_giant_config

# ViT-Small: embed_dim=384, depth=12, num_heads=6
small_config = vit_small_config(num_register_tokens=4)

# ViT-Base: embed_dim=768, depth=12, num_heads=12  
base_config = vit_base_config(num_register_tokens=4)

# ViT-Large: embed_dim=1024, depth=24, num_heads=16
large_config = vit_large_config(num_register_tokens=4)

# ViT-Giant: embed_dim=1536, depth=40, num_heads=24, ffn_layer="swiglufused"
giant_config = vit_giant_config(num_register_tokens=4)
```

### Custom Configurations

```python
# Completely custom ViT configuration
custom_model = EoMT(
    patch_size=16,           # Different patch size
    embed_dim=512,           # Custom embedding dimension  
    depth=12,                # Custom depth
    num_heads=8,             # Custom head count
    num_register_tokens=0,   # No register tokens
    drop_path_rate=0.1,      # Stochastic depth
    pretrained=False,        # Random initialization
    img_size=(224, 224),
    num_classes=1000,
    num_q=50
)
```

### Using Standalone DINOv2 Encoder

```python
from peaknet_eomt.modeling import create_dinov2_encoder

# Create encoder with configuration
encoder = create_dinov2_encoder(
    embed_dim=1024,
    depth=24,
    num_heads=16,
    num_register_tokens=4,
    pretrained=True
)
```

### Legacy Interface (Deprecated)

```python
# Old string-based interface (deprecated but still supported)
from peaknet_eomt.modeling import create_dinov2_encoder_legacy
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    encoder = create_dinov2_encoder_legacy(
        model_name="vit_large_patch14_reg4_dinov2",
        pretrained=True
    )
```

## 🧪 Testing

Run the test script to verify the implementation:

```bash
python test_eomt.py
```

Expected output:
```
Testing EoMT with standalone DINOv2...
✓ Model created successfully
✓ Forward pass successful  
✓ All output shapes are correct
🎉 All tests passed!
```

## 🔗 Model Loading

The implementation supports loading official DINOv2 pretrained weights:

- Models are downloaded from Meta's official hub
- Weights are automatically loaded when `pretrained=True`
- No TIMM registry or complex model loading required

## 🏗️ Architecture Details

### EoMT Components

1. **DINOv2 Encoder**: Standalone implementation with all DINOv2 features
2. **Learnable Queries**: `nn.Embedding` for object detection queries
3. **Class Head**: Linear layer for classification per query
4. **Mask Head**: 3-layer MLP for mask embedding generation
5. **Upsampling**: ScaleBlock layers for spatial resolution recovery

### Key Differences from Original

- **No TIMM dependency**: Uses standalone DINOv2 implementation
- **Enhanced compatibility**: Handles different DINOv2 attention interfaces
- **Production focus**: Minimal dependencies, clean code structure
- **Better error handling**: Graceful fallbacks for different model configurations

## ⚡ Performance

- **Memory Efficient**: Uses optimized attention when xFormers is available
- **Scalable**: Supports different image sizes via position embedding interpolation  
- **Fast**: Direct model loading without TIMM overhead
- **Flexible**: Easy to customize model parameters

## 📋 Requirements

- PyTorch >= 1.9
- Optional: xFormers (for memory-efficient attention)
- No TIMM dependency!

## 🧬 Configuration-Based Architecture

### Why Configuration Over String Names?

The new interface uses **explicit ViT configuration parameters** instead of magic string names:

```python
# ❌ Old way: Magic strings
model = EoMT(encoder_name="vit_large_patch14_reg4_dinov2")

# ✅ New way: Explicit configuration  
model = EoMT(
    patch_size=14,
    embed_dim=1024, 
    depth=24,
    num_heads=16,
    num_register_tokens=4
)
```

### Benefits for Production

1. **🔍 Self-Documenting**: Parameters are immediately clear
2. **⚙️ Hydra-Friendly**: Perfect for configuration management frameworks
3. **🎛️ Fine-Grained Control**: Adjust any parameter independently
4. **🚀 Custom Architectures**: Easy to create non-standard configurations
5. **🔒 Type Safety**: Proper type hints and validation

### Hydra Integration Example

```yaml
# config.yaml
model:
  _target_: peaknet_eomt.modeling.EoMT
  
  # ViT architecture
  patch_size: 14
  embed_dim: 1024
  depth: 24
  num_heads: 16
  mlp_ratio: 4.0
  
  # DINOv2 features
  num_register_tokens: 4
  ffn_layer: mlp
  
  # EoMT parameters
  img_size: [518, 518]
  num_classes: 150
  num_q: 100
  
  # Training
  pretrained: true
  drop_path_rate: 0.1
```

```python
# main.py
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    model = hydra.utils.instantiate(cfg.model)
    # model is ready to use!
```

## 🎯 Production Benefits

1. **🔧 Configuration Management**: Perfect integration with Hydra/OmegaConf
2. **📝 Explicit Parameters**: No magic strings, all parameters are clear
3. **🔄 Backward Compatibility**: Legacy interface still supported (deprecated)
4. **🏗️ Custom Architectures**: Easy to experiment with different configurations
5. **🚫 No TIMM Dependencies**: Complete control over model architecture
6. **📦 Dependency Control**: No external model registries or complex dependencies
7. **🔒 Version Stability**: Pinned implementation that won't break with library updates  
8. **🚀 Deployment Ready**: Clean, minimal codebase for reliable deployment
9. **⚡ Full Feature Parity**: All DINOv2 and EoMT features preserved
10. **🔧 Easy Customization**: Direct access to model internals for modifications

This implementation provides a robust, production-ready version of EoMT that eliminates TIMM dependencies while providing a modern, configuration-driven interface perfect for research and production use.