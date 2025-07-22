#!/usr/bin/env python3
"""
Test script to verify EoMT implementation with standalone DINOv2 encoder.
"""

import torch
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from peaknet_eomt.modeling import EoMT, vit_large_config


def test_eomt():
    """Test basic EoMT functionality with new configuration interface."""
    print("Testing EoMT with configuration-based interface...")
    
    # Test parameters
    img_size = (518, 518)
    batch_size = 2
    num_classes = 150
    num_queries = 100
    
    try:
        # Create model using direct configuration (without pretrained weights for faster testing)
        print("Creating EoMT model with direct configuration...")
        model = EoMT(
            # ViT-Large configuration with register tokens
            patch_size=14,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4.0,
            num_register_tokens=4,
            
            # EoMT parameters
            img_size=img_size,
            num_classes=num_classes,
            num_q=num_queries,
            pretrained=False  # Skip pretrained for faster testing
        )
        print(f"✓ Model created successfully")
        print(f"  - Encoder embed dim: {model.encoder.backbone.embed_dim}")
        print(f"  - Patch size: {model.encoder.backbone.patch_embed.patch_size}")
        print(f"  - Num prefix tokens: {model.encoder.backbone.num_prefix_tokens}")
        
        # Test forward pass
        print("\nTesting forward pass...")
        x = torch.randn(batch_size, 3, img_size[0], img_size[1])
        
        with torch.no_grad():
            mask_logits_per_layer, class_logits_per_layer = model(x)
            
        print(f"✓ Forward pass successful")
        print(f"  - Output layers: {len(mask_logits_per_layer)}")
        print(f"  - Mask logits shape: {mask_logits_per_layer[-1].shape}")
        print(f"  - Class logits shape: {class_logits_per_layer[-1].shape}")
        
        # Verify output shapes - account for upsampling in ScaleBlock
        # With patch size 14 and upsampling, the resolution will be higher than base patches
        base_patches_h = img_size[0] // 14  # 518 // 14 = 37
        base_patches_w = img_size[1] // 14  # 518 // 14 = 37
        # ScaleBlock does 2x upsampling, so final resolution should be 2x base patches
        expected_mask_shape = (batch_size, num_queries, base_patches_h * 2, base_patches_w * 2)
        expected_class_shape = (batch_size, num_queries, num_classes + 1)
        
        assert mask_logits_per_layer[-1].shape == expected_mask_shape, \
            f"Mask shape mismatch: {mask_logits_per_layer[-1].shape} vs {expected_mask_shape}"
        assert class_logits_per_layer[-1].shape == expected_class_shape, \
            f"Class shape mismatch: {class_logits_per_layer[-1].shape} vs {expected_class_shape}"
            
        print("✓ All output shapes are correct")
        
        print("\n✓ Direct configuration test passed!")
        
        # Test using configuration helper functions
        print("\nTesting with configuration helper functions...")
        config = vit_large_config(num_register_tokens=4)
        model2 = EoMT(
            **config,
            img_size=img_size,
            num_classes=num_classes,
            num_q=num_queries,
            pretrained=False
        )
        print("✓ Configuration helper test passed!")
        
        print("\n🎉 All tests passed! New configuration-based interface is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_compatibility():
    """Test backward compatibility with legacy interface."""
    print("\nTesting backward compatibility...")
    
    try:
        from peaknet_eomt.modeling import create_dinov2_encoder_legacy
        import warnings
        
        # Suppress the deprecation warning for this test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            encoder = create_dinov2_encoder_legacy(
                model_name="vit_large_patch14_reg4_dinov2",
                pretrained=False
            )
            print("✓ Legacy interface still works")
            
    except ImportError:
        print("⚠️  Legacy interface not available (expected)")
    except Exception as e:
        print(f"❌ Legacy compatibility test failed: {e}")
        return False
    
    return True


def test_configuration_options():
    """Test different configuration options."""
    print("\nTesting configuration flexibility...")
    
    try:
        from peaknet_eomt.modeling import vit_small_config, vit_base_config, vit_giant_config
        
        # Test ViT-Small
        small_config = vit_small_config()
        model_small = EoMT(**small_config, pretrained=False, num_classes=10, num_q=50)
        print(f"✓ ViT-Small: embed_dim={small_config['embed_dim']}, depth={small_config['depth']}")
        
        # Test ViT-Giant with SwiGLU
        giant_config = vit_giant_config()
        model_giant = EoMT(**giant_config, pretrained=False, num_classes=10, num_q=50)
        print(f"✓ ViT-Giant: embed_dim={giant_config['embed_dim']}, ffn_layer={giant_config['ffn_layer']}")
        
        # Test custom configuration
        custom_model = EoMT(
            patch_size=16,  # Different patch size
            embed_dim=512,  # Custom dimension
            depth=12,
            num_heads=8,
            num_register_tokens=0,  # No register tokens
            pretrained=False,
            num_classes=10,
            num_q=25
        )
        print("✓ Custom configuration works")
        
    except Exception as e:
        print(f"❌ Configuration options test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True


if __name__ == "__main__":
    success1 = test_eomt()
    success2 = test_backward_compatibility()
    success3 = test_configuration_options()
    
    overall_success = success1 and success2 and success3
    
    if overall_success:
        print("\n🎉 All tests passed! Configuration-based interface is ready for use.")
    else:
        print("\n❌ Some tests failed.")
    
    sys.exit(0 if overall_success else 1)