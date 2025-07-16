#!/usr/bin/env python3
"""
Weight Conversion Utility for ViT PyTorch to C++ HLS

This script converts PyTorch ViT weights to C++ HLS format.
It can be used to transfer trained weights from PyTorch to the C++ implementation.
"""

import torch
import numpy as np
import argparse
import os
from pathlib import Path

def convert_tensor_to_cpp(tensor, name, dtype='data_t'):
    """Convert a PyTorch tensor to C++ array initialization code."""
    if tensor.dim() == 1:
        # 1D array
        size = tensor.shape[0]
        values = tensor.detach().cpu().numpy().flatten()
        code = f"{dtype} {name}[{size}] = {{\n"
        for i, val in enumerate(values):
            if i % 8 == 0:
                code += "    "
            code += f"{val:.6f}"
            if i < len(values) - 1:
                code += ", "
            if i % 8 == 7 or i == len(values) - 1:
                code += "\n"
        code += "};\n"
        return code
    
    elif tensor.dim() == 2:
        # 2D array
        rows, cols = tensor.shape
        values = tensor.detach().cpu().numpy()
        code = f"{dtype} {name}[{rows}][{cols}] = {{\n"
        for i in range(rows):
            code += "    {"
            for j in range(cols):
                code += f"{values[i, j]:.6f}"
                if j < cols - 1:
                    code += ", "
            code += "}"
            if i < rows - 1:
                code += ","
            code += "\n"
        code += "};\n"
        return code
    
    else:
        # Higher dimensional arrays - flatten to 1D for simplicity
        values = tensor.detach().cpu().numpy().flatten()
        size = len(values)
        code = f"{dtype} {name}[{size}] = {{\n"
        for i, val in enumerate(values):
            if i % 8 == 0:
                code += "    "
            code += f"{val:.6f}"
            if i < len(values) - 1:
                code += ", "
            if i % 8 == 7 or i == len(values) - 1:
                code += "\n"
        code += "};\n"
        return code

def extract_vit_weights(model):
    """Extract weights from a PyTorch ViT model."""
    weights = {}
    
    # Check if it's a ViT model
    if hasattr(model, 'patch_embed'):
        # timm or similar implementation
        weights['patch_embed_weight'] = model.patch_embed.proj.weight.view(model.patch_embed.proj.weight.shape[0], -1)
        weights['patch_embed_bias'] = model.patch_embed.proj.bias
        weights['pos_embed'] = model.pos_embed.squeeze(0)
        weights['cls_token'] = model.cls_token.squeeze(0).squeeze(0)
        
        # Transformer layers
        for i, layer in enumerate(model.blocks):
            layer_prefix = f"layer_{i}"
            
            # Attention weights
            weights[f'{layer_prefix}_attn_qkv_weight'] = layer.attn.qkv.weight
            weights[f'{layer_prefix}_attn_qkv_bias'] = layer.attn.qkv.bias
            weights[f'{layer_prefix}_attn_proj_weight'] = layer.attn.proj.weight
            weights[f'{layer_prefix}_attn_proj_bias'] = layer.attn.proj.bias
            
            # MLP weights
            weights[f'{layer_prefix}_mlp_fc1_weight'] = layer.mlp.fc1.weight
            weights[f'{layer_prefix}_mlp_fc1_bias'] = layer.mlp.fc1.bias
            weights[f'{layer_prefix}_mlp_fc2_weight'] = layer.mlp.fc2.weight
            weights[f'{layer_prefix}_mlp_fc2_bias'] = layer.mlp.fc2.bias
            
            # Layer norm weights
            weights[f'{layer_prefix}_norm1_weight'] = layer.norm1.weight
            weights[f'{layer_prefix}_norm1_bias'] = layer.norm1.bias
            weights[f'{layer_prefix}_norm2_weight'] = layer.norm2.weight
            weights[f'{layer_prefix}_norm2_bias'] = layer.norm2.bias
        
        # Final norm and classifier
        weights['norm_weight'] = model.norm.weight
        weights['norm_bias'] = model.norm.bias
        weights['head_weight'] = model.head.weight
        weights['head_bias'] = model.head.bias
    
    elif hasattr(model, 'to_patch_embedding'):
        # vit-pytorch implementation
        weights['patch_embed_weight'] = model.to_patch_embedding[2].weight
        weights['patch_embed_bias'] = model.to_patch_embedding[2].bias
        weights['pos_embed'] = model.pos_embedding.squeeze(0)
        weights['cls_token'] = model.cls_token.squeeze(0).squeeze(0)
        
        # Transformer layers
        for i, (attn, ff) in enumerate(model.transformer.layers):
            layer_prefix = f"layer_{i}"
            
            # Attention weights (combined QKV)
            weights[f'{layer_prefix}_attn_qkv_weight'] = attn.to_qkv.weight
            weights[f'{layer_prefix}_attn_proj_weight'] = attn.to_out[0].weight
            weights[f'{layer_prefix}_attn_proj_bias'] = attn.to_out[0].bias
            
            # MLP weights
            weights[f'{layer_prefix}_mlp_fc1_weight'] = ff.net[1].weight
            weights[f'{layer_prefix}_mlp_fc1_bias'] = ff.net[1].bias
            weights[f'{layer_prefix}_mlp_fc2_weight'] = ff.net[4].weight
            weights[f'{layer_prefix}_mlp_fc2_bias'] = ff.net[4].bias
            
            # Layer norm weights
            weights[f'{layer_prefix}_norm1_weight'] = attn.norm.weight
            weights[f'{layer_prefix}_norm1_bias'] = attn.norm.bias
            weights[f'{layer_prefix}_norm2_weight'] = ff.net[0].weight
            weights[f'{layer_prefix}_norm2_bias'] = ff.net[0].bias
        
        # Final norm and classifier
        weights['norm_weight'] = model.transformer.norm.weight
        weights['norm_bias'] = model.transformer.norm.bias
        weights['head_weight'] = model.mlp_head.weight
        weights['head_bias'] = model.mlp_head.bias
    
    return weights

def generate_cpp_weights(weights, output_dir):
    """Generate C++ header files with weight definitions."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate header file
    header_path = os.path.join(output_dir, 'vit_weights.h')
    with open(header_path, 'w') as f:
        f.write('#ifndef VIT_WEIGHTS_H\n')
        f.write('#define VIT_WEIGHTS_H\n\n')
        f.write('#include "vit_types.h"\n\n')
        f.write('// Auto-generated weight definitions\n')
        f.write('// Generated from PyTorch ViT model\n\n')
        
        # Function declarations
        f.write('void load_vit_weights();\n')
        f.write('void init_patch_embedding_weights();\n')
        f.write('void init_transformer_weights();\n')
        f.write('void init_classification_weights();\n\n')
        
        f.write('#endif // VIT_WEIGHTS_H\n')
    
    # Generate implementation file
    impl_path = os.path.join(output_dir, 'vit_weights.cpp')
    with open(impl_path, 'w') as f:
        f.write('#include "vit_weights.h"\n\n')
        f.write('// Auto-generated weight implementations\n\n')
        
        # Patch embedding weights
        f.write('void init_patch_embedding_weights() {\n')
        f.write('    // TODO: Initialize patch embedding weights\n')
        f.write('}\n\n')
        
        # Transformer weights
        f.write('void init_transformer_weights() {\n')
        f.write('    // TODO: Initialize transformer weights\n')
        f.write('}\n\n')
        
        # Classification weights
        f.write('void init_classification_weights() {\n')
        f.write('    // TODO: Initialize classification weights\n')
        f.write('}\n\n')
        
        # Main loader function
        f.write('void load_vit_weights() {\n')
        f.write('    init_patch_embedding_weights();\n')
        f.write('    init_transformer_weights();\n')
        f.write('    init_classification_weights();\n')
        f.write('}\n')
    
    # Generate individual weight files
    for name, tensor in weights.items():
        weight_path = os.path.join(output_dir, f'{name}.hpp')
        with open(weight_path, 'w') as f:
            f.write(f'// Weight: {name}\n')
            f.write(f'// Shape: {list(tensor.shape)}\n\n')
            
            # Determine appropriate data type
            if 'weight' in name:
                dtype = 'weight_t'
            else:
                dtype = 'data_t'
            
            # Generate C++ array
            f.write(convert_tensor_to_cpp(tensor, name, dtype))
    
    print(f"Generated {len(weights)} weight files in {output_dir}")
    return header_path, impl_path

def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch ViT weights to C++ HLS format')
    parser.add_argument('--model-path', type=str, required=True, help='Path to PyTorch model checkpoint')
    parser.add_argument('--output-dir', type=str, default='weights', help='Output directory for C++ weight files')
    parser.add_argument('--model-type', type=str, choices=['vit-pytorch', 'timm'], default='vit-pytorch', help='Model implementation type')
    
    args = parser.parse_args()
    
    # Load PyTorch model
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location='cpu')
    
    # Extract state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Create a dummy model to load weights
    if args.model_type == 'vit-pytorch':
        from vit_pytorch import ViT
        model = ViT(
            image_size=224,
            patch_size=16,
            num_classes=1000,
            dim=768,
            depth=12,
            heads=12,
            mlp_dim=3072
        )
    else:
        # For timm models
        import timm
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=1000)
    
    # Load weights
    model.load_state_dict(state_dict, strict=False)
    
    # Extract weights
    print("Extracting weights...")
    weights = extract_vit_weights(model)
    
    # Generate C++ files
    print("Generating C++ weight files...")
    header_path, impl_path = generate_cpp_weights(weights, args.output_dir)
    
    print(f"Conversion complete!")
    print(f"Header file: {header_path}")
    print(f"Implementation file: {impl_path}")
    print(f"Weight files: {args.output_dir}/*.hpp")
    
    # Print usage instructions
    print("\nUsage instructions:")
    print("1. Include the generated header file in your HLS project")
    print("2. Call load_vit_weights() to initialize the weights")
    print("3. Modify the weight loading functions as needed for your specific use case")

if __name__ == '__main__':
    main()