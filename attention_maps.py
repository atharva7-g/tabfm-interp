import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from tabpfn import TabPFNRegressor, TabPFNClassifier
from einops import rearrange, repeat


def extract_attention_weights_from_tabpfn(model, input_data: np.ndarray, device: str = 'cuda') -> Dict[str, torch.Tensor]:
    """
    Extract attention weights from TabPFN model by hooking into the MultiHeadAttention modules.
    
    Args:
        model: TabPFN model instance
        input_data: Input data array (numpy array)
        device: Device to run on
        
    Returns:
        Dictionary containing attention weights for each layer
    """
    attention_weights = {}
    
    def attention_hook(name):
        def hook(module, input, output):
            # Check if this is a MultiHeadAttention module
            if hasattr(module, 'compute_qkv') and hasattr(module, '_compute'):
                attn_module = module
                
                # Get input hidden states
                hidden_states = input[0] if isinstance(input, tuple) else input
                
                # Compute Q, K, V using the MultiHeadAttention's compute_qkv method
                try:
                    x_kv = None
                    if isinstance(input, tuple) and len(input) > 1:
                        x_kv = input[1]
                    
                    # Flatten inputs using same logic as TabPFN's _rearrange_inputs_to_flat_batch
                    # This reshapes tensors to flatten extra batch dimensions: reshape(-1, *shape[-2:])
                    hidden_states_flat = hidden_states.reshape(-1, *hidden_states.shape[-2:])
                    x_kv_flat = None
                    if x_kv is not None:
                        x_kv_flat = x_kv.reshape(-1, *x_kv.shape[-2:])
                    
                    # No call counters in production
                    q, k, v, kv, qkv = attn_module.compute_qkv(
                        hidden_states_flat,
                        x_kv_flat,  # x_kv for cross-attention, None for self-attention
                        attn_module._k_cache,  # k_cache
                        attn_module._v_cache,  # v_cache
                        attn_module._kv_cache,  # kv_cache
                        cache_kv=False,
                        use_cached_kv=False,
                        reuse_first_head_kv=False
                    )
                    # Handle different QKV formats (same as TabPFN's compute_attention_heads)
                    if qkv is not None:
                        # Combined QKV format: shape [batch, seqlen, 3, nhead, d_k]
                        q, k, v = qkv.unbind(dim=-3)
                    elif kv is not None:
                        # Combined KV format: shape [batch, seqlen, 2, nhead_kv, d_k]
                        k, v = kv.unbind(dim=-3)
                    
                    # Ensure we have Q, K, V tensors
                    if q is not None and k is not None and v is not None:
                        # Q, K, V may have variable shapes depending on input
                        # TabPFN's _rearrange_inputs_to_flat_batch flattens extra batch dims
                        # but we're calling compute_qkv directly, so we need to handle shapes carefully
                        
                        # Flatten any extra batch dimensions to standardize shape
                        # Expected: [batch, seqlen, nhead, d_k] (4 dims)
                        # If qkv was used, unbind might preserve extra dims
                        if q.dim() == 5:
                            # Shape might be [batch, seqlen, 1, nhead, d_k] after unbind
                            # Reshape to [batch, seqlen, nhead, d_k]
                            q = q.squeeze(dim=2) if q.shape[2] == 1 else q.reshape(q.shape[0], q.shape[1], -1, q.shape[-1])
                            k = k.squeeze(dim=2) if k.shape[2] == 1 else k.reshape(k.shape[0], k.shape[1], -1, k.shape[-1])
                            v = v.squeeze(dim=2) if v.shape[2] == 1 else v.reshape(v.shape[0], v.shape[1], -1, v.shape[-1])
                        elif q.dim() > 4:
                            # Flatten all leading dimensions except last 3
                            q = q.reshape(-1, *q.shape[-3:])
                            k = k.reshape(-1, *k.shape[-3:])
                            v = v.reshape(-1, *v.shape[-3:])
                        elif q.dim() < 4:
                            # Unexpected - skip this layer
                            raise ValueError(f"Unexpected q dimension: {q.dim()}, shape: {q.shape}")
                        
                        # After normalization, q should have shape [batch, seqlen, nhead, d_k]
                        if q.dim() != 4:
                            raise ValueError(f"After processing, q has {q.dim()} dims, expected 4. Shape: {q.shape}")
                        
                        # Safely unpack with better error handling
                        try:
                            batch_size, seq_len_q, nhead, d_k = q.shape
                        except ValueError as e:
                            raise ValueError(f"Cannot unpack q.shape={q.shape} into 4 values: {e}")
                        
                        try:
                            _, seq_len_kv, nhead_kv, d_v = v.shape
                        except ValueError as e:
                            raise ValueError(f"Cannot unpack v.shape={v.shape} into 4 values: {e}")
                        share_kv_across_n_heads = nhead // nhead_kv
                        
                        # Broadcast K and V across heads if using grouped/shared KV (same as TabPFN)
                        # This matches TabPFN's broadcast_kv_across_heads method (full_attention.py lines 545-559)
                        if share_kv_across_n_heads > 1:
                            # Get the class method from the module's class
                            broadcast_fn = type(attn_module).broadcast_kv_across_heads
                            k = broadcast_fn(k, share_kv_across_n_heads)
                            v = broadcast_fn(v, share_kv_across_n_heads)
                        
                        # Compute attention logits using same einsum pattern as TabPFN
                        # TabPFN uses: "b q h d, b k h d -> b q k h"
                        # This matches the actual implementation in full_attention.py line 707
                        attention_scores = torch.einsum("b q h d, b k h d -> b q k h", q, k)
                        
                        # Scaling (same as TabPFN)
                        scale = attn_module.softmax_scale
                        if scale is None:
                            scale = torch.sqrt(torch.tensor(1.0 / d_k, dtype=q.dtype, device=q.device))
                        attention_scores = attention_scores * scale
                        
                        # Softmax over key dimension (dim=2 is the key sequence dimension in [b, q, k, h])
                        # This matches TabPFN's implementation in full_attention.py line 713
                        attention_probs = torch.nn.functional.softmax(attention_scores, dim=2)
                        
                        # Apply dropout if in training mode (TabPFN applies dropout, but for inference/visualization
                        # dropout_p is typically 0.0, so this won't affect results)
                        dropout_p = attn_module.dropout_p if attn_module.dropout_p is not None else 0.0
                        if dropout_p > 0.0:
                            attention_probs = torch.dropout(attention_probs, dropout_p, train=False)
                        
                        # Store attention weights: shape [batch, seq_q, seq_k, nhead]
                        # Reshape to [batch, nhead, seq_q, seq_k] for easier visualization
                        attention_probs = attention_probs.permute(0, 3, 1, 2)  # [b, q, k, h] -> [b, h, q, k]
                        attention_weights[name] = attention_probs.detach()  
                except Exception as e:
                    print(f"Warning: Could not extract attention from {name}: {e}")
                    
        return hook
    
    # Register hooks for transformer layers (avoid duplicate module registrations)
    hook_handles = []
    for i, layer in enumerate(model.model_.transformer_encoder.layers):
        # Hook the MultiHeadAttention modules directly
        if hasattr(layer, 'self_attn_between_features') and hasattr(layer, 'self_attn_between_items'):
            handle1 = layer.self_attn_between_features.register_forward_hook(attention_hook(f'layer_{i}_features'))
            handle2 = layer.self_attn_between_items.register_forward_hook(attention_hook(f'layer_{i}_items'))
            hook_handles.append(handle1)
            hook_handles.append(handle2)
    
    # Forward pass to extract attention weights
    with torch.no_grad():
        if isinstance(model, TabPFNRegressor):
            _ = model.predict(input_data)
        else:
            _ = model.predict_proba(input_data)
    
    # Remove hooks
    for handle in hook_handles:
        handle.remove()
    
    # Keep attention weights with all heads (no averaging):
    # Shape is [batch, nhead, seq_q, seq_k] for all attention types
    # We'll extract individual heads during visualization
    return attention_weights


def visualize_attention_heads(model, input_data: np.ndarray, output_dir: str = '.', 
                            filename: str = 'attentions.png', device: str = 'cuda',
                            sample_idx: int = 0, max_layers: Optional[int] = None) -> None:
    """
    Visualizes the attention maps of all attention heads in TabPFN during a forward pass.
    For items attention: creates separate plots for each head (shows all batch items).
    For features attention: creates separate plots for each head (averages over batch dimension).
    
    Args:
        model: TabPFN model instance (TabPFNRegressor or TabPFNClassifier)
        input_data: Input data array (numpy array)
        output_dir: Directory where the attention image will be saved
        filename: Name of the output image file
        device: Device to run on ('cuda' or 'cpu')
        sample_idx: Index of the sample to visualize attention for
        max_layers: Maximum number of layers to visualize (None = visualize all layers)
    """
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Select the specific sample to visualize if sample_idx is provided and valid
    # Input data shape is typically [num_samples, num_features]
    if len(input_data.shape) == 2 and input_data.shape[0] > sample_idx:
        input_data_for_extraction = input_data[sample_idx:sample_idx+1]  # Select single sample: [1, num_features]
        print(f"Visualizing attention for sample {sample_idx} (selecting from {input_data.shape[0]} samples)")
    else:
        input_data_for_extraction = input_data
        if len(input_data.shape) == 2:
            print(f"Visualizing attention for all {input_data.shape[0]} samples (sample_idx {sample_idx} >= num_samples)")
        else:
            print(f"Visualizing attention with input shape {input_data.shape}")
    
    print(f"Input data shape for extraction: {input_data_for_extraction.shape}")
    
    # Extract attention weights using the selected data
    print("Extracting attention weights...")
    attention_weights = extract_attention_weights_from_tabpfn(model, input_data_for_extraction, device)
    
    if not attention_weights:
        print("Warning: No attention weights extracted. The model might not have accessible attention mechanisms.")
        return
    
    # Separate items and features attention
    items_attention = {k: v for k, v in attention_weights.items() if '_items' in k}
    features_attention = {k: v for k, v in attention_weights.items() if '_features' in k}
    
    # Get layer names for items and features separately, filtering to only layers 1 and 2
    # Use exact matching to avoid matching layer_10, layer_11, layer_20, layer_21, etc.
    def is_layer_1_or_2(name):
        return name.startswith('layer_1_') or name.startswith('layer_2_')
    
    items_layer_names = sorted([k for k in items_attention.keys() if is_layer_1_or_2(k)])
    features_layer_names = sorted([k for k in features_attention.keys() if is_layer_1_or_2(k)])
    
    # Count the filtered layers
    num_items_layers = len(items_layer_names)
    num_features_layers = len(features_layer_names)
    
    if num_items_layers == 0 and num_features_layers == 0:
        print("No attention weights found!")
        return
    
    # Visualize items attention (averaged over heads only, shows all batch items)
    if num_items_layers > 0:
        # Create filename for items visualization
        base_filename = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1]
        items_filename = f"{base_filename}_items{ext}"
        # Use only the first num_items_layers layer names
        items_layer_names_to_use = items_layer_names[:num_items_layers]
        visualize_items_attention(
            items_attention, items_layer_names_to_use, num_items_layers,
            sample_idx, output_dir, items_filename
        )
    
    # Visualize features attention (average over heads only)
    if num_features_layers > 0:
        # Create filename for features visualization
        base_filename = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1]
        features_filename = f"{base_filename}_features{ext}"
        # Use only the first num_features_layers layer names
        features_layer_names_to_use = features_layer_names[:num_features_layers]
        visualize_features_attention(
            features_attention, features_layer_names_to_use, num_features_layers,
            sample_idx, output_dir, features_filename
        )


def visualize_items_attention(attention_weights: Dict[str, torch.Tensor], 
                              layer_names: List[str], num_layers: int,
                              sample_idx: int, output_dir: str, filename: str) -> None:
    """
    Visualize items attention maps for each head separately.
    Creates separate plots for each attention head and saves them as individual PNGs.
    Shows all batch samples (one graph per batch item per layer per head).
    
    Args:
        attention_weights: Dictionary of attention weights for items layers
                          Shape: [batch, nhead, seq_q, seq_k] for each layer
        layer_names: List of layer names
        num_layers: Number of layers to visualize
        sample_idx: Index of the sample to visualize (ignored, shows all batch items)
        output_dir: Directory to save the image
        filename: Name pattern for the output files (will be modified with head number)
    """
    print(f"Visualizing items attention for each head separately...")
    
    # Get attention shape from first layer: [batch, nhead, seq_q, seq_k]
    first_layer_attn = attention_weights[layer_names[0]]
    
    # Ensure it's a torch tensor for proper indexing
    if isinstance(first_layer_attn, np.ndarray):
        first_layer_attn = torch.from_numpy(first_layer_attn)
    
    batch_size = first_layer_attn.shape[0]
    num_heads = first_layer_attn.shape[1]
    print(f"Showing all {batch_size} batch samples and {num_heads} heads for items attention")
    
    # Extract base filename and extension
    base_filename = os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1]
    
    # Create a separate plot for each head
    for head_idx in range(num_heads):
        # Create figure with batch_size rows and num_layers columns for this head
        fig, axes = plt.subplots(batch_size, num_layers, figsize=(num_layers * 4, batch_size * 3))
        
        # Handle single batch or single layer cases
        if batch_size == 1 and num_layers == 1:
            axes = [[axes]]
        elif batch_size == 1:
            axes = [axes]
        elif num_layers == 1:
            axes = [[ax] for ax in axes]
        
        # Plot attention maps for each layer and each batch item for this head
        for layer_idx in range(num_layers):
            layer_name = layer_names[layer_idx]
            layer_attentions = attention_weights[layer_name]  # Shape: [batch, nhead, seq_q, seq_k]
            
            # Ensure it's a torch tensor
            if isinstance(layer_attentions, np.ndarray):
                layer_attentions = torch.from_numpy(layer_attentions)
            elif isinstance(layer_attentions, torch.Tensor):
                layer_attentions = layer_attentions.cpu()
            
            # Plot each batch item for this layer and head
            for batch_idx in range(batch_size):
                # Get the axis for this batch item and layer
                if batch_size == 1:
                    ax = axes[layer_idx]
                elif num_layers == 1:
                    ax = axes[batch_idx]
                else:
                    ax = axes[batch_idx][layer_idx]
                
                # Extract attention for this specific head and batch item
                # Shape: [batch, nhead, seq_q, seq_k] -> [seq_q, seq_k] for this head and batch
                if isinstance(layer_attentions, torch.Tensor):
                    attn_map = layer_attentions[batch_idx, head_idx].numpy()
                else:
                    attn_map = layer_attentions[batch_idx, head_idx]
                
                # Ensure attn_map is 2D
                if len(attn_map.shape) > 2:
                    print(f"Warning: Unexpected attention map shape {attn_map.shape} for layer {layer_name}, batch {batch_idx}, head {head_idx}, flattening extra dimensions")
                    while len(attn_map.shape) > 2:
                        attn_map = attn_map.squeeze()
                elif len(attn_map.shape) < 2:
                    print(f"Warning: Unexpected attention map shape {attn_map.shape} for layer {layer_name}, batch {batch_idx}, head {head_idx}")
                
                # Ensure it's a numpy array with proper shape
                attn_map = np.asarray(attn_map)
                
                # Get dimensions for tick labels
                seq_q, seq_k = attn_map.shape
                
                # Plot the attention map with extent to set proper axis ranges
                # extent = [left, right, bottom, top]
                # In image coordinates, y increases downward, so we invert for display
                im = ax.imshow(attn_map, cmap='Blues', aspect='auto', vmin=0, vmax=1, 
                              extent=[-0.5, seq_k-0.5, seq_q-0.5, -0.5], origin='upper')
                
                # Set integer ticks for x and y axes
                ax.set_xticks(range(seq_k))
                ax.set_yticks(range(seq_q))
                # Format ticks as integers to avoid fractional labels
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                ax.tick_params(axis='both', which='major', labelsize=8)
                
                # Set title and labels
                if batch_idx == 0:
                    ax.set_title(f'Layer {layer_idx + 1}, Head {head_idx + 1}', fontsize=12)
                if layer_idx == 0:
                    ax.set_ylabel(f'Item {batch_idx}', fontsize=10)
                
                # Only add colorbar to first subplot to save space
                if batch_idx == 0 and layer_idx == 0:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                # Add axis labels only to bottom row and left column
                if batch_idx == batch_size - 1:
                    ax.set_xlabel('Key Item', fontsize=10)
                if layer_idx == 0:
                    # ylabel already set above
                    pass
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure with head number in filename
        head_filename = f"{base_filename}_head_{head_idx + 1}{ext}"
        output_path = os.path.join(output_dir, head_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Items attention maps for head {head_idx + 1} saved to {output_path}")


def visualize_features_attention(attention_weights: Dict[str, torch.Tensor], 
                                 layer_names: List[str], num_layers: int,
                                 sample_idx: int, output_dir: str, filename: str) -> None:
    """
    Visualize features attention maps for each head separately.
    Creates separate plots for each attention head and saves them as individual PNGs.
    For features attention, averages over batch dimension but keeps heads separate.
    
    Args:
        attention_weights: Dictionary of attention weights for features layers
                          Shape: [batch, nhead, seq_q, seq_k] for each layer
        layer_names: List of layer names
        num_layers: Number of layers to visualize
        sample_idx: Index of the sample to visualize
        output_dir: Directory to save the image
        filename: Name pattern for the output files (will be modified with head number)
    """
    print(f"Visualizing features attention for each head separately...")
    
    # Get attention shape from first layer: [batch, nhead, seq_q, seq_k]
    first_layer_attn = attention_weights[layer_names[0]]
    
    # Ensure it's a torch tensor for proper indexing
    if isinstance(first_layer_attn, np.ndarray):
        first_layer_attn = torch.from_numpy(first_layer_attn)
    
    num_heads = first_layer_attn.shape[1]
    print(f"Showing {num_heads} heads for features attention (averaged over batch)")
    
    # Extract base filename and extension
    base_filename = os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1]
    
    # Create a separate plot for each head
    for head_idx in range(num_heads):
        # Create figure with one subplot per layer for this head
        fig, axes = plt.subplots(1, num_layers, figsize=(num_layers * 4, 4))
        
        # Handle single layer case
        if num_layers == 1:
            axes = [axes]
        
        # Plot attention maps for each layer for this head
        for layer_idx in range(num_layers):
            layer_name = layer_names[layer_idx]
            layer_attentions = attention_weights[layer_name]  # Shape: [batch, nhead, seq_q, seq_k]
            
            # Ensure it's a torch tensor
            if isinstance(layer_attentions, np.ndarray):
                layer_attentions = torch.from_numpy(layer_attentions)
            elif isinstance(layer_attentions, torch.Tensor):
                layer_attentions = layer_attentions.cpu()
            
            ax = axes[layer_idx]
            
            # Average over batch dimension (dim=0) but keep this specific head
            # Shape: [batch, nhead, seq_q, seq_k] -> [seq_q, seq_k] for this head
            if isinstance(layer_attentions, torch.Tensor):
                # Extract head and average over batch: [batch, seq_q, seq_k] -> [seq_q, seq_k]
                attn_map = layer_attentions[:, head_idx, :, :].mean(dim=0).numpy()
            else:
                attn_map = np.mean(layer_attentions[:, head_idx, :, :], axis=0)
            
            # Ensure attn_map is 2D
            if len(attn_map.shape) > 2:
                print(f"Warning: Unexpected attention map shape {attn_map.shape} for layer {layer_name}, head {head_idx}, flattening extra dimensions")
                while len(attn_map.shape) > 2:
                    attn_map = attn_map.squeeze()
            elif len(attn_map.shape) < 2:
                print(f"Warning: Unexpected attention map shape {attn_map.shape} for layer {layer_name}, head {head_idx}")
            
            # Ensure it's a numpy array with proper shape
            attn_map = np.asarray(attn_map)
            
            # Get dimensions for tick labels
            seq_q, seq_k = attn_map.shape
            
            # Plot the attention map with extent to set proper axis ranges
            # extent = [left, right, bottom, top]
            # In image coordinates, y increases downward, so we invert for display
            im = ax.imshow(attn_map, cmap='Blues', aspect='auto', vmin=0, vmax=1,
                          extent=[-0.5, seq_k-0.5, seq_q-0.5, -0.5], origin='upper')
            
            # Set integer ticks for x and y axes
            ax.set_xticks(range(seq_k))
            ax.set_yticks(range(seq_q))
            # Format ticks as integers to avoid fractional labels
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            # Set title and labels
            ax.set_title(f'Layer {layer_idx + 1} Features, Head {head_idx + 1} (Avg over Items)', fontsize=12)
            ax.set_xlabel('Key Feature')
            ax.set_ylabel('Query Feature')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        # Save the figure with head number in filename
        head_filename = f"{base_filename}_head_{head_idx + 1}{ext}"
        output_path = os.path.join(output_dir, head_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Features attention maps for head {head_idx + 1} saved to {output_path}")
    


def create_sample_data_for_visualization(num_samples: int = 10, num_features: int = 4) -> np.ndarray:
    """
    Create sample data for attention visualization.
    
    Args:
        num_samples: Number of samples to generate
        num_features: Number of features per sample
        
    Returns:
        Sample data array
    """
    np.random.seed(42)
    return np.random.randn(num_samples, num_features)


def create_mock_attention_visualization(input_data: np.ndarray, output_dir: str = '.', 
                                      filename: str = 'attentions.png', 
                                      num_layers: int = 6, num_heads: int = 8) -> None:
    """
    Create a mock attention visualization for demonstration purposes.
    This function creates synthetic attention patterns to show the visualization format.
    
    Args:
        input_data: Input data array
        output_dir: Directory to save the image
        filename: Name of the output file
        num_layers: Number of transformer layers to simulate
        num_heads: Number of attention heads per layer
    """
    print("Creating mock attention visualization...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create synthetic attention patterns
    seq_len = input_data.shape[1] if len(input_data.shape) > 1 else input_data.shape[0]
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_layers, num_heads, figsize=(num_heads * 2.5, num_layers * 2.5))
    
    # Handle single layer or single head cases
    if num_layers == 1 and num_heads == 1:
        axes = [[axes]]
    elif num_layers == 1:
        axes = [axes]
    elif num_heads == 1:
        axes = [[ax] for ax in axes]
    
    # Generate synthetic attention patterns
    np.random.seed(42)
    
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            ax = axes[layer_idx][head_idx]
            
            # Create different attention patterns for different layers/heads
            if layer_idx == 0:
                # First layer: more uniform attention
                attn_map = np.random.uniform(0.1, 0.3, (seq_len, seq_len))
            elif layer_idx < num_layers // 2:
                # Middle layers: diagonal patterns
                attn_map = np.eye(seq_len) * 0.5 + np.random.uniform(0.05, 0.15, (seq_len, seq_len))
            else:
                # Later layers: more focused attention
                attn_map = np.random.uniform(0.05, 0.2, (seq_len, seq_len))
                # Add some strong attention to specific positions
                focus_pos = np.random.randint(0, seq_len)
                attn_map[focus_pos, :] = np.random.uniform(0.3, 0.8, seq_len)
            
            # Normalize to make it look like attention probabilities
            attn_map = attn_map / attn_map.sum(axis=1, keepdims=True)
            
            # Plot the attention map
            im = ax.imshow(attn_map, cmap='Blues', aspect='auto', vmin=0, vmax=1)
            
            # Set title and labels
            ax.set_title(f'Layer {layer_idx + 1}, Head {head_idx + 1}', fontsize=9)
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Add overall title
    fig.suptitle('TabPFN Attention Maps (Mock Visualization)', fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Save the figure
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f'Mock attention maps saved to {output_path}')


def demo_attention_visualization():
    """
    Demo function to show how to use the attention visualization.
    """
    print("Creating demo TabPFN model and visualizing attention...")
    
    # Create sample data
    X_sample = create_sample_data_for_visualization(num_samples=5, num_features=4)
    y_sample = np.random.randn(5)  # Random targets for regression
    
    # Initialize TabPFN regressor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    regressor = TabPFNRegressor(device=device, n_estimators=1)
    
    # Fit the model
    regressor.fit(X_sample, y_sample)
    
    # Try to visualize real attention first
    try:
        visualize_attention_heads(
            model=regressor,
            input_data=X_sample,
            output_dir='.',
            filename='attentions.png',
            device=device,
            sample_idx=0,
            max_layers=4  # Show first 4 layers to avoid overcrowding
        )
        print("Real attention visualization completed!")
    except Exception as e:
        print(f"Real attention extraction failed: {e}")
        print("Creating mock attention visualization instead...")
        create_mock_attention_visualization(
            input_data=X_sample,
            output_dir='.',
            filename='attentions.png',
            num_layers=4,
            num_heads=8
        )
    
    print("Demo completed! Check 'attentions.png' for the visualization.")


if __name__ == "__main__":
    demo_attention_visualization()
    print("nigga")
