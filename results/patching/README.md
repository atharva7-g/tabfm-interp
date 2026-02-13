# Feature Attention Activation Patching Experiments

## Overview

This directory contains activation patching experiments that test the causal role of attention mechanisms in TabPFN models.

### Purpose
Tests whether specific attention components (heads or tokens) are causally responsible for computing certain features by:
1. Running inference on a **clean input** (original features)
2. Running inference on a **corrupted input** (one feature replaced with noise)
3. **Patching** activations from the clean run into the corrupted run at each layer
4. Measuring how much the output is **restored** toward the clean output

### Methodology
- **Clean Input**: Original test sample
- **Corrupted Input**: Same sample but with one feature column corrupted with Gaussian noise
- **Activation Patching**: Cache activations from clean input at each transformer layer, then patch them into the corrupted input run at that same layer
- **Measurement**: Track restoration (y_patched - y_corrupt) and recovery ratio (restoration / clean-corrupt gap)

## Datasets

### Multiplication Dataset
- **Formula**: `y = a * b + c`
- **Features**: 3 columns [a, b, c]
- **Common configurations**:
  - Sample sizes: 1000-6000 samples
  - Train/Test splits: 50/50 or 80/20
  - Feature ranges: Standard normal distribution (mean=0, std=1)

### Simple Addition Dataset
- **Formula**: `y = a + b`
- **Features**: 2 columns [a, b]
- **Sample sizes**: 4000-6000 samples
- **Train/Test splits**: 50/50

### Dataset Creation
Datasets are generated using functions from `src/utils/utils.py`:
- `create_multiplication_dataset()`: Generates a*b + c data
- `create_simple_dataset()`: Generates a + b data
- Controlled random seeds for reproducibility

## Patching Dimensions

The experiment supports patching along different dimensions:

### Dimension 1: Tokens (565 tokens)
- Patches a specific token position across all attention heads
- `patch_idx` range: [0, 564]
- Use case: Test causal importance of specific input positions

### Dimension 2: Attention Heads (4 heads)
- Patches a specific attention head across all tokens
- `patch_idx` range: [0, 3]
- Use case: Test causal importance of specific attention heads

### Dimension None: Full Layer Output
- Patches the entire layer output (all tokens and all heads)
- `patch_idx` is ignored when using this mode
- Use case: Test the causal importance of the complete layer computation
- **Implementation**: Returns `cached_activation.clone()` to replace the entire layer output tensor
- **When to use**: 
  - To establish a baseline maximum restoration achievable by patching any component
  - When you want to test if information is distributed across the entire layer
  - When individual head/token patching shows low recovery, suggesting distributed computation

## Key Functions

### 1. `create_corrupted_input()`
Generates corrupted input by adding Gaussian noise to a specific feature column.

**Parameters:**
- `X_clean`: Original input array
- `corrupt_idx`: Index of feature to corrupt (which column)
- `noise_std`: Standard deviation of noise
- `seed`: Random seed for reproducibility

### 2. `sweep_layers()`
Iterates through all transformer layers and runs patching experiment on each.

**Parameters:**
- `regressor`: Trained TabPFNRegressor
- `X_clean`: Clean input samples
- `X_corrupt`: Corrupted input samples
- `corrupt_idx`: Index of corrupted feature
- `n_train_samples`: Number of training samples used
- `patch_idx`: Which token or head to patch (depends on patch_dim). Ignored when patch_dim=None.
- `patch_dim`: Dimension to patch along (1=tokens, 2=heads, None=full layer output)
- `max_layers`: Optional limit on number of layers to test

### 3. `run_single_layer_patching()`
Executes patching for a single layer and computes metrics.

**Returns Dictionary:**
- `y_clean`: Output on clean input
- `y_corrupt`: Output on corrupted input
- `y_patched`: Output after patching
- `restoration`: y_patched - y_corrupt
- `recovery_ratio`: restoration / (y_clean - y_corrupt)
- `layer_idx`: Layer index

### 4. `plot_restoration_results()`
Visualizes restoration metrics across all layers.

**Generates:**
- Plot 1: Restoration value by layer
- Plot 2: Recovery ratio (%) by layer
- Optional: Saves to file

### 5. `run_feature_attention_causal_patching_experiment()`
Main orchestrator function that runs the complete experiment.

**Parameters:**
- `regressor`: Trained model
- `X_clean`: Clean test input
- `corrupt_idx`: Feature to corrupt
- `n_train_samples`: Training set size
- `patch_idx`: Which token/head to patch
- `patch_dim`: Dimension to patch (1=tokens, 2=heads, None=full layer output)
- `noise_std`: Noise level
- `noise_seed`: Random seed
- `max_layers`: Limit layers tested
- `plot`: Whether to generate plots
- `save_path`: Optional path to save plot

## Expected Output Metrics

### Per-Layer Results
The experiment outputs a table with the following columns for each layer:
- **Layer**: Layer index (0 to N-1)
- **y_patched**: Model output after patching
- **Restoration**: Difference between patched and corrupted outputs
- **Recovery %**: Percentage of clean-corrupt gap recovered

### Key Metrics
- **Clean Output**: Baseline prediction on uncorrupted input
- **Corrupted Output**: Prediction on corrupted input
- **Target Gap**: y_clean - y_corrupt (maximum possible restoration)
- **Best Layer**: Layer with highest absolute recovery ratio
- **Best Recovery**: Maximum recovery percentage achieved

## Results Interpretation

### High Recovery Ratio (>50%)
Indicates the patched attention component at that layer is causally important for computing the corrupted feature.

### Low Recovery Ratio (<20%)
Suggests either:
- The attention component is not critical for that feature
- Information is distributed across multiple components
- Computation happens at different layers

### Layer Trends
- **Early layers (0-3)**: May show lower recovery if information hasn't been processed yet
- **Middle layers (4-8)**: Often show peak recovery as features are actively computed
- **Late layers (9-11)**: May show declining recovery if information is already integrated into final representation

## File Structure

```
results/patching/
├── README.md           # This file - common experiment documentation
├── run_1/              # Individual experiment runs
│   ├── details.md      # Run-specific parameters and results
│   └── plots/          # Generated visualizations
├── run_2/
│   └── ...
```

## Script Location
- **File**: `src/hooks/feature_attention_patching.py`
- **Lines**: 313 lines

## Dependencies
- `src/utils/model_inspector.py`: Model inspection utilities
- `src/utils/shape_inspector.py`: Tensor shape tracking
- `src/utils/utils.py`: Dataset creation and seed setting

## Related Experiments
- `src/activation_patching/activation_patching_regression.py`: General activation patching
- `src/hooks/hooks.py`: Hook management utilities
- `src/hooks/activation_patching_regression_viz.py`: Visualization variants

## Requirements
- Python 3.12+
- PyTorch
- tabpfn>=2.0.3
- NumPy
- Matplotlib
- scikit-learn

## Notes
- This experiment focuses specifically on attention mechanisms (`self_attn_between_features`)
- Can patch either tokens (dim=1) or attention heads (dim=2)
- Corruption is applied to a single feature column
- The method assumes the model uses transformer encoder with self-attention layers
- Random seeds should be documented in individual run files for reproducibility
