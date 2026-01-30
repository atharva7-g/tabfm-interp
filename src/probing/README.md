# Probing Experiments

This directory contains scripts for probing transformer representations to understand what information is encoded where.

## Categories

### 1. Answer Probing (`answer_probing/`)
**Question:** Can we predict the final output from intermediate representations?

- `linear_probing_for_ans.py` - Basic answer probing across layers
- `linear_probing_for_ans_harder.py` - More challenging probing scenarios
- `linear_probing_ans_harder_results.py` - Results visualization and analysis

**Use case:** Understand at which layer the model has sufficient information to predict the final answer.

### 2. Input Probing (`input_probing/`)
**Question:** Does the model preserve input information through layers?

- `linear_probing_for_input_results.py` - Recover a, b, c, a+b, a+c from activations

**Use case:** Test if input values are still accessible in intermediate representations.

### 3. Intermediate Value Probing (`intermediate_value_probing/`)
**Question:** Does the model compute intermediate steps (e.g., a*b in y=a*b+c)?

#### Core Implementation
- `core/intermediate_value_probe.py` - Main implementation testing a*b and a/b recovery
- `core/intermediate_value_probe_results.py` - Results generation with comprehensive plots

#### Experiments
- `experiments/intermediate_value_probe_a_times_b.py` - Focused experiment on a*b recovery only

**Use case:** Discover if the model explicitly computes intermediate values or learns end-to-end shortcuts.

### 4. Cross-Dataset Probing (`cross_dataset_probing/`)
**Question:** Do learned probes generalize across different datasets?

#### Separate Datasets
Train on datasets with some weights, test on datasets with completely different weights.
- `separate_datasets/separate_datasets_probe.py` - Basic cross-dataset probing
- `separate_datasets/separate_datasets_probe_normalised_ver.py` - Normalized version (recommended)

#### Same Dataset (Random Split)
Train/test on random splits of a combined dataset with multiple relationships.
- `same_dataset/same_dataset_random_split_probe.py` - Basic implementation
- `same_dataset/same_dataset_random_split_probe_results.py` - **Main script with full analysis**
- `same_dataset/same_dataset_random_split_stronger_diffw1w2.py` - Harder test conditions

#### Enhanced Probing
Advanced probing experiments with layer sweeps and complexity analysis.
- `enhanced/enhanced_linear_probe_copy.py` - Sweeps all layers to find best performing layer
- `enhanced/enhanced_linear_probe_same_dataset_random_split_better.py` - Improved random split methodology
- `enhanced/enhanced_linear_probe_sanity.py` - Sanity check: can probe recover inputs?

**Use case:** Understand how well probes generalize and which layers encode transferable features.

## Quick Start

### To probe for answer:
```bash
python answer_probing/linear_probing_for_ans.py
```

### To probe for intermediate values:
```bash
python intermediate_value_probing/core/intermediate_value_probe.py
```

### To test cross-dataset generalization:
```bash
python cross_dataset_probing/same_dataset/same_dataset_random_split_probe_results.py
```

### To find the best layer:
```bash
python cross_dataset_probing/enhanced/enhanced_linear_probe_copy.py
```

## Typical Workflow

1. **Start with answer probing** to see if outputs are predictable from activations
2. **Try input probing** to verify information preservation
3. **Use intermediate value probing** to discover computational structure
4. **Test cross-dataset generalization** to validate findings across different scenarios

## Understanding Results

- **R² score close to 1.0**: Strong linear relationship, information is clearly encoded
- **R² score close to 0.0**: No linear relationship, information may not be present or is nonlinearly encoded
- **High R² in early layers**: Information is preserved from inputs
- **High R² in late layers**: Information emerges through computation
- **Good cross-dataset R²**: Probe has learned generalizable features, not dataset-specific patterns
