# TabPFN Patching Dashboard

A Streamlit web interface for viewing experiment results, managing configs, and running new experiments.

## Features

- **Results Viewer**: Browse all experiment runs with interactive plots
- **Config Manager**: Create, view, and select from saved configurations  
- **Run Experiment**: Launch new experiments with selected configs
- **Compare Experiments**: Side-by-side comparison of multiple runs

## Usage

### Start the Dashboard

```bash
uv run streamlit run src/dashboard/app.py
```

Then open your browser to `http://localhost:8501`

### Navigation

Use the sidebar to switch between:

1. **Results Viewer** - See all your experiment results with plots
2. **Config Manager** - Create new configs or select existing ones
3. **Run Experiment** - Launch experiments with selected configs
4. **Compare Experiments** - Compare multiple runs side-by-side

### File Structure

- Results are loaded from: `src/experiments/full_layer_patching/output/`
- Configs are loaded from: `src/experiments/hooks/`

## Screenshots

The dashboard shows:
- Restoration curves by layer
- Recovery ratios
- Config summaries
- Comparison tables
- Full JSON config views
