"""Example integration of Aim tracking with existing experiments."""

"""
This file shows how to integrate Aim tracking into your experiments.

To use Aim tracking in your experiments:

1. Import the tracking module:
   from src.tracking import AimExperimentTracker, setup_experiment_tracking

2. Wrap your experiment in the tracker context manager:
   
   with setup_experiment_tracking(config, experiment_type="patching") as tracker:
       # Your experiment code here
       tracker.log_params(config)
       
       # During layer sweep
       for layer_idx in range(n_layers):
           restoration = compute_restoration(...)
           recovery_ratio = compute_recovery_ratio(...)
           tracker.log_patching_layer(layer_idx, restoration, recovery_ratio)
       
       # At the end
       tracker.log_summary(y_clean, y_corrupt, best_recovery, best_layer)

Example integration with full_layer_patching.py:
"""

# Example: How to modify full_layer_patching.py

EXAMPLE_INTEGRATION = '''
# Add at the top of the file
from src.tracking import AimExperimentTracker, setup_experiment_tracking

def run_full_layer_patching_with_tracking(config):
    """Run patching experiment with Aim tracking."""
    
    # Initialize tracking
    with setup_experiment_tracking(
        config=config,
        experiment_type="full_layer_patching",
        tags=[config["dataset_type"], f"corrupt_{config['corrupt_idx']}"],
    ) as tracker:
        
        # Log all configuration parameters
        tracker.log_params(config)
        
        # Your existing experiment code...
        X, y = create_dataset(config["dataset_type"], ...)
        # ... setup code ...
        
        # During layer sweep
        results = []
        for layer_idx in range(n_layers):
            # Your existing patching code...
            restoration = y_patched - y_corrupt
            recovery_ratio = restoration / (y_clean - y_corrupt) if (y_clean - y_corrupt) != 0 else 0
            
            # Log to Aim
            tracker.log_patching_layer(layer_idx, restoration, recovery_ratio)
            
            results.append({
                "layer_idx": layer_idx,
                "restoration": restoration,
                "recovery_ratio": recovery_ratio,
            })
        
        # Find best layer
        best_result = max(results, key=lambda x: x["recovery_ratio"])
        
        # Log summary metrics
        tracker.log_summary(
            y_clean=y_clean,
            y_corrupt=y_corrupt,
            best_recovery=best_result["recovery_ratio"],
            best_layer=best_result["layer_idx"],
        )
        
        # Save and log artifacts
        save_results(results, output_path)
        tracker.log_artifacts([output_path / "summary.json", output_path / "plot.png"])

# Run the tracked experiment
if __name__ == "__main__":
    config = load_config("path/to/config.json")
    run_full_layer_patching_with_tracking(config)
'''

# Quick start example
QUICK_START = """
# To get started with Aim tracking:

# 1. First, add aim to your dependencies in pyproject.toml:
#    dependencies = [
#        ...
#        "aim>=3.29.0",
#    ]

# 2. Install the dependency:
#    uv sync

# 3. Initialize the Aim repository (happens automatically on first run)

# 4. Run an experiment with tracking (see example above)

# 5. View results:
#    aim up
#    
#    This starts the Aim UI at http://localhost:43800

# 6. Track metrics programmatically:
from src.tracking import AimExperimentTracker

with AimExperimentTracker(experiment_name="my-experiment") as tracker:
    tracker.log_params({"lr": 0.001, "batch_size": 32})
    
    for epoch in range(10):
        loss = train_epoch(...)
        tracker.track(loss, name="loss", step=epoch)
"""

if __name__ == "__main__":
    print("Aim Tracking Integration Examples")
    print("=" * 60)
    print("\n1. Example integration code:")
    print(EXAMPLE_INTEGRATION)
    print("\n2. Quick start guide:")
    print(QUICK_START)
