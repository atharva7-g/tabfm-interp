"""Minimal tracking integration example."""

from src.tracking import AimExperimentTracker

with AimExperimentTracker(experiment_name="my-exp", tags=["test"]) as tracker:
    tracker.log_params({"lr": 0.001, "dataset": "multiplication"})

    for epoch in range(10):
        loss = 0.1 / (epoch + 1)
        tracker.track(loss, name="loss", step=epoch)

    tracker.log_summary(y_clean=5.0, y_corrupt=3.0, best_recovery=0.9, best_layer=2)
