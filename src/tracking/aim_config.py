import os.path
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import aim
from aim import Run
from src.utils.utils import get_project_root


DEFAULT_AIM_REPO = Path(os.path.join(get_project_root(), ".aim"))


def get_aim_repo_path() -> Path:
    return DEFAULT_AIM_REPO


def init_aim_repo(repo_path: Optional[Path] = None) -> Path:
    """Initialize Aim repository if it doesn't exist.

    Args:
            repo_path: Path to Aim repository. Uses default if None.

    Returns:
            Path to the Aim repository.
    """
    repo_path = repo_path or DEFAULT_AIM_REPO

    if not repo_path.exists():
        print(f"Initializing Aim repository at {repo_path}")
        repo_path.mkdir(parents=True, exist_ok=True)

    return repo_path


def create_aim_run(
    experiment_name: str = "tabpfn-interpretability",
    repo_path: Optional[Path] = None,
    tags: Optional[List[str]] = None,
) -> Optional[Any]:
    """Create a new Aim Run for tracking.

    Args:
            experiment_name: Name of the experiment.
            repo_path: Path to Aim repository. Uses default if None.
            tags: Optional list of tags to attach to the run.

    Returns:
            Aim Run object or None if aim not available.
    """
    repo_path = repo_path or DEFAULT_AIM_REPO

    run = Run(
        repo=str(repo_path),
        experiment=experiment_name,
    )

    if tags:
        for tag in tags:
            run.add_tag(tag)

    return run


def log_experiment_params(run: Any, config: Dict[str, Any]) -> None:
    """Log experiment parameters to Aim.

    Args:
            run: Aim Run object.
            config: Dictionary of configuration parameters.
    """
    for key, value in config.items():
        if value is None:
            continue
        elif isinstance(value, (list, tuple)):
            run[key] = ",".join(map(str, value))
        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                if subvalue is not None:
                    run[f"{key}.{subkey}"] = subvalue
        else:
            run[key] = value


def log_patching_metrics(
    run: Any,
    layer_idx: int,
    restoration: float,
    recovery_ratio: float,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    run.track(restoration, name="restoration", step=layer_idx, context=context)
    run.track(recovery_ratio, name="recovery_ratio", step=layer_idx, context=context)


def log_summary_metrics(
    run: Any,
    y_clean: float,
    y_corrupt: float,
    best_recovery: float,
    best_layer: int,
    **additional_metrics,
) -> None:
    run.track(y_clean, name="y_clean")
    run.track(y_corrupt, name="y_corrupt")
    run.track(best_recovery, name="best_recovery")
    run.track(best_layer, name="best_layer")

    for metric_name, metric_value in additional_metrics.items():
        if isinstance(metric_value, (int, float)):
            run.track(metric_value, name=metric_name)
        else:
            # Non-numeric values (strings, etc.) are stored as run metadata
            run[metric_name] = metric_value


def log_artifact(
    run: Any,
    artifact_path: Union[str, Path],
    name: Optional[str] = None,
    caption: Optional[str] = None,
) -> None:
    """Log an artifact (file) to Aim.

    Args:
            run: Aim Run object.
            artifact_path: Path to the artifact file.
            name: Optional name for the artifact.
            caption: Optional caption for image artifacts.
    """
    artifact_path = Path(artifact_path)
    if not artifact_path.exists():
        print(f"Warning: Artifact not found: {artifact_path}")
        return

    # Log based on file type
    suffix = artifact_path.suffix.lower()

    if suffix in [".png", ".jpg", ".jpeg", ".gif"]:
        from aim import Image

        run.track(
            Image(str(artifact_path), caption=caption or ""),
            name=name or artifact_path.stem,
        )
    elif suffix == ".json":
        import json

        with open(artifact_path) as f:
            data = json.load(f)
        run[name or artifact_path.stem] = data
    else:
        # Store path as metadata
        run[f"artifact_{name or artifact_path.stem}"] = str(artifact_path)


class AimExperimentTracker:
    """Context manager for Aim experiment tracking."""

    def __init__(
        self,
        experiment_name: str = "tabpfn-interpretability",
        repo_path: Optional[Path] = None,
        tags: Optional[List[str]] = None,
    ):
        self.experiment_name = experiment_name
        self.repo_path = repo_path
        self.tags = tags
        self.run: Optional[Any] = None

    def __enter__(self):
        self.run = create_aim_run(
            experiment_name=self.experiment_name,
            repo_path=self.repo_path,
            tags=self.tags,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.run is not None:
            if exc_type is not None:
                self.run["error"] = str(exc_val)
                self.run["error_type"] = exc_type.__name__
            self.run.close()

    def log_params(self, config: Dict[str, Any]) -> None:
        log_experiment_params(self.run, config)

    def log_patching_layer(
        self,
        layer_idx: int,
        restoration: float,
        recovery_ratio: float,
        head_idx: Optional[int] = None,
        token_idx: Optional[int] = None,
    ) -> None:
        if token_idx is not None:
            context = {"token": token_idx}
        elif head_idx is not None:
            context = {"head": head_idx}
        else:
            context = None
        log_patching_metrics(self.run, layer_idx, restoration, recovery_ratio, context)

    def log_ablation_layer(
        self,
        layer_idx: int,
        effect: float,
        ratio: float,
        head_idx: Optional[int] = None,
        token_idx: Optional[int] = None,
    ) -> None:
        context = {}
        if token_idx is not None:
            context["token"] = token_idx
        if head_idx is not None:
            context["head"] = head_idx

        self.run.track(
            effect,
            name="ablation_effect",
            context=context if context else None,
            step=layer_idx,
        )
        self.run.track(
            ratio,
            name="ablation_ratio",
            context=context if context else None,
            step=layer_idx,
        )

    def log_summary(
        self,
        y_clean: float,
        y_corrupt: float,
        best_recovery: float,
        best_layer: int,
        **additional_metrics,
    ) -> None:
        log_summary_metrics(
            self.run,
            y_clean,
            y_corrupt,
            best_recovery,
            best_layer,
            **additional_metrics,
        )

    def log_artifacts(
        self,
        artifact_paths: Union[List[Path], Dict[Path, str]],
        names: Optional[List[str | None]] = None,
    ):
        # If artifact_paths is a dict, treat it as path -> caption mapping
        if isinstance(artifact_paths, dict):
            for path, caption in artifact_paths.items():
                log_artifact(self.run, path, caption=caption)
        else:
            if names is None:
                names = [None] * len(artifact_paths)

            for path, name in zip(artifact_paths, names):
                log_artifact(self.run, path, name)


def setup_experiment_tracking(
    config: Dict[str, Any],
    experiment_type: str = "patching",
    tags: Optional[List[str]] = None,
) -> AimExperimentTracker:
    init_aim_repo()

    all_tags = tags or []
    all_tags.append(experiment_type)

    if "dataset_type" in config:
        all_tags.append(config["dataset_type"])

    tracker = AimExperimentTracker(
        experiment_name=f"tabpfn-{experiment_type}",
        tags=all_tags,
    )

    return tracker


if __name__ == "__main__":
    print(f"Repository path: {get_aim_repo_path()}")

    repo = init_aim_repo()
    print(f"Repository initialized at: {repo}")

    print("\nCreating test run...")
    with AimExperimentTracker(
        experiment_name="test",
        tags=["demo", "test"],
    ) as tracker:
        tracker.log_params(
            {
                "dataset_type": "multiplication",
                "seed": 42,
                "n_samples": 1000,
            }
        )

        for i in range(5):
            tracker.log_patching_layer(
                layer_idx=i,
                restoration=float(i) * 0.1,
                recovery_ratio=float(i) * 0.05,
            )

        tracker.log_summary(
            y_clean=3.5,
            y_corrupt=1.2,
            best_recovery=0.95,
            best_layer=3,
        )
        print("Test run logged successfully!")
