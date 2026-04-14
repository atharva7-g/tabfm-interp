from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Union
import json
import torch
from datetime import datetime


@dataclass
class ExperimentConfig:
    corrupt_idx: Union[int, List[int]]
    noise_std: float
    seed: int
    n_train_samples: int
    corruption_mode: str = "gaussian_replace"
    corruption_strength: float = 1.0
    patch_dim: Optional[int] = 2
    max_layers: Optional[int] = None
    ratio_epsilon: float = 0.05

    def to_dict(self):
        return asdict(self)


class BaseExperiment:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_results(
        self, results: dict, subdir: str, filename_base: str, script_path: str
    ):
        """Save JSON metadata with script path"""
        save_dir = self.output_dir / subdir
        save_dir.mkdir(parents=True, exist_ok=True)

        results_with_meta = {
            **results,
            "script_path": script_path,
            "timestamp": self.timestamp,
        }

        filepath = save_dir / f"{filename_base}_{self.timestamp}.json"
        with open(filepath, "w") as f:
            json.dump(results_with_meta, f, indent=2)
        return filepath

    def save_tensors(self, tensors: dict, subdir: str, filename_base: str):
        """Save raw tensors as .pt files"""
        save_dir = self.output_dir / subdir
        save_dir.mkdir(parents=True, exist_ok=True)

        filepath = save_dir / f"{filename_base}_{self.timestamp}.pt"
        torch.save(tensors, filepath)
        return filepath
