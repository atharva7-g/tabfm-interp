"""Interactive configuration manager for patching experiments."""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


DEFAULT_CONFIG = {
    "dataset_type": "multiplication",
    "heads": [0, 1, 2, 3],
    "corrupt_idx": 2,
    "noise_std": 1.0,
    "corruption_mode": "gaussian_replace",
    "corruption_strength": 1.0,
    "seed": 42,
    "n_samples": 1000,
    "test_size": 0.5,
    "output_dir": "src/experiments/hooks/results",
    "device": None,
    "patch_dim": 2,
}

VALID_DATASET_TYPES = ["multiplication", "quadratic", "pairwise_50"]

PARAM_DESCRIPTIONS = {
    "dataset_type": "Dataset type (multiplication, quadratic, pairwise_50)",
    "heads": "Attention heads to patch (list: 0-3)",
    "tokens": "Tokens to patch (list: 0 to num_tokens-1)",
    "corrupt_idx": "Feature index/indices to corrupt (e.g., 2 or 0,1,2)",
    "noise_std": "Noise standard deviation",
    "corruption_mode": "Corruption mode (gaussian_replace, gaussian_add, mean_shift, scale, sign_flip, fixed, zero, permute)",
    "corruption_strength": "Corruption strength scalar (>=0)",
    "seed": "Random seed",
    "n_samples": "Dataset size",
    "test_size": "Test split (0-1)",
    "output_dir": "Output directory",
    "device": "Device (cuda/cpu/auto)",
    "patch_dim": "Patch dimension: 1=tokens, 2=heads, null=full layer",
}


def print_menu(config: Dict[str, Any]) -> None:
    """Print the parameter selection menu."""
    print("\n" + "=" * 60)
    print("CONFIGURATION MENU")
    print("=" * 60)
    print("\nCurrent configuration:")
    for i, (key, value) in enumerate(config.items(), 1):
        desc = PARAM_DESCRIPTIONS[key]
        print(f"  {i}. {desc}")
        print(f"     Current: {value}")
    print("\n  9. Done - Run experiment")
    print("  0. Cancel and exit")
    print("=" * 60)


def parse_input_value(key: str, user_input: str) -> Any:
    """Parse user input into the appropriate type."""
    user_input = user_input.strip()

    if key == "heads":
        if "," in user_input:
            return [int(x.strip()) for x in user_input.split(",")]
        else:
            return [int(x.strip()) for x in user_input.split()]

    elif key == "tokens":
        if "," in user_input:
            return [int(x.strip()) for x in user_input.split(",")]
        else:
            return [int(x.strip()) for x in user_input.split()]

    elif key == "corrupt_idx":
        if "," in user_input:
            return [int(x.strip()) for x in user_input.split(",") if x.strip() != ""]
        return int(user_input)

    elif key in ["seed", "n_samples"]:
        return int(user_input)

    elif key in ["noise_std", "test_size"]:
        return float(user_input)

    elif key == "corruption_strength":
        return float(user_input)

    elif key == "corruption_mode":
        return user_input.strip()

    elif key == "device":
        if user_input.lower() in ["auto", "", "none"]:
            return None
        return user_input.lower()

    elif key == "patch_dim":
        if user_input.lower() in ["null", "none", ""]:
            return None
        return int(user_input)

    else:
        return user_input


def validate_value(key: str, value: Any) -> tuple[bool, str]:
    """Validate a configuration value."""
    if key == "dataset_type":
        if value not in VALID_DATASET_TYPES:
            return False, f"must be one of {VALID_DATASET_TYPES}"

    elif key == "corrupt_idx":
        if isinstance(value, int):
            if value < 0:
                return False, "must be >= 0"
        elif isinstance(value, list):
            if len(value) == 0:
                return False, "list must be non-empty"
            if not all(isinstance(v, int) and v >= 0 for v in value):
                return False, "list values must be non-negative integers"
        else:
            return False, "must be an int or a list of ints"

    elif key == "heads":
        if not isinstance(value, list) or len(value) == 0:
            return False, "must be non-empty list"
        if not all(0 <= h <= 3 for h in value):
            return False, "indices must be 0-3"

    elif key == "tokens":
        if not isinstance(value, list) or len(value) == 0:
            return False, "must be non-empty list"
        if not all(isinstance(t, int) and t >= 0 for t in value):
            return False, "must be non-negative integers"

    elif key == "noise_std":
        if value < 0:
            return False, "must be >= 0"

    elif key == "corruption_strength":
        if value < 0:
            return False, "must be >= 0"

    elif key == "corruption_mode":
        valid_modes = [
            "gaussian_replace",
            "gaussian_add",
            "mean_shift",
            "scale",
            "sign_flip",
            "fixed",
            "zero",
            "permute",
        ]
        if value not in valid_modes:
            return False, f"must be one of {valid_modes}"

    elif key == "test_size":
        if not 0 < value < 1:
            return False, "must be between 0 and 1"

    elif key == "n_samples":
        if value < 10:
            return False, "must be >= 10"

    elif key == "device":
        if value is not None and value not in ["cuda", "cpu"]:
            return False, "must be 'cuda', 'cpu', or 'auto'"

    elif key == "patch_dim":
        if value is not None and value not in [1, 2]:
            return False, "must be 1 (tokens), 2 (heads), or null (full layer)"

    return True, ""


def get_parameter_value(key: str, current_value: Any) -> Any:
    """Ask user for a new parameter value."""
    desc = PARAM_DESCRIPTIONS[key]
    print(f"\n{desc}")
    print(f"Current: {current_value}")
    print("Enter new value (or press Enter to keep):")

    while True:
        user_input = input("> ").strip()

        if user_input == "":
            return current_value

        try:
            new_value = parse_input_value(key, user_input)
            is_valid, error_msg = validate_value(key, new_value)

            if is_valid:
                return new_value
            else:
                print(f"Invalid: {error_msg}")
                print("Try again (or Enter to keep):")
        except ValueError as e:
            print(f"Invalid format: {e}")
            print("Try again (or Enter to keep):")


def interactive_config() -> Optional[Dict[str, Any]]:
    """Run interactive configuration."""
    print("\n" + "=" * 60)
    print("Patching Experiment Configuration")
    print("=" * 60)

    print("\nUse default configuration?")
    print("  Y - Use defaults")
    print("  n - Customize")
    print("\nDefaults:")
    for key, value in DEFAULT_CONFIG.items():
        print(f"  {key}: {value}")

    use_defaults = input("\n[Y/n]: ").strip().lower()

    if use_defaults in ["", "y", "yes"]:
        return DEFAULT_CONFIG.copy()

    config = DEFAULT_CONFIG.copy()

    while True:
        print_menu(config)

        choice = input("\nSelect parameters to change (e.g., 1 2 3): ").strip()

        if choice == "0":
            print("\nExiting.")
            return None

        if choice == "9":
            break

        try:
            selections = [int(x) for x in choice.split()]
            keys = list(config.keys())

            for selection in selections:
                if 1 <= selection <= len(keys):
                    key = keys[selection - 1]
                    config[key] = get_parameter_value(key, config[key])
                else:
                    print(f"Invalid: {selection}")
        except ValueError:
            print("Invalid input. Enter numbers separated by spaces.")

    return config


def save_config(config: Dict[str, Any]) -> Path:
    """Save configuration to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    corrupt_idx = config["corrupt_idx"]
    if isinstance(corrupt_idx, list):
        corrupt_feature = f"feat_{len(corrupt_idx)}"
    else:
        if corrupt_idx < 3:
            corrupt_feature = ["a", "b", "c"][corrupt_idx]
        else:
            corrupt_feature = f"feat_{corrupt_idx}"
    heads_str = "-".join(map(str, config["heads"]))

    # Simplified filename (dataset_type is now in the folder path)
    filename = f"config_{timestamp}.json"

    # Dataset-specific subdirectory
    save_dir = Path(config["output_dir"]) / config["dataset_type"] / "configs"
    save_dir.mkdir(parents=True, exist_ok=True)

    config_path = save_dir / filename

    config_with_meta = {
        **config,
        "_saved_at": timestamp,
    }

    with open(config_path, "w") as f:
        json.dump(config_with_meta, f, indent=2)

    print(f"\nConfig saved: {config_path}")
    return config_path


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        config = json.load(f)

    config.pop("_saved_at", None)
    return config


if __name__ == "__main__":
    config = interactive_config()
    if config:
        print("\nFinal config:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        save_config(config)
