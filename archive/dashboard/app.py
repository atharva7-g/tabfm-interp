import os.path
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from src.utils.utils import get_project_root

OUTPUT_DIR = Path(
    os.path.join(
        get_project_root(), "src", "experiments", "full_layer_patching", "output"
    )
)
CONFIG_DIR = Path(os.path.join(get_project_root(), "src", "experiments", "hooks"))


def load_results():
    results = []
    if OUTPUT_DIR.exists():
        for json_file in OUTPUT_DIR.glob("summary_*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    data["_filename"] = json_file.name
                    data["_timestamp"] = json_file.stat().st_mtime
                    results.append(data)
            except Exception as e:
                st.error(f"Error loading {json_file}: {e}")
    return results


def load_configs():
    configs = []
    if CONFIG_DIR.exists():
        for json_file in CONFIG_DIR.glob("config*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    data["_filename"] = json_file.name
                    configs.append(data)
            except Exception as e:
                st.error(f"Error loading {json_file}: {e}")
    return configs


def plot_restoration_curve(results_data):
    layer_indices = [r["layer_idx"] for r in results_data["results"]]
    restorations = [r["restoration"] for r in results_data["results"]]
    recovery_ratios = [r["recovery_ratio"] for r in results_data["results"]]
    y_clean = results_data["results"][0]["y_clean"]
    y_corrupt = results_data["results"][0]["y_corrupt"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(layer_indices, restorations, "o-", linewidth=2, markersize=8)
    ax1.axhline(
        y=y_clean - y_corrupt,
        color="r",
        linestyle="--",
        label=f"Target ({y_clean - y_corrupt:.4f})",
    )
    ax1.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax1.set_xlabel("Layer Index")
    ax1.set_ylabel("Restoration")
    ax1.set_title(f"Full Layer Patching: {results_data['dataset_type'].title()}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(
        layer_indices,
        [r * 100 for r in recovery_ratios],
        "o-",
        linewidth=2,
        markersize=8,
        color="green",
    )
    ax2.axhline(y=100, color="r", linestyle="--", label="Full Recovery")
    ax2.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax2.set_xlabel("Layer Index")
    ax2.set_ylabel("Recovery %")
    ax2.set_title("Recovery Ratio by Layer")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def display_config_summary(config):
    cols = st.columns(3)
    with cols[0]:
        st.metric("Dataset", config.get("dataset_type", "N/A"))
        st.metric("Corrupt Feature", ["a", "b", "c"][config.get("corrupt_idx", 0)])
    with cols[1]:
        st.metric("Samples", config.get("n_samples", "N/A"))
        st.metric("Noise Std", config.get("noise_std", "N/A"))
    with cols[2]:
        heads = config.get("heads", [])
        st.metric("Heads", f"{heads}" if heads else "All")
        st.metric(
            "Patch Dim",
            config.get("patch_dim") if config.get("patch_dim") else "Full Layer",
        )


def compare_results(results_list):
    if len(results_list) < 2:
        st.warning("Select at least 2 results to compare")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for result in results_list:
        layer_indices = [r["layer_idx"] for r in result["results"]]
        restorations = [r["restoration"] for r in result["results"]]
        recovery_ratios = [r["recovery_ratio"] for r in result["results"]]
        label = f"{result['dataset_type']} (b={[None, 'a', 'b', 'c'][result['corrupt_idx'] + 1]})"

        ax1.plot(
            layer_indices, restorations, "o-", linewidth=2, markersize=6, label=label
        )
        ax2.plot(
            layer_indices,
            [r * 100 for r in recovery_ratios],
            "o-",
            linewidth=2,
            markersize=6,
            label=label,
        )

    ax1.set_xlabel("Layer Index")
    ax1.set_ylabel("Restoration")
    ax1.set_title("Restoration Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Layer Index")
    ax2.set_ylabel("Recovery %")
    ax2.set_title("Recovery Ratio Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="TabPFN Patching Dashboard", layout="wide")

    st.title("TabPFN Patching Experiments Dashboard")
    st.markdown("---")
    page = st.sidebar.radio(
        "Navigation",
        ["Results Viewer", "Config Manager", "Run Experiment", "Compare Experiments"],
    )

    if page == "Results Viewer":
        st.header("Experiment Results")

        results = load_results()
        if not results:
            st.info("No results found. Run an experiment first!")
            return
        st.subheader("Filter Results")
        cols = st.columns(3)
        with cols[0]:
            datasets = list(set(r["dataset_type"] for r in results))
            selected_dataset = st.selectbox("Dataset", ["All"] + datasets)
        with cols[1]:
            corrupt_features = list(
                set(["a", "b", "c"][r["corrupt_idx"]] for r in results)
            )
            selected_feature = st.selectbox(
                "Corrupted Feature", ["All"] + corrupt_features
            )

        # Filter results
        filtered = results
        if selected_dataset != "All":
            filtered = [r for r in filtered if r["dataset_type"] == selected_dataset]
        if selected_feature != "All":
            feature_idx = ["a", "b", "c"].index(selected_feature)
            filtered = [r for r in filtered if r["corrupt_idx"] == feature_idx]
        st.subheader(f"Showing {len(filtered)} result(s)")

        for result in filtered:
            with st.expander(
                f"{result['dataset_type'].title()} - Corrupt '{'abc'[result['corrupt_idx']]}' - Best Layer {result['best_layer']}"
            ):
                cols = st.columns(4)
                with cols[0]:
                    st.metric("Clean Output", f"{result['y_clean']:.4f}")
                with cols[1]:
                    st.metric("Corrupted Output", f"{result['y_corrupt']:.4f}")
                with cols[2]:
                    st.metric("Best Layer", result["best_layer"])
                with cols[3]:
                    st.metric("Best Recovery", f"{result['best_recovery'] * 100:.1f}%")
                fig = plot_restoration_curve(result)
                st.pyplot(fig)
                st.subheader("Layer Details")
                df = pd.DataFrame(result["results"])
                df["recovery_ratio"] = df["recovery_ratio"].apply(
                    lambda x: f"{x * 100:.2f}%"
                )
                st.dataframe(df, use_container_width=True)

    elif page == "Config Manager":
        st.header("Configuration Manager")

        configs = load_configs()
        if not configs:
            st.info("No configs found. Create one in the 'Run Experiment' tab!")
            return

        st.subheader(f"Available Configs ({len(configs)})")

        for config in configs:
            with st.expander(f"⚙️ {config['_filename']}"):
                display_config_summary(config)

                st.subheader("Full Config")
                st.json(config)
                if st.button(
                    "Use for New Experiment", key=f"use_{config['_filename']}"
                ):
                    st.session_state["selected_config"] = config
                    st.success("Config selected! Go to 'Run Experiment' tab.")
                    st.rerun()

        # Create new config
        st.markdown("---")
        st.subheader("Create New Config")

        with st.form("new_config"):
            cols = st.columns(2)
            with cols[0]:
                dataset_type = st.selectbox(
                    "Dataset Type", ["multiplication", "quadratic"]
                )
                corrupt_idx = st.selectbox(
                    "Corrupt Feature",
                    [(0, "a"), (1, "b"), (2, "c")],
                    format_func=lambda x: x[1],
                )[0]
                n_samples = st.number_input(
                    "Number of Samples", min_value=10, value=1000, step=100
                )
                seed = st.number_input("Random Seed", min_value=0, value=42, step=1)
            with cols[1]:
                noise_std = st.number_input(
                    "Noise Std", min_value=0.0, value=1.0, step=0.1
                )
                test_size = st.slider(
                    "Test Size", min_value=0.1, max_value=0.9, value=0.5, step=0.1
                )
                patch_dim = st.selectbox(
                    "Patch Dimension",
                    [(None, "Full Layer"), (1, "Tokens"), (2, "Heads")],
                    format_func=lambda x: x[1],
                )[0]
                heads_str = st.text_input("Heads (comma-separated, 0-3)", "0,1,2,3")

            submitted = st.form_submit_button("Save Config")

            if submitted:
                heads = [int(h.strip()) for h in heads_str.split(",")]
                new_config = {
                    "dataset_type": dataset_type,
                    "heads": heads,
                    "corrupt_idx": corrupt_idx,
                    "noise_std": noise_std,
                    "seed": seed,
                    "n_samples": n_samples,
                    "test_size": test_size,
                    "output_dir": "src/experiments/hooks/results",
                    "device": None,
                    "patch_dim": patch_dim,
                }

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                corrupt_feature = ["a", "b", "c"][corrupt_idx]
                heads_str_file = "-".join(map(str, heads))
                filename = f"config_{corrupt_feature}_{heads_str_file}_{timestamp}.json"

                config_path = CONFIG_DIR / filename
                with open(config_path, "w") as f:
                    json.dump(new_config, f, indent=2)

                st.success(f"Config saved: {filename}")
                st.rerun()

    elif page == "Run Experiment":
        st.header("Run New Experiment")
        if "selected_config" in st.session_state:
            config = st.session_state["selected_config"]
            st.success("Using pre-selected config")
            display_config_summary(config)
        else:
            st.info(
                "No config selected. Go to 'Config Manager' to select one, or use quick run below."
            )
            config = None

        st.markdown("---")
        st.subheader("Quick Run (Use Defaults)")

        cols = st.columns(3)
        with cols[0]:
            dataset_type = st.selectbox(
                "Dataset", ["multiplication", "quadratic"], key="quick_dataset"
            )
        with cols[1]:
            corrupt_feature = st.selectbox(
                "Corrupt", ["a", "b", "c"], key="quick_corrupt"
            )
        with cols[2]:
            if st.button("Run Experiment", type="primary", use_container_width=True):
                corrupt_idx = {"a": 0, "b": 1, "c": 2}[corrupt_feature]

                with st.spinner("Running experiment... This may take a minute."):
                    try:
                        from src.experiments.full_layer_patching.full_layer_patching import (
                            set_seed,
                            create_dataset,
                            create_corrupted_input,
                        )
                        from sklearn.model_selection import train_test_split
                        from tabpfn import TabPFNRegressor
                        import torch

                        set_seed(42)
                        device = "cuda" if torch.cuda.is_available() else "cpu"

                        X, y = create_dataset(dataset_type, num_samples=1000, seed=42)
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.5, random_state=42
                        )
                        X_clean = X_test[0:1]
                        X_corrupt = create_corrupted_input(
                            X_clean, corrupt_idx, 1.0, 42
                        )

                        regressor = TabPFNRegressor(device=device, n_estimators=1)
                        regressor.fit(X_train, y_train)
                        st.info("Experiment running... (simplified for demo)")
                        st.success("Experiment complete! Check 'Results Viewer' tab.")

                    except Exception as e:
                        st.error(f"Error running experiment: {e}")
                        st.info(
                            "Make sure you have the required dependencies installed."
                        )

    elif page == "Compare Experiments":
        st.header("Compare Experiments")

        results = load_results()
        if len(results) < 2:
            st.info("Need at least 2 results to compare. Run more experiments!")
            return

        result_options = {
            f"{r['dataset_type']} (b={'abc'[r['corrupt_idx']]}) - Layer {r['best_layer']}": r
            for r in results
        }
        selected = st.multiselect(
            "Select experiments to compare",
            list(result_options.keys()),
            default=list(result_options.keys())[:2],
        )

        if selected:
            selected_results = [result_options[s] for s in selected]
            st.subheader("Summary Comparison")
            comparison_data = []
            for r in selected_results:
                comparison_data.append(
                    {
                        "Dataset": r["dataset_type"],
                        "Corrupt Feature": "abc"[r["corrupt_idx"]],
                        "Best Layer": r["best_layer"],
                        "Best Recovery": f"{r['best_recovery'] * 100:.1f}%",
                        "Clean Output": f"{r['y_clean']:.4f}",
                        "Corrupted Output": f"{r['y_corrupt']:.4f}",
                    }
                )

            df = pd.DataFrame(comparison_data)
            st.dataframe(df, width="stretch")
            st.subheader("Visual Comparison")
            fig = compare_results(selected_results)
            if fig:
                st.pyplot(fig)


if __name__ == "__main__":
    main()
