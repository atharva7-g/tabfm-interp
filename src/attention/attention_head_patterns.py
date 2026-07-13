#!/usr/bin/env python3
"""
Extract and visualize attention head entropy patterns from TabPFN's
self_attn_between_features.

Produces a figure with two entropy heatmaps side by side:
  Left  — Multiplication dataset (d=3, 4 feature blocks)
  Right — Pairwise-50 dataset   (d=50, 35 feature blocks)

Both show normalised entropy (0=fully concentrated, 1=uniform) for each
head x layer combination.  The cross-dataset consistency of Head 2's low
entropy at L0, L6, and L13 is the main visual claim.

ENTROPY FIX: Entropy is computed per sample first, then averaged -- NOT on
the batch-averaged attention matrix.  Averaging first overestimates entropy
via Jensen's inequality (H(E[A]) >= E[H(A)] since entropy is concave).
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.datasets.synthetic import create_dataset
from src.attention.attention_maps import extract_attention_weights_from_tabpfn

KEY_LAYERS = [0, 6, 12, 13, 17]
N_EVAL = 32
SEED = 42
DPI = 200


def _fit_model(dataset_type: str, device: str) -> Tuple[TabPFNRegressor, np.ndarray]:
    X, y = create_dataset(dataset_type, num_samples=1000, seed=SEED)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.8, random_state=SEED
    )
    model = TabPFNRegressor(device=device, n_estimators=1)
    model.fit(X_train, y_train)
    return model, X_test[:N_EVAL]


def _extract_features_attn(
    model, X: np.ndarray, device: str
) -> Dict[int, np.ndarray]:
    weights = extract_attention_weights_from_tabpfn(model, X, device)
    features = {}
    for k, v in sorted(weights.items()):
        if "_features" not in k:
            continue
        layer_idx = int(k.split("_")[1])
        features[layer_idx] = v.cpu().numpy() if isinstance(v, torch.Tensor) else v
    return features


def _normalized_entropy(
    features_attn: Dict[int, np.ndarray], layer: int, head: int
) -> float:
    """
    Normalised entropy H(K|Q) / log(n_keys), averaged correctly.

    Compute entropy PER SAMPLE then average -- not on the batch-averaged
    attention matrix, which overestimates via Jensen's inequality.

    Returns value in [0, 1]: 0 = fully concentrated, 1 = uniform.
    """
    arr = features_attn[layer]        # [batch, n_heads, seq_q, seq_k]
    if arr.ndim == 4:
        arr = arr[:, head, :, :]      # [batch, seq_q, seq_k]
    n_keys = arr.shape[2]
    if n_keys <= 1:
        return 1.0
    max_ent = np.log(n_keys)
    eps = 1e-12
    row_entropies = -np.sum(arr * np.log(arr + eps), axis=2)  # [batch, seq_q]
    return float(row_entropies.mean()) / max_ent


def _build_entropy_table(attn: Dict[int, np.ndarray], n_heads: int) -> np.ndarray:
    """Returns array of shape [n_heads, len(KEY_LAYERS)]."""
    table = np.zeros((n_heads, len(KEY_LAYERS)))
    for h in range(n_heads):
        for ci, layer in enumerate(KEY_LAYERS):
            if layer in attn:
                table[h, ci] = _normalized_entropy(attn, layer, h)
    return table


def _plot_entropy_panel(ax, table, title, cmap, vmin, vmax, show_ylabel=False):
    n_heads, n_layers = table.shape
    im = ax.imshow(table, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"L{l}" for l in KEY_LAYERS], fontsize=10)
    ax.set_yticks(range(n_heads))
    ax.set_yticklabels([f"Head {h}" for h in range(n_heads)], fontsize=10)
    ax.set_xlabel("Layer", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    if show_ylabel:
        ax.set_ylabel("Head", fontsize=11, fontweight="bold")
    # Annotate cells
    for i in range(n_heads):
        for j in range(n_layers):
            val = table[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")
    return im


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Fitting Multiplication model...")
    model_mult, X_mult = _fit_model("multiplication", device)
    print("Extracting Multiplication attention...")
    attn_mult = _extract_features_attn(model_mult, X_mult, device)

    print("Fitting Pairwise-50 model...")
    model_pw, X_pw = _fit_model("pairwise_50", device)
    print("Extracting Pairwise-50 attention...")
    attn_pw = _extract_features_attn(model_pw, X_pw, device)

    sample_layer = list(attn_mult.keys())[0]
    n_heads = attn_mult[sample_layer].shape[1]

    table_mult = _build_entropy_table(attn_mult, n_heads)
    table_pw   = _build_entropy_table(attn_pw,   n_heads)

    print("\nNormalised entropy -- Multiplication:")
    header = f"{'Head':<8}" + "".join(f"L{l:<7}" for l in KEY_LAYERS)
    print(header)
    for h in range(n_heads):
        print(f"Head {h:<3}" + "".join(f"{table_mult[h, ci]:.3f}  "
              for ci in range(len(KEY_LAYERS))))

    print("\nNormalised entropy -- Pairwise-50:")
    print(header)
    for h in range(n_heads):
        print(f"Head {h:<3}" + "".join(f"{table_pw[h, ci]:.3f}  "
              for ci in range(len(KEY_LAYERS))))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        1, 2, figsize=(8.5, 2.8),
        gridspec_kw={"wspace": 0.38, "left": 0.08, "right": 0.88,
                     "top": 0.78, "bottom": 0.18}
    )

    vmin, vmax = 0.0, 1.0
    cmap = "YlOrRd"

    _plot_entropy_panel(
        axes[0], table_mult,
        title=r"Multiplication ($d=3$, 4 blocks)",
        cmap=cmap, vmin=vmin, vmax=vmax, show_ylabel=True
    )

    im = _plot_entropy_panel(
        axes[1], table_pw,
        title=r"Pairwise-50 ($d=50$, 35 blocks)",
        cmap=cmap, vmin=vmin, vmax=vmax, show_ylabel=False
    )

    # Shared colourbar
    cbar_ax = fig.add_axes([0.90, 0.18, 0.016, 0.60])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label("Normalised Entropy", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Head 2 attention is most selective at layers 0, 6, and 13 across both datasets\n"
        r"(lower entropy $\rightarrow$ more concentrated attention)",
        fontsize=10, fontweight="bold", y=1.02
    )

    output_dir = Path("docs/attention_head_patching")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "attention_head_patterns.png"
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()