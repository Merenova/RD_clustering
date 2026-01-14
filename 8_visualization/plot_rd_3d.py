#!/usr/bin/env -S uv run python
"""
3D Visualization of Rate-Distortion Objective over beta-gamma sweep.

Creates 3D surface plots showing:
1. H (entropy/rate) vs (beta, gamma)
2. Weighted Distortion: gamma*D_e + (1-gamma)*D_a vs (beta, gamma)
3. D_e (embedding distortion) vs (beta, gamma)
4. D_a (attribution distortion) vs (beta, gamma)
5. L_total (full RD objective) vs (beta, gamma)
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')


def load_sweep_data(validation_dir: Path, prefix_id: str = None):
    """Load sweep data from JSON files.

    Returns:
        dict: {prefix_id: list of per_config entries}
    """
    data = {}

    if prefix_id:
        files = [validation_dir / f"{prefix_id}_clustering_sweep_1209.json"]
    else:
        files = list(validation_dir.glob("cloze_*_clustering_sweep_1209.json"))

    for f in files:
        if not f.exists():
            continue
        try:
            with open(f) as fp:
                sweep = json.load(fp)

            pid = sweep.get("prefix_id", f.stem.replace("_clustering_sweep_1209", ""))

            if "rd_objective" in sweep and "per_config" in sweep["rd_objective"]:
                data[pid] = sweep["rd_objective"]["per_config"]
        except Exception as e:
            print(f"Error loading {f}: {e}")

    return data


def aggregate_rd_data(sweep_data: dict):
    """Aggregate RD data across all prefixes into beta-gamma grid.

    Returns:
        dict with keys: beta_values, gamma_values, H, D_e, D_a, D_weighted, L_total
              each value is a 2D array of shape (n_beta, n_gamma)
    """
    # Collect all data points
    all_points = defaultdict(list)  # (beta, gamma) -> list of dicts

    for prefix_id, configs in sweep_data.items():
        for cfg in configs:
            # Skip configs missing required keys
            if not all(k in cfg for k in ["beta", "gamma", "H", "D_e", "D_a", "L_total"]):
                continue
            key = (cfg["beta"], cfg["gamma"])
            all_points[key].append({
                "H": cfg["H"],
                "D_e": cfg["D_e"],
                "D_a": cfg["D_a"],
                "L_total": cfg["L_total"],
                "K": cfg.get("n_components", cfg.get("K", 0))
            })

    # Get unique beta and gamma values
    beta_values = sorted(set(k[0] for k in all_points.keys()))
    gamma_values = sorted(set(k[1] for k in all_points.keys()))

    n_beta = len(beta_values)
    n_gamma = len(gamma_values)

    # Create 2D arrays (mean across prefixes)
    H = np.zeros((n_beta, n_gamma))
    D_e = np.zeros((n_beta, n_gamma))
    D_a = np.zeros((n_beta, n_gamma))
    D_e_std = np.zeros((n_beta, n_gamma))
    D_a_std = np.zeros((n_beta, n_gamma))
    D_weighted = np.zeros((n_beta, n_gamma))
    L_total = np.zeros((n_beta, n_gamma))
    K = np.zeros((n_beta, n_gamma))

    for i, beta in enumerate(beta_values):
        for j, gamma in enumerate(gamma_values):
            points = all_points.get((beta, gamma), [])
            if points:
                H[i, j] = np.mean([p["H"] for p in points])
                
                de_vals = [p["D_e"] for p in points]
                da_vals = [p["D_a"] for p in points]
                
                D_e[i, j] = np.mean(de_vals)
                D_a[i, j] = np.mean(da_vals)
                
                D_e_std[i, j] = np.std(de_vals)
                D_a_std[i, j] = np.std(da_vals)
                
                L_total[i, j] = np.mean([p["L_total"] for p in points])
                K[i, j] = np.mean([p["K"] for p in points])
                # Weighted distortion: gamma*D_e + (1-gamma)*D_a
                D_weighted[i, j] = gamma * D_e[i, j] + (1 - gamma) * D_a[i, j]

    return {
        "beta_values": np.array(beta_values),
        "gamma_values": np.array(gamma_values),
        "H": H,
        "D_e": D_e,
        "D_a": D_a,
        "D_e_std": D_e_std,
        "D_a_std": D_a_std,
        "D_weighted": D_weighted,
        "L_total": L_total,
        "K": K
    }


def plot_3d_surface(ax, X, Y, Z, title, xlabel="beta", ylabel="gamma", zlabel="Value", cmap='viridis'):
    """Plot a 3D surface on the given axes."""
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none', alpha=0.8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_zlabel(zlabel, fontsize=10)
    ax.set_title(title, fontsize=12)
    return surf


def create_rd_3d_visualization(agg_data: dict, output_dir: Path, prefix_label: str = "All Prefixes"):
    """Create 3D visualizations of RD components."""

    beta = agg_data["beta_values"]
    gamma = agg_data["gamma_values"]

    # Create meshgrid
    BETA, GAMMA = np.meshgrid(beta, gamma, indexing='ij')

    # Figure 1: 2x2 grid of main components
    fig1 = plt.figure(figsize=(16, 14))
    fig1.suptitle(f"Rate-Distortion Components ({prefix_label})", fontsize=14, y=0.98)

    # Plot 1: H (Entropy/Rate)
    ax1 = fig1.add_subplot(2, 2, 1, projection='3d')
    plot_3d_surface(ax1, BETA, GAMMA, agg_data["H"],
                    "H (Entropy)", "beta", "gamma", "H", cmap='Blues')

    # Plot 2: Weighted Distortion
    ax2 = fig1.add_subplot(2, 2, 2, projection='3d')
    plot_3d_surface(ax2, BETA, GAMMA, agg_data["D_weighted"],
                    "Weighted Distortion\n(gamma*D_e + (1-gamma)*D_a)",
                    "beta", "gamma", "D", cmap='Reds')

    # Plot 3: D_e (Embedding Distortion)
    ax3 = fig1.add_subplot(2, 2, 3, projection='3d')
    plot_3d_surface(ax3, BETA, GAMMA, agg_data["D_e"],
                    "D_e (Embedding Distortion)", "beta", "gamma", "D_e", cmap='Greens')

    # Plot 4: D_a (Attribution Distortion)
    ax4 = fig1.add_subplot(2, 2, 4, projection='3d')
    plot_3d_surface(ax4, BETA, GAMMA, agg_data["D_a"],
                    "D_a (Attribution Distortion)", "beta", "gamma", "D_a", cmap='Oranges')

    plt.tight_layout()
    out1 = output_dir / "rd_3d_components.png"
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out1}")

    # Figure 2: L_total and K
    fig2 = plt.figure(figsize=(16, 7))
    fig2.suptitle(f"Rate-Distortion Objective and Cluster Count ({prefix_label})", fontsize=14, y=0.98)

    # Plot 1: L_total
    ax1 = fig2.add_subplot(1, 2, 1, projection='3d')
    surf1 = plot_3d_surface(ax1, BETA, GAMMA, agg_data["L_total"],
                            "L_total = H + beta*(gamma*D_e + (1-gamma)*D_a)",
                            "beta", "gamma", "L_total", cmap='plasma')

    # Plot 2: K (number of clusters)
    ax2 = fig2.add_subplot(1, 2, 2, projection='3d')
    surf2 = plot_3d_surface(ax2, BETA, GAMMA, agg_data["K"],
                            "K (Number of Clusters)",
                            "beta", "gamma", "K", cmap='viridis')

    plt.tight_layout()
    out2 = output_dir / "rd_3d_objective.png"
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out2}")

    # Figure 3: Interactive-style plot with multiple views
    fig3 = plt.figure(figsize=(18, 12))
    fig3.suptitle(f"Rate-Distortion Landscape ({prefix_label})", fontsize=14, y=0.98)

    views = [(30, 45), (30, 135), (30, 225), (30, 315), (60, 45), (0, 0)]
    titles = ["View 1 (30°, 45°)", "View 2 (30°, 135°)", "View 3 (30°, 225°)",
              "View 4 (30°, 315°)", "View 5 (60°, 45°)", "Top View"]

    for i, (elev, azim) in enumerate(views):
        ax = fig3.add_subplot(2, 3, i+1, projection='3d')
        surf = ax.plot_surface(BETA, GAMMA, agg_data["L_total"],
                               cmap='plasma', edgecolor='none', alpha=0.8)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("beta", fontsize=9)
        ax.set_ylabel("gamma", fontsize=9)
        ax.set_zlabel("L_total", fontsize=9)
        ax.set_title(titles[i], fontsize=10)

    plt.tight_layout()
    out3 = output_dir / "rd_3d_multiview.png"
    plt.savefig(out3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out3}")

    # Figure 4: Heatmaps for easier interpretation
    fig4, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig4.suptitle(f"Rate-Distortion Heatmaps ({prefix_label})", fontsize=14)

    data_list = [
        (agg_data["H"], "H (Entropy)", "Blues"),
        (agg_data["D_weighted"], "Weighted D", "Reds"),
        (agg_data["L_total"], "L_total", "plasma"),
        (agg_data["D_e"], "D_e", "Greens"),
        (agg_data["D_a"], "D_a", "Oranges"),
        (agg_data["K"], "K (Clusters)", "viridis")
    ]

    for ax, (data, title, cmap) in zip(axes.flat, data_list):
        im = ax.imshow(data.T, aspect='auto', origin='lower', cmap=cmap,
                       extent=[beta.min(), beta.max(), gamma.min(), gamma.max()])
        ax.set_xlabel("beta")
        ax.set_ylabel("gamma")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    out4 = output_dir / "rd_heatmaps.png"
    plt.savefig(out4, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out4}")

    return [out1, out2, out3, out4]


def create_single_prefix_visualization(prefix_data: list, output_dir: Path, prefix_id: str):
    """Create visualization for a single prefix."""

    # Convert to grid format
    beta_gamma_map = {}
    for cfg in prefix_data:
        # Skip configs missing required keys
        if not all(k in cfg for k in ["beta", "gamma", "H", "D_e", "D_a", "L_total"]):
            continue
        key = (cfg["beta"], cfg["gamma"])
        beta_gamma_map[key] = cfg

    beta_values = sorted(set(k[0] for k in beta_gamma_map.keys()))
    gamma_values = sorted(set(k[1] for k in beta_gamma_map.keys()))

    n_beta = len(beta_values)
    n_gamma = len(gamma_values)

    H = np.zeros((n_beta, n_gamma))
    D_e = np.zeros((n_beta, n_gamma))
    D_a = np.zeros((n_beta, n_gamma))
    D_weighted = np.zeros((n_beta, n_gamma))
    L_total = np.zeros((n_beta, n_gamma))
    K = np.zeros((n_beta, n_gamma))

    for i, beta in enumerate(beta_values):
        for j, gamma in enumerate(gamma_values):
            cfg = beta_gamma_map.get((beta, gamma))
            if cfg:
                H[i, j] = cfg["H"]
                D_e[i, j] = cfg["D_e"]
                D_a[i, j] = cfg["D_a"]
                L_total[i, j] = cfg["L_total"]
                K[i, j] = cfg.get("n_components", cfg.get("K", 0))
                D_weighted[i, j] = gamma * D_e[i, j] + (1 - gamma) * D_a[i, j]

    agg_data = {
        "beta_values": np.array(beta_values),
        "gamma_values": np.array(gamma_values),
        "H": H, "D_e": D_e, "D_a": D_a,
        "D_weighted": D_weighted, "L_total": L_total, "K": K
    }

    # Create output subdir for this prefix
    prefix_output = output_dir / prefix_id
    prefix_output.mkdir(parents=True, exist_ok=True)

    return create_rd_3d_visualization(agg_data, prefix_output, prefix_id)


def main():
    parser = argparse.ArgumentParser(description="3D Rate-Distortion Visualization")
    parser.add_argument("--validation-dir", type=Path,
                        default=Path("results/validation"),
                        help="Directory with sweep JSON files")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/visualization"),
                        help="Output directory for plots")
    parser.add_argument("--prefix-id", type=str, default=None,
                        help="Single prefix to visualize (default: aggregate all)")
    parser.add_argument("--per-prefix", action="store_true",
                        help="Generate separate plots for each prefix")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading sweep data from {args.validation_dir}...")
    sweep_data = load_sweep_data(args.validation_dir, args.prefix_id)
    print(f"Loaded data for {len(sweep_data)} prefixes")

    if not sweep_data:
        print("No sweep data found!")
        return

    if args.prefix_id:
        # Single prefix
        if args.prefix_id in sweep_data:
            create_single_prefix_visualization(
                sweep_data[args.prefix_id], args.output_dir, args.prefix_id)
        else:
            print(f"Prefix {args.prefix_id} not found")
    else:
        # Aggregate visualization
        print("Creating aggregate visualization...")
        agg_data = aggregate_rd_data(sweep_data)
        create_rd_3d_visualization(agg_data, args.output_dir, f"Aggregate ({len(sweep_data)} prefixes)")

        # Per-prefix if requested
        if args.per_prefix:
            print("\nGenerating per-prefix visualizations...")
            for pid, pdata in sweep_data.items():
                print(f"  Processing {pid}...")
                create_single_prefix_visualization(pdata, args.output_dir, pid)

    print("\nDone!")


if __name__ == "__main__":
    main()
