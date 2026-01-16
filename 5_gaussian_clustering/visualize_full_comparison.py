import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

def main():
    base_dir = Path("/home/hyunjin/latent_planning/Qwen3_4B_results/sweep_span_full_pool_sum/results/compare_clustering")
    full_grid_dir = base_dir / "full_grid"
    output_path = base_dir / "full_comparison_rd_curves.png"
    
    data = []
    files = list(full_grid_dir.glob("*_full_comparison.json"))
    print(f"Loading {len(files)} comparison files...")
    
    for fpath in files:
        try:
            with open(fpath) as f:
                res = json.load(f)
            
            for pt in res["results"]:
                beta = pt["beta"]
                gamma = pt["gamma"]
                # pt contains Adaptive, Fixed_E, Fixed_A dicts with H, D_e, D_a
                
                # Compute combined D for each
                def get_stats(method, stats):
                    d_comb = gamma * stats["D_e"] + (1-gamma) * stats["D_a"]
                    return {
                        "beta": beta,
                        "gamma": gamma,
                        "method": method,
                        "Rate (H)": stats["H"],
                        "Distortion (D)": d_comb
                    }
                    
                data.append(get_stats("Adaptive (Full)", pt["Adaptive"]))
                data.append(get_stats("Fixed K (E-init)", pt["Fixed_E"]))
                data.append(get_stats("Fixed K (A-init)", pt["Fixed_A"]))
                
        except Exception as e:
            # print(f"Error {fpath}: {e}")
            continue
            
    df = pd.DataFrame(data)
    
    # Aggregate over prefixes
    agg = df.groupby(["beta", "gamma", "method"]).mean().reset_index()
    
    # Plot
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    
    gammas = sorted(agg["gamma"].unique())
    palette = sns.color_palette("viridis", len(gammas))
    color_map = {g: palette[i] for i, g in enumerate(gammas)}
    
    line_styles = {
        "Adaptive (Full)": "-",
        "Fixed K (E-init)": "--",
        "Fixed K (A-init)": ":"
    }
    markers = {
        "Adaptive (Full)": "o",
        "Fixed K (E-init)": "^",
        "Fixed K (A-init)": "s"
    }
    
    # Loop gammas
    for gamma in gammas:
        gamma_subset = agg[agg["gamma"] == gamma]
        
        for method in ["Adaptive (Full)", "Fixed K (E-init)", "Fixed K (A-init)"]:
            subset = gamma_subset[gamma_subset["method"] == method].sort_values("beta")
            
            # Label only for Adaptive to keep legend clean?
            # Or use custom legend.
            
            plt.plot(
                subset["Distortion (D)"],
                subset["Rate (H)"],
                marker=markers[method],
                linestyle=line_styles[method],
                color=color_map[gamma],
                linewidth=1.5 if method == "Adaptive (Full)" else 1,
                markersize=6 if method == "Adaptive (Full)" else 4,
                alpha=0.9
            )
            
    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=color_map[g], lw=2, label=f"$\gamma={g}$") for g in gammas]
    legend_elements.append(Line2D([0], [0], color='black', lw=0, label=" "))
    legend_elements.append(Line2D([0], [0], color='black', linestyle='-', marker='o', label="Adaptive"))
    legend_elements.append(Line2D([0], [0], color='black', linestyle='--', marker='^', label="Fixed K (E-init)"))
    legend_elements.append(Line2D([0], [0], color='black', linestyle=':', marker='s', label="Fixed K (A-init)"))
    
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.xlabel("Weighted Distortion $D = \gamma D_e + (1-\gamma) D_a$")
    plt.ylabel("Rate $H$")
    plt.title("Full Rate-Distortion Comparison: Adaptive vs Fixed K (Averaged across 97 Prefixes)")
    plt.grid(True, alpha=0.3)
    
    # Magnified Inset
    zoom_x_min, zoom_x_max = 2.0, 3.5
    zoom_y_min, zoom_y_max = 1.5, 3.5
    
    axins = zoomed_inset_axes(ax, 2.5, loc='lower left', bbox_to_anchor=(0.1, 0.1), bbox_transform=ax.transAxes)
    
    for gamma in gammas:
        gamma_subset = agg[agg["gamma"] == gamma]
        for method in ["Adaptive (Full)", "Fixed K (E-init)", "Fixed K (A-init)"]:
            subset = gamma_subset[gamma_subset["method"] == method].sort_values("beta")
            axins.plot(
                subset["Distortion (D)"],
                subset["Rate (H)"],
                marker=markers[method],
                linestyle=line_styles[method],
                color=color_map[gamma],
                linewidth=1.5 if method == "Adaptive (Full)" else 1,
                markersize=6 if method == "Adaptive (Full)" else 4,
                alpha=0.9
            )
            
    axins.set_xlim(zoom_x_min, zoom_x_max)
    axins.set_ylim(zoom_y_min, zoom_y_max)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved full comparison plot to {output_path}")

if __name__ == "__main__":
    main()




