# plot_scalability.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_scalability_file(
    csv_path: str,
    *,
    metric: str = "time",          # "time" or "utility"
    utility_metric: str = "util_count",  # if metric="utility": "util_count" or "util_theta"
    save_png: str | None = None,
    title: str | None = None
) -> None:
    """
    Quick plot for scalability CSVs produced by the experiments.

    - Auto-detects 1D vs 2D:
        * 1D: x = nbins   (for discretized) OR domain_size (for full)
        * 2D: x = nbins_age * nbins_priors (for discretized) OR domain_size (for full)
    - metric="time": plots total_time = time_build + time_star
    - metric="utility": plots chosen utility ("util_count" or "util_theta")
    - curves per (bias_kind, label), e.g., "per_value · q_star"
    """
    df = pd.read_csv(csv_path).copy()
    if df.empty:
        print("[plot] nothing to plot (empty file).")
        return

    # --- derive columns we need ---
    df["total_time"] = df.get("time_build", 0.0) + df.get("time_star", 0.0)

    is_2d = ("nbins_age" in df.columns) and ("nbins_priors" in df.columns)

    if is_2d:
        # complexity = #grid cells when discretized, else use domain_size for "full"
        df["complexity"] = np.where(
            df["mode"].eq("discretized"),
            df["nbins_age"].astype(float) * df["nbins_priors"].astype(float),
            df["domain_size"].astype(float),
        )
        x_label = "Grid size (#cells) or domain size (full)"
    else:
        # complexity = #bins when discretized, else domain_size for "full"
        # nbins may be NaN for full runs; replace with domain_size for consistency
        nbins = df.get("nbins", np.nan)
        df["complexity"] = np.where(
            df["mode"].eq("discretized"),
            nbins.astype(float),
            df["domain_size"].astype(float),
        )
        x_label = "Bins (discretized) or domain size (full)"

    if metric == "time":
        y_col = "total_time"
        y_label = "Total time (s)"
        default_title = "Scalability: total time vs discretization/domain size"
    elif metric == "utility":
        if utility_metric not in {"util_count", "util_theta"}:
            raise ValueError("utility_metric must be 'util_count' or 'util_theta'")
        y_col = utility_metric
        y_label = utility_metric
        default_title = f"Utility ({utility_metric}) vs discretization/domain size"
    else:
        raise ValueError("metric must be 'time' or 'utility'")

    # --- plotting ---
    plt.figure(figsize=(8, 5))
    # Sort ensures nice monotone x when drawing lines
    df = df.sort_values("complexity")

    # One line per (bias_kind, label)
    for (bk, lbl), g in df.groupby(["bias_kind", "label"], dropna=False):
        x = g["complexity"].values
        y = g[y_col].values
        plt.plot(x, y, marker="o", linewidth=2, label=f"{bk} · {lbl}")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(alpha=0.3)
    plt.legend(frameon=False, loc="best")

    ttl = title or default_title
    if ttl:
        plt.title(ttl)

    plt.tight_layout()

    if save_png:
        Path(save_png).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_png, dpi=150, bbox_inches="tight")
        print(f"[plot] saved: {save_png}")
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    # Example usage:
    # 1D results (e.g., ../results/scalability_1d.csv)
    plot_scalability_file("../results/scalability_1d.csv", metric="time",
                          save_png="../results/plot_1d_time.png")
    plot_scalability_file("../results/scalability_1d.csv", metric="utility", utility_metric="util_count",
                          save_png="../results/plot_1d_util_count.png")

    # # 2D results (e.g., ../results/scalability_2d.csv)
    # plot_scalability_file("../results/scalability_2d.csv", metric="time",
    #                       save_png="../results/plot_2d_time.png")
    # plot_scalability_file("../results/scalability_2d.csv", metric="utility", utility_metric="util_theta",
    #                       save_png="../results/plot_2d_util_theta.png")