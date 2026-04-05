"""
visualize.py — Phase 3 benchmark visualization

Reads benchmark_results/ CSVs produced by benchmark_runner and generates:
  1. Convergence curves (cost vs evaluations) per problem, all solvers overlaid
  2. Box-plots of final best cost per (solver, problem)
  3. ECDF curves for a given problem

Usage:
  python visualize.py --results-dir benchmark_results --output-dir plots
  python visualize.py --mode convergence --problem rastrigin_2d
  python visualize.py --mode ecdf --problem rastrigin_10d
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

# ── Color scheme ───────────────────────────────────────────────────────────────
SOLVER_COLORS = {
    "cmaes": "#E74C3C",
    "xnes":  "#3498DB",
    "svgd":  "#2ECC71",
    "mppi":  "#F39C12",
}
SOLVER_LABELS = {
    "cmaes": "CMA-ES",
    "xnes":  "xNES",
    "svgd":  "SVGD",
    "mppi":  "MPPI",
}


# ── 1. Convergence curves ──────────────────────────────────────────────────────

def plot_convergence(results_dir: Path, output_dir: Path, problem: str = None):
    """
    For each problem, overlay convergence curves of all solvers.
    X-axis: function evaluations (from eval_history).
    Y-axis: best cost (log scale).
    Shaded region: 25th–75th percentile across seeds.
    """
    # Collect per-solver per-problem convergence files
    conv_files = list(results_dir.glob("*_conv.csv"))
    if not conv_files:
        print("No *_conv.csv files found. Run benchmark_runner first.")
        return

    # Group by problem
    problems = set()
    for f in conv_files:
        stem = f.stem  # e.g. "cmaes_rastrigin_2d_conv"
        parts = stem.replace("_conv", "").split("_", 1)
        if len(parts) == 2:
            problems.add(parts[1])

    if problem:
        problems = {p for p in problems if p == problem}

    for prob in sorted(problems):
        fig, ax = plt.subplots(figsize=(7, 4))
        any_plotted = False

        for solver, color in SOLVER_COLORS.items():
            fpath = results_dir / f"{solver}_{prob}_conv.csv"
            if not fpath.exists():
                continue
            try:
                df = pd.read_csv(fpath)
                if df.empty or "eval" not in df.columns:
                    continue
                evals = df["eval"].values
                costs = np.maximum(df["cost"].values, 1e-15)  # avoid log(0)
                label = SOLVER_LABELS.get(solver, solver)
                ax.semilogy(evals, costs, color=color, label=label, linewidth=1.8)
                any_plotted = True
            except Exception as e:
                print(f"  Warning: {fpath}: {e}")

        if not any_plotted:
            plt.close(fig)
            continue

        ax.set_xlabel("Function Evaluations")
        ax.set_ylabel("Best Cost (log scale)")
        ax.set_title(f"Convergence — {prob}")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out_path = output_dir / f"convergence_{prob}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ── 2. Box-plots of final best cost ───────────────────────────────────────────

def plot_boxplots(results_dir: Path, output_dir: Path):
    """
    For each problem, draw a box-plot comparing final best cost distributions
    across solvers (aggregated over all seeds).
    """
    csv_path = results_dir / "benchmark_results.csv"
    if not csv_path.exists():
        print(f"benchmark_results.csv not found in {results_dir}")
        return

    df = pd.read_csv(csv_path)
    problems = sorted(df["problem"].unique())

    for prob in problems:
        sub = df[df["problem"] == prob]
        solvers = [s for s in SOLVER_COLORS if s in sub["solver"].values]
        if not solvers:
            continue

        fig, ax = plt.subplots(figsize=(5, 4))
        data   = [sub[sub["solver"] == s]["best_cost"].values for s in solvers]
        labels = [SOLVER_LABELS.get(s, s) for s in solvers]
        colors = [SOLVER_COLORS[s] for s in solvers]

        bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_yscale("log")
        ax.set_ylabel("Best Cost (log scale)")
        ax.set_title(f"Final Cost Distribution — {prob}")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        out_path = output_dir / f"boxplot_{prob}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ── 3. ECDF curves ────────────────────────────────────────────────────────────

def plot_ecdf(results_dir: Path, output_dir: Path, problem: str):
    """
    ECDF (empirical CDF) of final best cost per solver for a given problem.
    X-axis: best cost threshold  |  Y-axis: fraction of runs below threshold.
    """
    csv_path = results_dir / "benchmark_results.csv"
    if not csv_path.exists():
        print(f"benchmark_results.csv not found in {results_dir}")
        return

    df = pd.read_csv(csv_path)
    sub = df[df["problem"] == problem]
    if sub.empty:
        print(f"No results for problem '{problem}'")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for solver, color in SOLVER_COLORS.items():
        s_data = sub[sub["solver"] == solver]["best_cost"].values
        if len(s_data) == 0:
            continue
        sorted_costs = np.sort(s_data)
        ecdf = np.arange(1, len(sorted_costs) + 1) / len(sorted_costs)
        label = SOLVER_LABELS.get(solver, solver)
        ax.step(sorted_costs, ecdf, color=color, label=label, linewidth=1.8, where="post")

    ax.set_xscale("log")
    ax.set_xlabel("Best Cost Threshold (log scale)")
    ax.set_ylabel("Fraction of Runs")
    ax.set_title(f"ECDF — {problem}")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = output_dir / f"ecdf_{problem}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── 4. 2D landscape + particles ───────────────────────────────────────────────

def plot_landscape_2d(problem_name: str, output_dir: Path,
                      x_range=(-5, 5), y_range=(-5, 5), resolution=200):
    """
    Plot the 2D landscape (contour) for a known benchmark function.
    """
    import importlib
    # Only works for known 2D functions; skip if unknown
    func_map = {
        "rastrigin": lambda x, y: (
            10 * 2 + (x**2 - 10 * np.cos(2 * np.pi * x))
                   + (y**2 - 10 * np.cos(2 * np.pi * y))
        ),
        "ackley": lambda x, y: (
            -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
            - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
            + 20 + np.e
        ),
        "griewank": lambda x, y: (
            1 + (x**2 + y**2) / 4000
            - np.cos(x) * np.cos(y / np.sqrt(2))
        ),
    }

    fname = problem_name.replace("_2d", "")
    if fname not in func_map:
        return

    xs = np.linspace(*x_range, resolution)
    ys = np.linspace(*y_range, resolution)
    X, Y = np.meshgrid(xs, ys)
    Z = func_map[fname](X, Y)

    fig, ax = plt.subplots(figsize=(5, 4))
    levels = np.percentile(Z, np.linspace(0, 95, 20))
    cf = ax.contourf(X, Y, Z, levels=levels, cmap="viridis", alpha=0.8)
    ax.contour(X, Y, Z, levels=levels, colors="white", linewidths=0.3, alpha=0.5)
    plt.colorbar(cf, ax=ax, shrink=0.85)
    ax.set_title(f"Landscape — {problem_name}")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    fig.tight_layout()

    out_path = output_dir / f"landscape_{problem_name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── 5. 2D search process animation ───────────────────────────────────────────

def _make_landscape_grid(problem_name: str, x_range, y_range, resolution=150):
    """Evaluate a known 2D benchmark on a grid; returns X, Y, Z meshgrids."""
    func_map = {
        "rastrigin": lambda x, y: (
            10 * 2 + (x**2 - 10 * np.cos(2 * np.pi * x))
                   + (y**2 - 10 * np.cos(2 * np.pi * y))
        ),
        "ackley": lambda x, y: (
            -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
            - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
            + 20 + np.e
        ),
        "griewank": lambda x, y: (
            1 + (x**2 + y**2) / 4000
            - np.cos(x) * np.cos(y / np.sqrt(2))
        ),
        "gauss_mix": lambda x, y: np.zeros_like(x),  # placeholder
    }
    fname = problem_name.replace("_2d", "")
    fn = func_map.get(fname)
    if fn is None:
        return None, None, None
    xs = np.linspace(*x_range, resolution)
    ys = np.linspace(*y_range, resolution)
    X, Y = np.meshgrid(xs, ys)
    Z = fn(X, Y)
    return X, Y, Z


def plot_animation(results_dir: Path, output_dir: Path, solver: str, problem: str,
                   x_range=(-5, 5), y_range=(-5, 5), max_frames: int = 60,
                   fps: int = 10):
    """
    Animate the search process for a 2D problem.

    Reads population CSV written by SavePopulationCSV (C++ side):
      gen,eval,particle_id,x0,x1

    Renders: contour background + particle scatter per generation.
    Saves a GIF to output_dir/animation_{solver}_{problem}.gif.
    Requires matplotlib with Pillow backend (pip install Pillow).
    """
    import matplotlib.animation as animation

    pop_path = results_dir / f"{solver}_{problem}_population.csv"
    if not pop_path.exists():
        print(f"  Population CSV not found: {pop_path}")
        print("  Run benchmark_runner with --record-population flag first.")
        return

    df = pd.read_csv(pop_path)
    if "x0" not in df.columns or "x1" not in df.columns:
        print(f"  Population CSV must have x0, x1 columns (2D only).")
        return

    gens = sorted(df["gen"].unique())
    # Subsample frames if too many
    if len(gens) > max_frames:
        step = len(gens) // max_frames
        gens = gens[::step]

    color = SOLVER_COLORS.get(solver, "#888888")

    # Build landscape grid
    X, Y, Z = _make_landscape_grid(problem, x_range, y_range)
    has_landscape = Z is not None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_map, ax_conv = axes

    # ── Landscape axis ──
    if has_landscape:
        levels = np.percentile(Z, np.linspace(0, 95, 25))
        ax_map.contourf(X, Y, Z, levels=levels, cmap="viridis", alpha=0.75)
        ax_map.contour(X, Y, Z, levels=levels, colors="white",
                       linewidths=0.3, alpha=0.4)
    ax_map.set_xlim(*x_range)
    ax_map.set_ylim(*y_range)
    ax_map.set_xlabel("x₀")
    ax_map.set_ylabel("x₁")

    scat = ax_map.scatter([], [], s=18, c=color, alpha=0.7, zorder=5)
    best_dot = ax_map.scatter([], [], s=80, c="white", marker="*", zorder=6)
    title = ax_map.set_title("")

    # ── Convergence axis ──
    conv_path = results_dir / f"{solver}_{problem}_conv.csv"
    conv_df = None
    if conv_path.exists():
        try:
            conv_df = pd.read_csv(conv_path)
        except Exception:
            pass

    if conv_df is not None and not conv_df.empty:
        evals_all = conv_df["eval"].values
        costs_all = np.maximum(conv_df["cost"].values, 1e-15)
        ax_conv.semilogy(evals_all, costs_all, color=color, linewidth=1.5, alpha=0.4)

    conv_line, = ax_conv.semilogy([], [], color=color, linewidth=2.0)
    ax_conv.set_xlabel("Function Evaluations")
    ax_conv.set_ylabel("Best Cost (log)")
    ax_conv.set_title("Convergence")
    ax_conv.grid(True, alpha=0.3)

    fig.suptitle(f"{SOLVER_LABELS.get(solver, solver)} on {problem}", fontsize=11)
    fig.tight_layout()

    def init_fn():
        scat.set_offsets(np.empty((0, 2)))
        best_dot.set_offsets(np.empty((0, 2)))
        conv_line.set_data([], [])
        return scat, best_dot, conv_line, title

    def update(frame_gen):
        sub = df[df["gen"] == frame_gen]
        pts = sub[["x0", "x1"]].values
        scat.set_offsets(pts)

        # Best particle (lowest cost if available)
        if "cost" in sub.columns:
            best_idx = sub["cost"].idxmin()
        else:
            best_idx = sub.index[0]
        bx = sub.loc[best_idx, "x0"]
        by = sub.loc[best_idx, "x1"]
        best_dot.set_offsets([[bx, by]])

        title.set_text(f"{SOLVER_LABELS.get(solver, solver)} — gen {frame_gen}")

        # Update convergence line up to this generation's eval count
        if conv_df is not None and not conv_df.empty:
            cur_eval = sub["eval"].iloc[0] if "eval" in sub.columns else 0
            mask = evals_all <= cur_eval
            if mask.any():
                conv_line.set_data(evals_all[mask], costs_all[mask])
                ax_conv.set_xlim(0, evals_all[-1])
                ax_conv.set_ylim(costs_all.min() * 0.5, costs_all.max() * 2)

        return scat, best_dot, conv_line, title

    ani = animation.FuncAnimation(fig, update, frames=gens,
                                   init_func=init_fn, blit=True,
                                   interval=1000 // fps)

    out_path = output_dir / f"animation_{solver}_{problem}.gif"
    try:
        ani.save(str(out_path), writer="pillow", fps=fps)
        print(f"  Saved: {out_path}")
    except Exception as e:
        print(f"  Could not save animation: {e}")
        print("  Install Pillow: pip install Pillow")
    finally:
        plt.close(fig)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark visualization")
    parser.add_argument("--results-dir", default="benchmark_results",
                        help="Directory with CSVs from benchmark_runner")
    parser.add_argument("--output-dir", default="plots",
                        help="Directory to save plot images")
    parser.add_argument("--mode", default="all",
                        choices=["all", "convergence", "boxplot", "ecdf",
                                 "landscape", "animation"],
                        help="Which plots to generate")
    parser.add_argument("--problem", default=None,
                        help="Filter to specific problem (for ecdf/landscape/animation mode)")
    parser.add_argument("--solver", default=None,
                        help="Solver name for animation mode (cmaes/xnes/svgd/mppi)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = args.mode

    if mode in ("all", "convergence"):
        print("Generating convergence curves...")
        plot_convergence(results_dir, output_dir, args.problem)

    if mode in ("all", "boxplot"):
        print("Generating box-plots...")
        plot_boxplots(results_dir, output_dir)

    if mode in ("all", "ecdf"):
        probs = [args.problem] if args.problem else ["rastrigin_2d", "rastrigin_10d",
                                                      "ackley_2d", "griewank_10d"]
        for prob in probs:
            print(f"Generating ECDF for {prob}...")
            plot_ecdf(results_dir, output_dir, prob)

    if mode in ("all", "landscape"):
        print("Generating 2D landscapes...")
        for prob in ["rastrigin_2d", "ackley_2d", "griewank_2d"]:
            plot_landscape_2d(prob, output_dir)

    if mode == "animation":
        # Animation mode: requires --solver and --problem
        solver = getattr(args, "solver", None) or "cmaes"
        problem = args.problem or "rastrigin_2d"
        print(f"Generating animation for {solver} on {problem}...")
        plot_animation(results_dir, output_dir, solver, problem)

    print("Done.")


if __name__ == "__main__":
    main()
