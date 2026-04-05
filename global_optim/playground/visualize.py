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
matplotlib.rcParams["savefig.dpi"] = 72        # lower DPI → faster HTML export
matplotlib.rcParams["figure.dpi"]  = 72
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

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, notch=False)
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
    levels = _safe_contour_levels(Z)
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

def _make_landscape_grid(problem_name: str, x_range, y_range, resolution=100):
    """Evaluate a known 2D benchmark on a grid; returns X, Y, Z meshgrids."""

    def gauss_mix_2d(x, y):
        # Matches C++ GaussianMixtureProblem(2, 3, spread=4.0):
        # k=0 → (0,0), k=1 → (0,-4), k=2 → (4,0), sigma=1, equal weights
        modes = [(0.0, 0.0), (0.0, -4.0), (4.0, 0.0)]
        w = 1.0 / 3
        mix = np.zeros_like(x, dtype=float)
        for mx, my in modes:
            d2 = (x - mx)**2 + (y - my)**2
            mix += w * np.exp(-0.5 * d2) / (2 * np.pi)
        return -np.log(mix + 1e-300)

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
        "dbl_rosen": lambda x, y: np.minimum(
            100 * ((-y - 10 - (-x - 10)**2))**2 + ((-x - 10) - 1)**2,
            5 + 100 * (((y - 10)/4 - ((x - 10)/4)**2))**2 + (((x - 10)/4) - 1)**2
        ),
        "gauss_mix": gauss_mix_2d,
        "schwefel": lambda x, y: (
            418.9829 * 2
            - x * np.sin(np.sqrt(np.abs(x)))
            - y * np.sin(np.sqrt(np.abs(y)))
        ),
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


def _safe_contour_levels(Z, n=20):
    """Return unique, strictly increasing contour levels from Z."""
    raw = np.percentile(Z[np.isfinite(Z)], np.linspace(2, 98, n))
    levels = np.unique(raw)
    if len(levels) < 2:
        zmin, zmax = np.nanmin(Z), np.nanmax(Z)
        levels = np.linspace(zmin, zmax + 1e-10, n)
    return levels


def plot_animation(results_dir: Path, output_dir: Path, solver: str, problem: str,
                   max_frames: int = 50, fps: int = 8):
    """
    Render the search process as an interactive HTML page with a slider.

    Reads population CSV (gen, eval, particle_id, x0, x1) produced by
    benchmark_runner --animate, then saves an HTML file you can open in
    any browser — use the slider or play button to step through generations.

    Fixes applied vs previous version:
      1. Output is HTML (to_jshtml) with built-in slider, not a GIF.
      2. Axis range is computed from all particle positions (dynamic bbox).
      3. Contour levels are deduplicated so constant / near-flat regions work.
      4. MPPI shows the IS-weighted centroid (red X) in addition to samples.
    """
    import matplotlib.animation as mpl_animation
    matplotlib.rcParams["animation.embed_limit"] = 64  # MB, allow larger HTML

    pop_path = results_dir / f"{solver}_{problem}_population.csv"
    if not pop_path.exists():
        print(f"  Population CSV not found: {pop_path}")
        print("  Run:  ./playground/benchmark_runner --animate "
              f"{solver} {problem}")
        return

    df = pd.read_csv(pop_path)
    if "x0" not in df.columns or "x1" not in df.columns:
        print("  Population CSV must have x0, x1 columns (2D problems only).")
        return

    gens = sorted(df["gen"].unique())
    if len(gens) > max_frames:
        step = max(1, len(gens) // max_frames)
        gens = gens[::step]

    color = SOLVER_COLORS.get(solver, "#888888")
    label = SOLVER_LABELS.get(solver, solver)

    # ── [FIX 2] Dynamic axis range from all particle positions ────────────────
    all_x = df["x0"].values
    all_y = df["x1"].values
    px = max(1.5, (all_x.max() - all_x.min()) * 0.15)
    py = max(1.5, (all_y.max() - all_y.min()) * 0.15)
    x_range = (all_x.min() - px, all_x.max() + px)
    y_range = (all_y.min() - py, all_y.max() + py)

    # ── Build landscape on the computed range ─────────────────────────────────
    X, Y, Z = _make_landscape_grid(problem, x_range, y_range)
    has_landscape = (Z is not None) and np.isfinite(Z).any()

    # ── Figure setup ──────────────────────────────────────────────────────────
    fig, (ax_map, ax_conv) = plt.subplots(1, 2, figsize=(12, 5))

    # ── [FIX 3] Safe contour drawing ──────────────────────────────────────────
    if has_landscape:
        levels = _safe_contour_levels(Z)
        ax_map.contourf(X, Y, Z, levels=levels, cmap="viridis", alpha=0.75)
        ax_map.contour(X, Y, Z, levels=levels, colors="white",
                       linewidths=0.3, alpha=0.4)

    ax_map.set_xlim(*x_range)
    ax_map.set_ylim(*y_range)
    ax_map.set_xlabel("x₀")
    ax_map.set_ylabel("x₁")

    # Particle scatter (subsample display max 60 pts per frame for clarity)
    MAX_SHOW = 60
    scat = ax_map.scatter([], [], s=20, c=color, alpha=0.65, zorder=5,
                          label="particles")
    # Best / representative point (white star)
    best_dot = ax_map.scatter([], [], s=150, c="white", marker="*", zorder=7,
                               edgecolors="black", linewidths=0.5,
                               label="best particle")
    # [FIX 4] MPPI: show IS-weighted centroid (red X = current mean estimate)
    is_mppi = (solver == "mppi")
    centroid_dot = ax_map.scatter([], [], s=120, c="red", marker="X",
                                   zorder=8, label="IS centroid") if is_mppi else None

    gen_text = ax_map.text(0.02, 0.97, "", transform=ax_map.transAxes,
                            va="top", fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.3",
                                      fc="white", alpha=0.7))
    ax_map.legend(loc="lower right", fontsize=8, framealpha=0.8)

    # ── Convergence axis ──────────────────────────────────────────────────────
    conv_path = results_dir / f"{solver}_{problem}_conv.csv"
    evals_all = costs_all = None
    if conv_path.exists():
        try:
            cdf = pd.read_csv(conv_path)
            if not cdf.empty and "eval" in cdf.columns:
                evals_all = cdf["eval"].values
                costs_all = np.maximum(cdf["cost"].values, 1e-15)
                ax_conv.semilogy(evals_all, costs_all, color=color,
                                  linewidth=1.5, alpha=0.3)
                ax_conv.set_xlim(0, evals_all[-1])
                ax_conv.set_ylim(costs_all.min() * 0.5, costs_all.max() * 2)
        except Exception:
            pass

    conv_line, = ax_conv.semilogy([], [], color=color, linewidth=2.5)
    vline = ax_conv.axvline(x=0, color="gray", linestyle="--",
                             linewidth=1.0, alpha=0.7)
    ax_conv.set_xlabel("Function Evaluations")
    ax_conv.set_ylabel("Best Cost (log scale)")
    ax_conv.set_title("Convergence")
    ax_conv.grid(True, alpha=0.3)

    fig.suptitle(f"{label} on {problem}", fontsize=12, fontweight="bold")
    fig.tight_layout()

    # ── Animation callbacks ───────────────────────────────────────────────────
    def update(frame_gen):
        sub = df[df["gen"] == frame_gen]

        # Subsample particles for display
        display = sub.sample(min(MAX_SHOW, len(sub)),
                              random_state=0) if len(sub) > MAX_SHOW else sub
        scat.set_offsets(display[["x0", "x1"]].values)

        # Best particle = row with smallest distance to origin as fallback
        if len(sub) > 0:
            # Use centroid as best approximation (no cost in population CSV)
            bx, by = sub["x0"].mean(), sub["x1"].mean()
            best_dot.set_offsets([[bx, by]])

        # MPPI centroid marker (same as best here, but styled differently)
        if is_mppi and centroid_dot is not None:
            centroid_dot.set_offsets([[sub["x0"].mean(), sub["x1"].mean()]])

        # Generation / eval label
        cur_eval = int(sub["eval"].iloc[0]) if "eval" in sub.columns else 0
        gen_text.set_text(f"gen {frame_gen}  |  evals {cur_eval}")

        # Convergence progress line
        if evals_all is not None:
            mask = evals_all <= cur_eval
            if mask.any():
                conv_line.set_data(evals_all[mask], costs_all[mask])
            vline.set_xdata([cur_eval, cur_eval])

    ani = mpl_animation.FuncAnimation(
        fig, update, frames=gens,
        interval=max(80, 1000 // fps), repeat=True)

    # ── [FIX 1] Save as interactive HTML with slider ──────────────────────────
    out_html = output_dir / f"animation_{solver}_{problem}.html"
    try:
        html_str = ani.to_jshtml(fps=fps, default_mode="loop")
        out_html.write_text(html_str)
        print(f"  Saved: {out_html}")
        print("  Open in browser — use the slider to step through generations.")
    except Exception as e:
        print(f"  HTML export failed ({e}), trying GIF fallback...")
        out_gif = output_dir / f"animation_{solver}_{problem}.gif"
        try:
            ani.save(str(out_gif), writer="pillow", fps=fps)
            print(f"  Saved (GIF fallback): {out_gif}")
        except Exception as e2:
            print(f"  GIF also failed: {e2}  (pip install Pillow)")
    finally:
        plt.close(fig)


# ── 6. Multi-algorithm comparison dashboard ───────────────────────────────────

def compare_dashboard(results_dir: Path, output_dir: Path, problem: str,
                      solvers: list = None):
    """
    Generate a 4-panel comparison dashboard for all solvers on one problem.

    Requires two CSV files in results_dir:
      benchmark_results.csv          — per-run final costs (multi-seed)
      compare_{problem}_conv.csv     — per-seed convergence curves (multi-seed)

    Both are produced by:
      benchmark_runner --problem <problem> --seeds N

    Panels:
      Top-left  : Convergence curves — median + IQR band, interpolated onto a
                  common eval grid.  Reveals speed of convergence.
      Top-right : ECDF of final best cost — shows reliability across seeds.
      Bottom-left: Box plots of final best cost (log scale).
      Bottom-right: Summary table — median / success-rate / median-evals.

    Output: compare_{problem}.html  (interactive, open in browser)
    """
    # ── Load data ──────────────────────────────────────────────────────────────
    results_csv = results_dir / "benchmark_results.csv"
    conv_csv    = results_dir / f"compare_{problem}_conv.csv"

    if not results_csv.exists():
        print(f"  benchmark_results.csv not found in {results_dir}.")
        print(f"  Run:  benchmark_runner --problem {problem} --seeds 20")
        return

    df_runs = pd.read_csv(results_csv)
    df_runs = df_runs[df_runs["problem"] == problem]
    if df_runs.empty:
        print(f"  No data for problem '{problem}' in benchmark_results.csv.")
        return

    has_conv = conv_csv.exists()
    df_conv  = pd.read_csv(conv_csv) if has_conv else None

    active_solvers = solvers or [s for s in SOLVER_COLORS
                                  if s in df_runs["solver"].unique()]
    if not active_solvers:
        print("  No solver data found.")
        return

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_conv, ax_ecdf, ax_box, ax_table = (axes[0, 0], axes[0, 1],
                                           axes[1, 0], axes[1, 1])
    fig.suptitle(f"Solver Comparison — {problem}", fontsize=13, fontweight="bold")

    # ── Panel 1: Convergence bands ────────────────────────────────────────────
    if has_conv and df_conv is not None:
        # Build a common eval grid (log-spaced, up to max eval seen)
        max_eval = df_conv["eval"].max()
        grid     = np.unique(np.round(
            np.geomspace(1, max_eval, 200)).astype(int))

        for solver in active_solvers:
            color = SOLVER_COLORS.get(solver, "#888888")
            label = SOLVER_LABELS.get(solver, solver)
            sub   = df_conv[df_conv["solver"] == solver]
            seeds = sub["seed"].unique()
            if len(seeds) == 0:
                continue

            # Interpolate each seed onto the common grid
            curves = []
            for seed in seeds:
                s = sub[sub["seed"] == seed].sort_values("eval")
                if s.empty:
                    continue
                # Step-interpolate: last known cost before each grid point
                interp = np.interp(grid, s["eval"].values,
                                   np.log10(np.maximum(s["cost"].values, 1e-15)))
                curves.append(interp)

            if not curves:
                continue
            curves = np.array(curves)          # (n_seeds, n_grid)
            med    = np.median(curves, axis=0)
            q25    = np.percentile(curves, 25, axis=0)
            q75    = np.percentile(curves, 75, axis=0)

            ax_conv.plot(grid, med, color=color, linewidth=2, label=label)
            ax_conv.fill_between(grid, q25, q75, color=color, alpha=0.18)

        ax_conv.set_xscale("log")
        ax_conv.set_xlabel("Function Evaluations (log)")
        ax_conv.set_ylabel("Best Cost log₁₀ (median ± IQR)")
        ax_conv.set_title("Convergence Curves")
        ax_conv.legend(fontsize=9)
        ax_conv.grid(True, alpha=0.3, which="both")
    else:
        ax_conv.text(0.5, 0.5,
                     "Run with --seeds N to get\nconvergence band data.",
                     ha="center", va="center", transform=ax_conv.transAxes,
                     fontsize=10, color="gray")
        ax_conv.set_title("Convergence Curves (no data)")

    # ── Panel 2: ECDF ─────────────────────────────────────────────────────────
    for solver in active_solvers:
        color  = SOLVER_COLORS.get(solver, "#888888")
        label  = SOLVER_LABELS.get(solver, solver)
        costs  = df_runs[df_runs["solver"] == solver]["best_cost"].dropna()
        if costs.empty:
            continue
        sc     = np.sort(costs.values)
        ecdf   = np.arange(1, len(sc) + 1) / len(sc)
        ax_ecdf.step(sc, ecdf, color=color, linewidth=2,
                     label=f"{label} (n={len(sc)})", where="post")

    ax_ecdf.set_xscale("log")
    ax_ecdf.set_xlabel("Best Cost Threshold (log)")
    ax_ecdf.set_ylabel("Fraction of Runs ≤ threshold")
    ax_ecdf.set_title("ECDF of Final Cost")
    ax_ecdf.legend(fontsize=9)
    ax_ecdf.grid(True, alpha=0.3)
    ax_ecdf.set_ylim(0, 1.05)

    # ── Panel 3: Box plots ────────────────────────────────────────────────────
    box_data   = []
    box_labels = []
    box_colors = []
    for solver in active_solvers:
        costs = df_runs[df_runs["solver"] == solver]["best_cost"].dropna()
        if costs.empty:
            continue
        box_data.append(np.maximum(costs.values, 1e-15))
        box_labels.append(SOLVER_LABELS.get(solver, solver))
        box_colors.append(SOLVER_COLORS.get(solver, "#888888"))

    if box_data:
        bp = ax_box.boxplot(box_data, tick_labels=box_labels, patch_artist=True,
                            notch=False, showfliers=True,
                            flierprops=dict(marker=".", markersize=4, alpha=0.5))
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        for median_line in bp["medians"]:
            median_line.set_color("white")
            median_line.set_linewidth(2)
    ax_box.set_yscale("log")
    ax_box.set_ylabel("Final Best Cost (log)")
    ax_box.set_title("Final Cost Distribution")
    ax_box.grid(True, axis="y", alpha=0.3)

    # ── Panel 4: Summary table ────────────────────────────────────────────────
    ax_table.axis("off")
    headers = ["Solver", "Median", "Q25", "Q75", "Success%", "Median Evals"]
    rows = []
    for solver in active_solvers:
        sub = df_runs[df_runs["solver"] == solver]
        if sub.empty:
            continue
        costs = np.sort(sub["best_cost"].dropna().values)
        evals = sub["num_evaluations"].dropna().values
        thresh = 1e-3
        sr = (costs <= thresh).mean() * 100
        med_ev = np.median(evals) if len(evals) else float("nan")

        def pct(v, p):
            return f"{np.percentile(v, p):.2e}" if len(v) else "—"

        rows.append([
            SOLVER_LABELS.get(solver, solver),
            pct(costs, 50), pct(costs, 25), pct(costs, 75),
            f"{sr:.0f}%",
            f"{med_ev:.0f}" if not np.isnan(med_ev) else "—",
        ])

    if rows:
        tbl = ax_table.table(
            cellText=rows, colLabels=headers,
            loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.1, 1.8)
        # Color header
        for j in range(len(headers)):
            tbl[0, j].set_facecolor("#2C3E50")
            tbl[0, j].set_text_props(color="white", fontweight="bold")
        # Color row backgrounds by solver
        for i, solver in enumerate(active_solvers):
            col = SOLVER_COLORS.get(solver, "#EEEEEE")
            for j in range(len(headers)):
                tbl[i + 1, j].set_facecolor(col + "33")  # 20% alpha hex
    ax_table.set_title("Summary Statistics  (success threshold = 1e-3)", pad=10)

    fig.tight_layout()

    # ── Save as interactive HTML ───────────────────────────────────────────────
    # The table panel is static; wrap the whole figure as a static PNG inside
    # an HTML page (no animation needed for compare dashboard).
    out_html = output_dir / f"compare_{problem}.html"
    out_png  = output_dir / f"compare_{problem}.png"
    fig.savefig(str(out_png), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Embed PNG in a minimal HTML with a clean background
    import base64
    png_b64 = base64.b64encode(out_png.read_bytes()).decode()
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Solver Comparison — {problem}</title>
<style>
  body {{ margin:0; background:#1a1a2e; display:flex;
          flex-direction:column; align-items:center; padding:20px; }}
  h2   {{ color:#eee; font-family:sans-serif; margin-bottom:10px; }}
  img  {{ max-width:100%; border-radius:8px;
          box-shadow:0 4px 20px rgba(0,0,0,0.5); }}
</style></head>
<body>
<h2>Solver Comparison — {problem}</h2>
<img src="data:image/png;base64,{png_b64}" alt="comparison dashboard">
</body></html>"""
    out_html.write_text(html)
    print(f"  Saved: {out_html}  (open in browser)")
    print(f"  PNG:   {out_png}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark visualization")
    parser.add_argument("--results-dir", default="benchmark_results",
                        help="Directory with CSVs from benchmark_runner")
    parser.add_argument("--output-dir", default="plots",
                        help="Directory to save plot images")
    parser.add_argument("--mode", default="all",
                        choices=["all", "convergence", "boxplot", "ecdf",
                                 "landscape", "animation", "compare"],
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
        solver = getattr(args, "solver", None) or "cmaes"
        problem = args.problem or "rastrigin_2d"
        print(f"Generating animation for {solver} on {problem}...")
        plot_animation(results_dir, output_dir, solver, problem)

    if mode == "compare":
        problem = args.problem or "rastrigin_2d"
        solvers = args.solver.split(",") if args.solver else None
        print(f"Generating comparison dashboard for {problem}...")
        compare_dashboard(results_dir, output_dir, problem, solvers)

    print("Done.")


if __name__ == "__main__":
    main()
