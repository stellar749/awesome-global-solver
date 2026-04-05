#!/usr/bin/env python3
"""
run.py — 一键运行脚本，整合 C++ benchmark_runner 与 Python 可视化。

用法:
  python run.py animate <solver> <problem>        # 搜索过程动画
  python run.py compare <problem> [选项]           # 多算法对比仪表板
  python run.py benchmark [选项]                   # 完整 benchmark

示例:
  python run.py animate cmaes rastrigin_2d
  python run.py animate svgd  gauss_mix_2d
  python run.py compare rastrigin_2d
  python run.py compare gauss_mix_2d --solvers cmaes,xnes,svgd --seeds 30
  python run.py benchmark --seeds 51

可用 solver:   cmaes  xnes  svgd  mppi
可用 problem:  rastrigin_2d  rastrigin_10d  ackley_2d  ackley_10d
               schwefel_2d  dbl_rosen_2d  gauss_mix_2d
               griewank_2d  griewank_10d  random_basin_2d  random_basin_4d
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


# ── 找 benchmark_runner 二进制 ─────────────────────────────────────────────────

def find_binary() -> Path:
    """Search common build locations for benchmark_runner."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent / "build" / "playground" / "benchmark_runner",
        script_dir.parent.parent / "build" / "playground" / "benchmark_runner",
        Path("build") / "playground" / "benchmark_runner",
        Path("playground") / "benchmark_runner",
    ]
    for p in candidates:
        if p.exists():
            return p

    # Also try PATH
    import shutil
    found = shutil.which("benchmark_runner")
    if found:
        return Path(found)

    print("ERROR: benchmark_runner not found.")
    print("Build it first:")
    print("  cd global_optim && mkdir -p build && cd build")
    print("  cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)")
    sys.exit(1)


def find_visualize() -> Path:
    script_dir = Path(__file__).resolve().parent
    return script_dir / "visualize.py"


# ── 辅助：调用外部命令，实时打印输出 ──────────────────────────────────────────

def run_cmd(cmd: list, cwd: Path = None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    proc = subprocess.run(cmd, cwd=cwd)
    if proc.returncode != 0:
        print(f"Command failed (exit {proc.returncode})")
        sys.exit(proc.returncode)


# ── 模式：animate ──────────────────────────────────────────────────────────────

def cmd_animate(args):
    binary   = find_binary()
    viz      = find_visualize()
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: 生成数据
    run_cmd([binary,
             "--animate", args.solver, args.problem,
             "--output-dir", out_dir])

    # Step 2: 渲染 HTML
    run_cmd([sys.executable, viz,
             "--mode", "animation",
             "--solver", args.solver,
             "--problem", args.problem,
             "--results-dir", out_dir,
             "--output-dir", out_dir / "plots"])

    html = out_dir / "plots" / f"animation_{args.solver}_{args.problem}.html"
    print(f"\n✓ 完成！用浏览器打开查看：\n  {html.resolve()}")


# ── 模式：compare ──────────────────────────────────────────────────────────────

def cmd_compare(args):
    binary   = find_binary()
    viz      = find_visualize()
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: 运行多 seed benchmark（仅指定 problem）
    run_cmd([binary,
             "--problem", args.problem,
             "--seeds", str(args.seeds),
             "--output-dir", out_dir])

    # Step 2: 渲染对比仪表板
    viz_cmd = [sys.executable, viz,
               "--mode", "compare",
               "--problem", args.problem,
               "--results-dir", out_dir,
               "--output-dir", out_dir / "plots"]
    if args.solvers:
        viz_cmd += ["--solver", args.solvers]
    run_cmd(viz_cmd)

    html = out_dir / "plots" / f"compare_{args.problem}.html"
    print(f"\n✓ 完成！用浏览器打开查看：\n  {html.resolve()}")


# ── 模式：benchmark ────────────────────────────────────────────────────────────

def cmd_benchmark(args):
    binary  = find_binary()
    viz     = find_visualize()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 完整 benchmark
    run_cmd([binary,
             "--seeds", str(args.seeds),
             "--output-dir", out_dir])

    # 所有图表
    run_cmd([sys.executable, viz,
             "--mode", "all",
             "--results-dir", out_dir,
             "--output-dir", out_dir / "plots"])

    print(f"\n✓ 完成！结果保存在：{(out_dir / 'plots').resolve()}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="global_optim 一键运行脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    # ── animate ──
    p_anim = sub.add_parser("animate", help="搜索过程动画（交互式 HTML）")
    p_anim.add_argument("solver",  help="cmaes / xnes / svgd / mppi")
    p_anim.add_argument("problem", help="rastrigin_2d / gauss_mix_2d / ...")
    p_anim.add_argument("--output-dir", default="results", metavar="DIR")

    # ── compare ──
    p_cmp = sub.add_parser("compare", help="多算法对比仪表板")
    p_cmp.add_argument("problem", help="要对比的问题名称")
    p_cmp.add_argument("--solvers", default=None,
                        help="逗号分隔的算法列表，默认全部，如 cmaes,xnes,svgd")
    p_cmp.add_argument("--seeds",  type=int, default=20,
                        help="随机种子数（越多统计越稳定，默认 20）")
    p_cmp.add_argument("--output-dir", default="results", metavar="DIR")

    # ── benchmark ──
    p_bench = sub.add_parser("benchmark", help="完整 benchmark（所有算法 × 所有函数）")
    p_bench.add_argument("--seeds",  type=int, default=51)
    p_bench.add_argument("--output-dir", default="results", metavar="DIR")

    args = parser.parse_args()

    if args.mode == "animate":
        cmd_animate(args)
    elif args.mode == "compare":
        cmd_compare(args)
    elif args.mode == "benchmark":
        cmd_benchmark(args)


if __name__ == "__main__":
    main()
