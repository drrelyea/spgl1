#!/usr/bin/env python3
"""Main benchmark runner for SPGL1 Python vs Octave comparison.

This script orchestrates running both Python and Octave benchmarks,
then generates comparison reports and plots.

Usage:
    python -m benchmarks.run_benchmarks [--micro] [--macro] [--sizes SIZES]

Examples:
    # Run all benchmarks with default sizes
    python -m benchmarks.run_benchmarks

    # Run only micro-benchmarks on small problems
    python -m benchmarks.run_benchmarks --micro --sizes small

    # Run only macro-benchmarks
    python -m benchmarks.run_benchmarks --macro --sizes small,medium
"""

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.benchmark_micro import run_all_benchmarks as run_micro_python
from benchmarks.benchmark_micro import PROBLEM_SIZES as MICRO_SIZES
from benchmarks.benchmark_macro import run_all_benchmarks as run_macro_python
from benchmarks.benchmark_macro import PROBLEM_SIZES as MACRO_SIZES


def check_octave_available() -> bool:
    """Check if Octave is available."""
    try:
        result = subprocess.run(
            ["octave", "--version"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def run_octave_benchmarks(
    output_file: str,
    sizes: str,
    benchmark_dir: Path,
) -> bool:
    """Run Octave micro-benchmarks.

    Parameters
    ----------
    output_file : str
        Path for output CSV
    sizes : str
        Comma-separated size names
    benchmark_dir : Path
        Directory containing octave_benchmarks.m

    Returns
    -------
    success : bool
        True if benchmarks ran successfully
    """
    script_path = benchmark_dir / "octave_benchmarks.m"

    if not script_path.exists():
        print(f"Error: Octave script not found: {script_path}")
        return False

    print(f"\nRunning Octave benchmarks...")
    print(f"  Sizes: {sizes}")
    print(f"  Output: {output_file}")

    try:
        # Convert to absolute path for Octave
        abs_output = str(Path(output_file).resolve())

        # Run octave with the benchmark script
        result = subprocess.run(
            [
                "octave", "--quiet", "--eval",
                f"cd('{benchmark_dir}'); octave_benchmarks('{abs_output}', '{sizes}')"
            ],
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            cwd=benchmark_dir,
        )

        if result.returncode != 0:
            print(f"Octave error:\n{result.stderr}")
            return False

        print(result.stdout)
        return True

    except subprocess.TimeoutExpired:
        print("Error: Octave benchmarks timed out")
        return False
    except Exception as e:
        print(f"Error running Octave: {e}")
        return False


def load_csv_results(filepath: str) -> list[dict]:
    """Load benchmark results from CSV file."""
    results = []
    try:
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                for key in ["n", "m", "num_runs"]:
                    if key in row:
                        row[key] = int(row[key])
                for key in ["median_time_ms", "std_time_ms", "min_time_ms",
                            "max_time_ms", "wall_time_ms", "sparsity",
                            "final_objective", "final_residual"]:
                    if key in row:
                        row[key] = float(row[key])
                results.append(row)
    except FileNotFoundError:
        pass
    return results


def generate_comparison_report(
    python_results: list[dict],
    octave_results: list[dict],
    output_file: str,
) -> None:
    """Generate a comparison report between Python and Octave results.

    Parameters
    ----------
    python_results : list of dict
        Python benchmark results
    octave_results : list of dict
        Octave benchmark results
    output_file : str
        Path for output comparison CSV
    """
    # Create lookup for Octave results
    octave_lookup = {}
    for r in octave_results:
        key = (r.get("function_name") or r.get("problem_type"),
               r.get("size_category"))
        octave_lookup[key] = r

    comparisons = []
    for py in python_results:
        key = (py.get("function_name") or py.get("problem_type"),
               py.get("size_category"))

        oct = octave_lookup.get(key)
        if oct:
            py_time = py.get("median_time_ms") or py.get("wall_time_ms")
            oct_time = oct.get("median_time_ms") or oct.get("wall_time_ms")

            if py_time and oct_time:
                speedup = oct_time / py_time  # >1 means Python is faster
                comparisons.append({
                    "name": key[0],
                    "size": key[1],
                    "n": py.get("n"),
                    "python_ms": py_time,
                    "octave_ms": oct_time,
                    "speedup": speedup,
                    "python_faster": speedup > 1.0,
                    "diff_percent": (oct_time - py_time) / oct_time * 100,
                })

    # Save comparison results
    if comparisons:
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "name", "size", "n", "python_ms", "octave_ms",
                "speedup", "python_faster", "diff_percent"
            ])
            writer.writeheader()
            writer.writerows(comparisons)

        print(f"\nComparison saved to {output_file}")

    return comparisons


def print_comparison_summary(comparisons: list[dict]) -> None:
    """Print a summary of the comparison results."""
    if not comparisons:
        print("\nNo comparison data available")
        return

    print("\n" + "="*80)
    print("PYTHON vs OCTAVE COMPARISON")
    print("="*80)
    print(f"{'Function/Problem':<30} {'Size':<10} {'Python':>12} {'Octave':>12} {'Speedup':>10}")
    print("-"*80)

    for c in sorted(comparisons, key=lambda x: (x["name"], x.get("n", 0))):
        speedup_str = f"{c['speedup']:.2f}x"
        if c["python_faster"]:
            status = f"[green]{speedup_str}[/green]" if c["speedup"] > 1.1 else speedup_str
        else:
            status = f"[red]{speedup_str}[/red]" if c["speedup"] < 0.9 else speedup_str

        # Simple terminal coloring (ANSI codes)
        if c["python_faster"] and c["speedup"] > 1.1:
            speedup_str = f"\033[92m{speedup_str}\033[0m"  # Green
        elif not c["python_faster"] and c["speedup"] < 0.9:
            speedup_str = f"\033[91m{speedup_str}\033[0m"  # Red

        print(f"{c['name']:<30} {c['size']:<10} {c['python_ms']:>10.2f}ms "
              f"{c['octave_ms']:>10.2f}ms {speedup_str:>10}")

    # Summary statistics
    speedups = [c["speedup"] for c in comparisons]
    python_wins = sum(1 for c in comparisons if c["python_faster"])

    print("-"*80)
    print(f"Python faster: {python_wins}/{len(comparisons)} "
          f"({100*python_wins/len(comparisons):.1f}%)")
    print(f"Median speedup: {np.median(speedups):.2f}x")
    print(f"Mean speedup: {np.mean(speedups):.2f}x")
    print(f"Range: {min(speedups):.2f}x - {max(speedups):.2f}x")

    # Highlight slowest Python functions
    slow_funcs = [c for c in comparisons if c["speedup"] < 0.95]
    if slow_funcs:
        print("\n** Functions where Python is slower (needs optimization): **")
        for c in sorted(slow_funcs, key=lambda x: x["speedup"]):
            print(f"  - {c['name']} ({c['size']}): {c['speedup']:.2f}x "
                  f"({-c['diff_percent']:.1f}% slower)")


def generate_results_markdown(
    comparisons: list[dict],
    output_file: str,
) -> None:
    """Generate a markdown results file."""
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write("# SPGL1 Benchmark Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        if not comparisons:
            f.write("No benchmark data available.\n")
            return

        # Summary
        speedups = [c["speedup"] for c in comparisons]
        python_wins = sum(1 for c in comparisons if c["python_faster"])

        f.write("## Summary\n\n")
        f.write(f"- **Python faster**: {python_wins}/{len(comparisons)} "
                f"({100*python_wins/len(comparisons):.1f}%)\n")
        f.write(f"- **Median speedup**: {np.median(speedups):.2f}x\n")
        f.write(f"- **Mean speedup**: {np.mean(speedups):.2f}x\n")
        f.write(f"- **Range**: {min(speedups):.2f}x - {max(speedups):.2f}x\n\n")

        # Detailed table
        f.write("## Detailed Results\n\n")
        f.write("| Function | Size | n | Python (ms) | Octave (ms) | Speedup | Status |\n")
        f.write("|----------|------|---|-------------|-------------|---------|--------|\n")

        for c in sorted(comparisons, key=lambda x: (x["name"], x.get("n", 0))):
            status = "faster" if c["speedup"] > 1.05 else "slower" if c["speedup"] < 0.95 else "~equal"
            emoji = {"faster": " ", "slower": " ", "~equal": "="}[status]
            f.write(f"| {c['name']} | {c['size']} | {c.get('n', 'N/A'):,} | "
                    f"{c['python_ms']:.2f} | {c['octave_ms']:.2f} | "
                    f"{c['speedup']:.2f}x | {emoji} {status} |\n")

        # Optimization candidates
        slow_funcs = [c for c in comparisons if c["speedup"] < 0.95]
        if slow_funcs:
            f.write("\n## Optimization Candidates\n\n")
            f.write("Functions where Python is >5% slower than Octave:\n\n")
            for c in sorted(slow_funcs, key=lambda x: x["speedup"]):
                f.write(f"- **{c['name']}** ({c['size']}): "
                        f"{c['speedup']:.2f}x ({-c['diff_percent']:.1f}% slower)\n")

    print(f"Results markdown saved to {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run SPGL1 benchmarks (Python vs Octave)"
    )
    parser.add_argument(
        "--micro",
        action="store_true",
        help="Run micro-benchmarks only"
    )
    parser.add_argument(
        "--macro",
        action="store_true",
        help="Run macro-benchmarks only"
    )
    parser.add_argument(
        "--python-only",
        action="store_true",
        help="Run only Python benchmarks (skip Octave)"
    )
    parser.add_argument(
        "--octave-only",
        action="store_true",
        help="Run only Octave benchmarks (skip Python)"
    )
    parser.add_argument(
        "--sizes", "-s",
        type=str,
        default="small,medium",
        help="Comma-separated list of sizes to benchmark"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="benchmarks/results",
        help="Output directory for results"
    )

    args = parser.parse_args()

    # If neither micro nor macro specified, run both
    run_micro = args.micro or (not args.micro and not args.macro)
    run_macro = args.macro or (not args.micro and not args.macro)

    run_python = not args.octave_only
    run_octave = not args.python_only

    sizes = args.sizes.split(",")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_dir = Path(__file__).parent

    print("="*60)
    print("SPGL1 Benchmark Suite")
    print("="*60)
    print(f"Sizes: {sizes}")
    print(f"Micro-benchmarks: {run_micro}")
    print(f"Macro-benchmarks: {run_macro}")
    print(f"Python: {run_python}")
    print(f"Octave: {run_octave}")

    # Check Octave availability
    if run_octave and not check_octave_available():
        print("\nWarning: Octave not available, skipping Octave benchmarks")
        run_octave = False

    all_comparisons = []

    # --- Micro-benchmarks ---
    if run_micro:
        print("\n" + "="*60)
        print("MICRO-BENCHMARKS")
        print("="*60)

        # Python micro-benchmarks
        if run_python:
            print("\n--- Python Micro-Benchmarks ---")
            py_micro_file = str(output_dir / "micro_python.csv")
            run_micro_python(sizes=sizes, output_file=py_micro_file)

        # Octave micro-benchmarks
        if run_octave:
            print("\n--- Octave Micro-Benchmarks ---")
            oct_micro_file = str(output_dir / "micro_octave.csv")
            run_octave_benchmarks(oct_micro_file, args.sizes, benchmark_dir)

        # Compare
        if run_python and run_octave:
            py_results = load_csv_results(str(output_dir / "micro_python.csv"))
            oct_results = load_csv_results(str(output_dir / "micro_octave.csv"))
            comparisons = generate_comparison_report(
                py_results, oct_results,
                str(output_dir / "micro_comparison.csv")
            )
            all_comparisons.extend(comparisons)
            print_comparison_summary(comparisons)

    # --- Macro-benchmarks ---
    if run_macro:
        print("\n" + "="*60)
        print("MACRO-BENCHMARKS")
        print("="*60)

        # Python macro-benchmarks
        if run_python:
            print("\n--- Python Macro-Benchmarks ---")
            py_macro_file = str(output_dir / "macro_python.csv")
            run_macro_python(sizes=sizes, output_file=py_macro_file)

        # Note: Octave macro-benchmarks would need a separate script
        # For now, we only compare micro-benchmarks with Octave

    # Generate final report
    if all_comparisons:
        generate_results_markdown(
            all_comparisons,
            str(output_dir / "RESULTS.md")
        )

    print("\n" + "="*60)
    print("Benchmarks complete!")
    print(f"Results saved to {output_dir}/")
    print("="*60)


if __name__ == "__main__":
    main()
