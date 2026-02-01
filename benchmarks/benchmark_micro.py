"""Micro-benchmarks for individual SPGL1 functions.

This module benchmarks individual Python functions in isolation,
measuring execution time across various problem sizes.

Usage:
    python -m benchmarks.benchmark_micro [--functions FUNC1,FUNC2] [--sizes small,medium,large]

Example:
    python -m benchmarks.benchmark_micro --functions oneprojector,norm_l1_primal --sizes small,medium
"""

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import random as sparse_random

# Import SPGL1 functions to benchmark
from spgl1.spgl1 import (
    oneprojector,
    _oneprojector_i,
    _oneprojector_d,
    _norm_l1_primal,
    _norm_l1_dual,
    _norm_l1_project,
    _norm_l12_primal,
    _norm_l12_dual,
    _norm_l12_project,
    _find_lambda_star,
)
from spgl1.lbfgs import (
    lbfgs_init,
    lbfgs_update,
    lbfgs_hprod,
    lbfgs_bprod,
)
from spgl1.productB import (
    product_b,
    compute_sqrt_vectors,
)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    function_name: str
    size_category: str
    n: int
    median_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    num_runs: int


@dataclass
class ProblemSize:
    """Problem size configuration."""
    name: str
    n: int  # Number of variables
    m: int  # Number of measurements (for solver benchmarks)
    sparsity: float  # Fraction of non-zeros


# Standard problem sizes
PROBLEM_SIZES = {
    "tiny": ProblemSize("tiny", 100, 50, 0.1),
    "small": ProblemSize("small", 1_000, 500, 0.05),
    "medium": ProblemSize("medium", 10_000, 5_000, 0.05),
    "large": ProblemSize("large", 100_000, 50_000, 0.01),
    "very_large": ProblemSize("very_large", 1_000_000, 100_000, 0.001),
}


def time_function(
    func: Callable,
    args: tuple,
    num_warmup: int = 3,
    num_runs: int = 10,
) -> BenchmarkResult:
    """Time a function with warmup and multiple runs.

    Parameters
    ----------
    func : callable
        Function to benchmark
    args : tuple
        Arguments to pass to the function
    num_warmup : int
        Number of warmup runs (not timed)
    num_runs : int
        Number of timed runs

    Returns
    -------
    times : list of float
        Execution times in milliseconds
    """
    # Warmup runs
    for _ in range(num_warmup):
        func(*args)

    # Timed runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return times


def generate_test_data(size: ProblemSize, seed: int = 42) -> dict:
    """Generate test data for benchmarks.

    Parameters
    ----------
    size : ProblemSize
        Problem size configuration
    seed : int
        Random seed for reproducibility

    Returns
    -------
    data : dict
        Dictionary containing test vectors and matrices
    """
    rng = np.random.default_rng(seed)
    n = size.n
    m = size.m

    # Basic vectors
    x = rng.standard_normal(n)
    b = rng.standard_normal(n)
    d = np.abs(rng.standard_normal(n)) + 0.1  # Positive weights

    # Sparse signal
    x_sparse = np.zeros(n)
    nnz = int(n * size.sparsity)
    idx = rng.choice(n, nnz, replace=False)
    x_sparse[idx] = rng.standard_normal(nnz)

    # Tau values
    tau = np.linalg.norm(x_sparse, 1) * 0.5

    # For L-BFGS
    k = 8  # History size
    g1 = rng.standard_normal(n)
    g2 = g1 + 0.1 * rng.standard_normal(n)  # Slightly perturbed gradient
    p = -g1  # Search direction
    step = 0.1

    # For productB
    sqrt1, sqrt2 = compute_sqrt_vectors(n)

    # For group norms (g groups of size m/g each, assuming n divisible)
    g_groups = min(10, n // 10)  # Number of groups

    # For find_lambda_star
    z = np.abs(rng.standard_normal(n))
    w = np.ones(n)
    mu = 0.1

    return {
        "n": n,
        "m": m,
        "x": x,
        "b": b,
        "d": d,
        "x_sparse": x_sparse,
        "tau": tau,
        "k": k,
        "g1": g1,
        "g2": g2,
        "p": p,
        "step": step,
        "sqrt1": sqrt1,
        "sqrt2": sqrt2,
        "g_groups": g_groups,
        "z": z,
        "w": w,
        "mu": mu,
        "weights": 1.0,  # Scalar weight
        "weights_vec": d,  # Vector weights
    }


# Benchmark function definitions
# Each returns (function, args_generator)

def bench_oneprojector(data: dict) -> tuple:
    """Benchmark oneprojector with scalar weights."""
    return (oneprojector, (data["b"], 1.0, data["tau"]))


def bench_oneprojector_weighted(data: dict) -> tuple:
    """Benchmark oneprojector with vector weights."""
    return (oneprojector, (data["b"], data["d"], data["tau"]))


def bench_oneprojector_i(data: dict) -> tuple:
    """Benchmark _oneprojector_i (internal, unweighted)."""
    return (_oneprojector_i, (np.abs(data["b"]), data["tau"]))


def bench_oneprojector_d(data: dict) -> tuple:
    """Benchmark _oneprojector_d (internal, weighted)."""
    return (_oneprojector_d, (np.abs(data["b"]), data["d"], data["tau"]))


def bench_norm_l1_primal(data: dict) -> tuple:
    """Benchmark L1 primal norm."""
    return (_norm_l1_primal, (data["x"], data["weights"]))


def bench_norm_l1_primal_weighted(data: dict) -> tuple:
    """Benchmark L1 primal norm with vector weights."""
    return (_norm_l1_primal, (data["x"], data["weights_vec"]))


def bench_norm_l1_dual(data: dict) -> tuple:
    """Benchmark L1 dual norm (L-infinity)."""
    return (_norm_l1_dual, (data["x"], data["weights"]))


def bench_norm_l1_project(data: dict) -> tuple:
    """Benchmark L1 projection."""
    return (_norm_l1_project, (data["x"], data["weights"], data["tau"]))


def bench_norm_l12_primal(data: dict) -> tuple:
    """Benchmark L12 group primal norm."""
    g = data["g_groups"]
    # Reshape x to be divisible by g
    n_adj = (data["n"] // g) * g
    x = data["x"][:n_adj]
    return (_norm_l12_primal, (g, x, 1.0))


def bench_norm_l12_dual(data: dict) -> tuple:
    """Benchmark L12 group dual norm."""
    g = data["g_groups"]
    n_adj = (data["n"] // g) * g
    x = data["x"][:n_adj]
    return (_norm_l12_dual, (g, x, 1.0))


def bench_norm_l12_project(data: dict) -> tuple:
    """Benchmark L12 group projection."""
    g = data["g_groups"]
    n_adj = (data["n"] // g) * g
    x = data["x"][:n_adj]
    tau = np.linalg.norm(x) * 0.5
    return (_norm_l12_project, (g, x, 1.0, tau))


def bench_find_lambda_star(data: dict) -> tuple:
    """Benchmark dual objective computation."""
    return (_find_lambda_star, (data["z"], data["w"], data["tau"], data["mu"]))


def bench_lbfgs_init(data: dict) -> tuple:
    """Benchmark L-BFGS initialization."""
    return (lbfgs_init, (data["n"], data["k"], 1.0))


def bench_lbfgs_hprod(data: dict) -> tuple:
    """Benchmark L-BFGS H*g product (two-loop recursion)."""
    # Need initialized state with some history
    H = lbfgs_init(data["n"], data["k"], 1.0)
    # Add a few updates to build history
    for i in range(min(3, data["k"])):
        g1 = data["g1"] + i * 0.01 * data["x"]
        g2 = g1 + 0.1 * data["x"]
        p = -g1
        lbfgs_update(H, data["step"], p, g1, g2)
    return (lbfgs_hprod, (H, data["g1"]))


def bench_lbfgs_bprod(data: dict) -> tuple:
    """Benchmark L-BFGS B*g product (compact representation)."""
    H = lbfgs_init(data["n"], data["k"], 1.0)
    for i in range(min(3, data["k"])):
        g1 = data["g1"] + i * 0.01 * data["x"]
        g2 = g1 + 0.1 * data["x"]
        p = -g1
        lbfgs_update(H, data["step"], p, g1, g2)
    return (lbfgs_bprod, (H, data["g1"]))


def bench_lbfgs_update(data: dict) -> tuple:
    """Benchmark L-BFGS update."""
    H = lbfgs_init(data["n"], data["k"], 1.0)
    return (lbfgs_update, (H, data["step"], data["p"], data["g1"], data["g2"]))


def bench_product_b_forward(data: dict) -> tuple:
    """Benchmark productB forward mode."""
    return (product_b, (data["x"], 0, data["sqrt1"], data["sqrt2"]))


def bench_product_b_transpose(data: dict) -> tuple:
    """Benchmark productB transpose mode."""
    # Need d+1 dimensional input for transpose
    x_ext = np.concatenate([[0.0], data["x"]])
    sqrt1, sqrt2 = compute_sqrt_vectors(data["n"])
    return (product_b, (x_ext, 1, sqrt1, sqrt2))


def bench_compute_sqrt_vectors(data: dict) -> tuple:
    """Benchmark sqrt vector computation."""
    return (compute_sqrt_vectors, (data["n"],))


# Registry of all benchmarks
BENCHMARKS = {
    # Projection functions (Priority 1)
    "oneprojector": bench_oneprojector,
    "oneprojector_weighted": bench_oneprojector_weighted,
    "oneprojector_i": bench_oneprojector_i,
    "oneprojector_d": bench_oneprojector_d,

    # Norm functions (Priority 1)
    "norm_l1_primal": bench_norm_l1_primal,
    "norm_l1_primal_weighted": bench_norm_l1_primal_weighted,
    "norm_l1_dual": bench_norm_l1_dual,
    "norm_l1_project": bench_norm_l1_project,

    # Group norms (Priority 2)
    "norm_l12_primal": bench_norm_l12_primal,
    "norm_l12_dual": bench_norm_l12_dual,
    "norm_l12_project": bench_norm_l12_project,

    # Dual objective (Priority 2)
    "find_lambda_star": bench_find_lambda_star,

    # L-BFGS (Priority 2)
    "lbfgs_init": bench_lbfgs_init,
    "lbfgs_hprod": bench_lbfgs_hprod,
    "lbfgs_bprod": bench_lbfgs_bprod,
    "lbfgs_update": bench_lbfgs_update,

    # ProductB (baseline - already JIT)
    "product_b_forward": bench_product_b_forward,
    "product_b_transpose": bench_product_b_transpose,
    "compute_sqrt_vectors": bench_compute_sqrt_vectors,
}


def run_benchmark(
    func_name: str,
    size: ProblemSize,
    num_warmup: int = 3,
    num_runs: int = 10,
) -> BenchmarkResult:
    """Run a single benchmark.

    Parameters
    ----------
    func_name : str
        Name of the benchmark function
    size : ProblemSize
        Problem size configuration
    num_warmup : int
        Number of warmup runs
    num_runs : int
        Number of timed runs

    Returns
    -------
    result : BenchmarkResult
        Benchmark result
    """
    if func_name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {func_name}")

    # Generate test data
    data = generate_test_data(size)

    # Get function and args
    bench_func = BENCHMARKS[func_name]
    func, args = bench_func(data)

    # Time the function
    times = time_function(func, args, num_warmup, num_runs)

    return BenchmarkResult(
        function_name=func_name,
        size_category=size.name,
        n=size.n,
        median_time_ms=float(np.median(times)),
        std_time_ms=float(np.std(times)),
        min_time_ms=float(np.min(times)),
        max_time_ms=float(np.max(times)),
        num_runs=num_runs,
    )


def run_all_benchmarks(
    functions: list[str] | None = None,
    sizes: list[str] | None = None,
    num_warmup: int = 3,
    num_runs: int = 10,
    output_file: str | None = None,
) -> list[BenchmarkResult]:
    """Run all specified benchmarks.

    Parameters
    ----------
    functions : list of str, optional
        Functions to benchmark. If None, run all.
    sizes : list of str, optional
        Problem sizes to test. If None, use ["small", "medium", "large"].
    num_warmup : int
        Number of warmup runs
    num_runs : int
        Number of timed runs
    output_file : str, optional
        Path to save CSV results

    Returns
    -------
    results : list of BenchmarkResult
        All benchmark results
    """
    if functions is None:
        functions = list(BENCHMARKS.keys())
    if sizes is None:
        sizes = ["small", "medium", "large"]

    results = []

    for size_name in sizes:
        if size_name not in PROBLEM_SIZES:
            print(f"Warning: Unknown size '{size_name}', skipping")
            continue
        size = PROBLEM_SIZES[size_name]

        print(f"\n{'='*60}")
        print(f"Problem size: {size.name} (n={size.n:,})")
        print(f"{'='*60}")

        for func_name in functions:
            if func_name not in BENCHMARKS:
                print(f"Warning: Unknown function '{func_name}', skipping")
                continue

            try:
                result = run_benchmark(func_name, size, num_warmup, num_runs)
                results.append(result)
                print(f"  {func_name:30s}: {result.median_time_ms:10.4f} ms "
                      f"(+/- {result.std_time_ms:.4f})")
            except Exception as e:
                print(f"  {func_name:30s}: ERROR - {e}")

    # Save results to CSV if requested
    if output_file:
        save_results_csv(results, output_file)
        print(f"\nResults saved to {output_file}")

    return results


def save_results_csv(results: list[BenchmarkResult], filepath: str) -> None:
    """Save benchmark results to CSV file.

    Parameters
    ----------
    results : list of BenchmarkResult
        Benchmark results
    filepath : str
        Output file path
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "function_name",
            "size_category",
            "n",
            "median_time_ms",
            "std_time_ms",
            "min_time_ms",
            "max_time_ms",
            "num_runs",
        ])
        for r in results:
            writer.writerow([
                r.function_name,
                r.size_category,
                r.n,
                f"{r.median_time_ms:.4f}",
                f"{r.std_time_ms:.4f}",
                f"{r.min_time_ms:.4f}",
                f"{r.max_time_ms:.4f}",
                r.num_runs,
            ])


def print_summary(results: list[BenchmarkResult]) -> None:
    """Print a summary table of benchmark results.

    Parameters
    ----------
    results : list of BenchmarkResult
        Benchmark results
    """
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY (Python)")
    print("="*80)

    # Group by function
    funcs = sorted(set(r.function_name for r in results))
    sizes = sorted(set(r.size_category for r in results),
                   key=lambda s: PROBLEM_SIZES.get(s, ProblemSize(s, 0, 0, 0)).n)

    # Header
    header = f"{'Function':30s}"
    for size in sizes:
        header += f" | {size:>12s}"
    print(header)
    print("-" * len(header))

    # Data rows
    for func in funcs:
        row = f"{func:30s}"
        for size in sizes:
            matching = [r for r in results
                        if r.function_name == func and r.size_category == size]
            if matching:
                row += f" | {matching[0].median_time_ms:10.4f}ms"
            else:
                row += f" | {'N/A':>12s}"
        print(row)


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Run SPGL1 micro-benchmarks (Python)"
    )
    parser.add_argument(
        "--functions", "-f",
        type=str,
        default=None,
        help="Comma-separated list of functions to benchmark (default: all)"
    )
    parser.add_argument(
        "--sizes", "-s",
        type=str,
        default="small,medium,large",
        help="Comma-separated list of sizes: tiny,small,medium,large,very_large"
    )
    parser.add_argument(
        "--runs", "-r",
        type=int,
        default=10,
        help="Number of timed runs per benchmark (default: 10)"
    )
    parser.add_argument(
        "--warmup", "-w",
        type=int,
        default=3,
        help="Number of warmup runs (default: 3)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="benchmarks/results/micro_python.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available benchmarks and exit"
    )

    args = parser.parse_args()

    if args.list:
        print("Available benchmarks:")
        for name in sorted(BENCHMARKS.keys()):
            print(f"  - {name}")
        print("\nAvailable sizes:")
        for name, size in PROBLEM_SIZES.items():
            print(f"  - {name}: n={size.n:,}")
        return

    functions = args.functions.split(",") if args.functions else None
    sizes = args.sizes.split(",")

    print("SPGL1 Micro-Benchmarks (Python)")
    print(f"Functions: {functions or 'all'}")
    print(f"Sizes: {sizes}")
    print(f"Runs: {args.runs}, Warmup: {args.warmup}")

    results = run_all_benchmarks(
        functions=functions,
        sizes=sizes,
        num_warmup=args.warmup,
        num_runs=args.runs,
        output_file=args.output,
    )

    print_summary(results)


if __name__ == "__main__":
    main()
