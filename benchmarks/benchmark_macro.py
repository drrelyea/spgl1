"""Macro-benchmarks for full SPGL1 solver.

This module benchmarks complete solver runs (spg_bp, spg_bpdn, spg_lasso, etc.)
on various problem types and sizes, comparing Python against Octave.

Usage:
    python -m benchmarks.benchmark_macro [--problems bp,bpdn,lasso] [--sizes small,medium]

Example:
    python -m benchmarks.benchmark_macro --problems bp,bpdn --sizes small,medium
"""

import argparse
import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator

# Import SPGL1 solvers
from spgl1 import spgl1, spg_bp, spg_bpdn, spg_lasso, spg_mmv, spg_group


@dataclass
class MacroBenchmarkResult:
    """Result of a full solver benchmark."""
    problem_type: str
    size_category: str
    m: int
    n: int
    sparsity: float
    wall_time_ms: float
    num_iterations: int
    num_matvec_A: int
    num_matvec_At: int
    final_objective: float
    final_residual: float
    exit_flag: int
    extra_info: dict = field(default_factory=dict)


@dataclass
class ProblemConfig:
    """Problem configuration."""
    name: str
    m: int
    n: int
    sparsity: float
    noise_level: float = 0.01


# Standard problem sizes
PROBLEM_SIZES = {
    "tiny": ProblemConfig("tiny", 50, 100, 0.1, 0.01),
    "small": ProblemConfig("small", 500, 1000, 0.05, 0.01),
    "medium": ProblemConfig("medium", 2000, 5000, 0.02, 0.01),
    "large": ProblemConfig("large", 5000, 20000, 0.01, 0.01),
}


def generate_sparse_problem(
    cfg: ProblemConfig,
    seed: int = 42,
    complex_valued: bool = False,
) -> tuple[NDArray, NDArray, NDArray]:
    """Generate a sparse recovery problem.

    Parameters
    ----------
    cfg : ProblemConfig
        Problem configuration
    seed : int
        Random seed
    complex_valued : bool
        If True, generate complex-valued problem

    Returns
    -------
    A : ndarray (m, n)
        Measurement matrix
    b : ndarray (m,)
        Measurements
    x_true : ndarray (n,)
        True sparse signal
    """
    rng = np.random.default_rng(seed)

    m, n = cfg.m, cfg.n
    k = int(n * cfg.sparsity)

    # Generate random measurement matrix
    if complex_valued:
        A = (rng.standard_normal((m, n)) +
             1j * rng.standard_normal((m, n))) / np.sqrt(2)
    else:
        A = rng.standard_normal((m, n))

    # Normalize columns
    A = A / np.linalg.norm(A, axis=0)

    # Generate sparse signal
    if complex_valued:
        x_true = np.zeros(n, dtype=complex)
        x_true[:k] = (rng.standard_normal(k) +
                      1j * rng.standard_normal(k)) / np.sqrt(2)
    else:
        x_true = np.zeros(n)
        x_true[:k] = rng.standard_normal(k)

    # Shuffle support
    rng.shuffle(x_true)

    # Generate measurements with noise
    b = A @ x_true
    if complex_valued:
        noise = (rng.standard_normal(m) +
                 1j * rng.standard_normal(m)) / np.sqrt(2)
    else:
        noise = rng.standard_normal(m)
    b = b + cfg.noise_level * np.linalg.norm(b) * noise / np.linalg.norm(noise)

    return A, b, x_true


def generate_group_sparse_problem(
    cfg: ProblemConfig,
    num_groups: int = 10,
    seed: int = 42,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Generate a group-sparse recovery problem.

    Parameters
    ----------
    cfg : ProblemConfig
        Problem configuration
    num_groups : int
        Number of groups
    seed : int
        Random seed

    Returns
    -------
    A : ndarray (m, n)
        Measurement matrix
    b : ndarray (m,)
        Measurements
    x_true : ndarray (n,)
        True group-sparse signal
    groups : ndarray (num_groups, n)
        Group indicator matrix
    """
    rng = np.random.default_rng(seed)

    m, n = cfg.m, cfg.n

    # Ensure n is divisible by num_groups
    n = (n // num_groups) * num_groups
    group_size = n // num_groups

    # Generate random measurement matrix
    A = rng.standard_normal((m, n))
    A = A / np.linalg.norm(A, axis=0)

    # Generate group-sparse signal (activate ~20% of groups)
    x_true = np.zeros(n)
    active_groups = int(num_groups * 0.2)
    active_idx = rng.choice(num_groups, active_groups, replace=False)

    for g in active_idx:
        start = g * group_size
        end = (g + 1) * group_size
        x_true[start:end] = rng.standard_normal(group_size)

    # Generate measurements with noise
    b = A @ x_true
    noise = rng.standard_normal(m)
    b = b + cfg.noise_level * np.linalg.norm(b) * noise / np.linalg.norm(noise)

    # Create group indicator matrix (sparse)
    from scipy.sparse import lil_matrix
    groups = lil_matrix((num_groups, n))
    for g in range(num_groups):
        start = g * group_size
        end = (g + 1) * group_size
        groups[g, start:end] = 1
    groups = groups.tocsr()

    return A, b, x_true, groups


def generate_mmv_problem(
    cfg: ProblemConfig,
    num_vectors: int = 5,
    seed: int = 42,
) -> tuple[NDArray, NDArray, NDArray]:
    """Generate a multi-measurement vector problem.

    Parameters
    ----------
    cfg : ProblemConfig
        Problem configuration
    num_vectors : int
        Number of measurement vectors
    seed : int
        Random seed

    Returns
    -------
    A : ndarray (m, n)
        Measurement matrix
    B : ndarray (m, num_vectors)
        Measurements matrix
    X_true : ndarray (n, num_vectors)
        True row-sparse signal matrix
    """
    rng = np.random.default_rng(seed)

    m, n = cfg.m, cfg.n
    k = int(n * cfg.sparsity)
    g = num_vectors

    # Generate random measurement matrix
    A = rng.standard_normal((m, n))
    A = A / np.linalg.norm(A, axis=0)

    # Generate row-sparse signal (same support across columns)
    X_true = np.zeros((n, g))
    support = rng.choice(n, k, replace=False)
    X_true[support, :] = rng.standard_normal((k, g))

    # Generate measurements with noise
    B = A @ X_true
    noise = rng.standard_normal((m, g))
    B = B + cfg.noise_level * np.linalg.norm(B) * noise / np.linalg.norm(noise)

    return A, B, X_true


# Benchmark functions

def bench_bp(cfg: ProblemConfig, seed: int = 42) -> MacroBenchmarkResult:
    """Benchmark Basis Pursuit (spg_bp)."""
    A, b, x_true = generate_sparse_problem(cfg, seed)

    start = time.perf_counter()
    x, resid, grad, info = spg_bp(A, b, verbosity=0)
    elapsed = (time.perf_counter() - start) * 1000

    return MacroBenchmarkResult(
        problem_type="bp",
        size_category=cfg.name,
        m=cfg.m,
        n=cfg.n,
        sparsity=cfg.sparsity,
        wall_time_ms=elapsed,
        num_iterations=info["niters"],
        num_matvec_A=info["nprodA"],
        num_matvec_At=info["nprodAt"],
        final_objective=info["rnorm"],
        final_residual=float(np.linalg.norm(x - x_true)),
        exit_flag=info["stat"],
        extra_info={"tau": info["tau"]},
    )


def bench_bpdn(cfg: ProblemConfig, seed: int = 42) -> MacroBenchmarkResult:
    """Benchmark Basis Pursuit Denoise (spg_bpdn)."""
    A, b, x_true = generate_sparse_problem(cfg, seed)

    # Set sigma based on noise level
    sigma = cfg.noise_level * np.linalg.norm(b)

    start = time.perf_counter()
    x, resid, grad, info = spg_bpdn(A, b, sigma, verbosity=0)
    elapsed = (time.perf_counter() - start) * 1000

    return MacroBenchmarkResult(
        problem_type="bpdn",
        size_category=cfg.name,
        m=cfg.m,
        n=cfg.n,
        sparsity=cfg.sparsity,
        wall_time_ms=elapsed,
        num_iterations=info["niters"],
        num_matvec_A=info["nprodA"],
        num_matvec_At=info["nprodAt"],
        final_objective=info["rnorm"],
        final_residual=float(np.linalg.norm(x - x_true)),
        exit_flag=info["stat"],
        extra_info={"sigma": sigma, "tau": info["tau"]},
    )


def bench_lasso(cfg: ProblemConfig, seed: int = 42) -> MacroBenchmarkResult:
    """Benchmark LASSO (spg_lasso)."""
    A, b, x_true = generate_sparse_problem(cfg, seed)

    # Set tau based on true signal
    tau = np.linalg.norm(x_true, 1) * 1.2  # Slightly larger than true norm

    start = time.perf_counter()
    x, resid, grad, info = spg_lasso(A, b, tau, verbosity=0)
    elapsed = (time.perf_counter() - start) * 1000

    return MacroBenchmarkResult(
        problem_type="lasso",
        size_category=cfg.name,
        m=cfg.m,
        n=cfg.n,
        sparsity=cfg.sparsity,
        wall_time_ms=elapsed,
        num_iterations=info["niters"],
        num_matvec_A=info["nprodA"],
        num_matvec_At=info["nprodAt"],
        final_objective=info["rnorm"],
        final_residual=float(np.linalg.norm(x - x_true)),
        exit_flag=info["stat"],
        extra_info={"tau": tau},
    )


def bench_mmv(cfg: ProblemConfig, seed: int = 42) -> MacroBenchmarkResult:
    """Benchmark Multi-Measurement Vector (spg_mmv)."""
    A, B, X_true = generate_mmv_problem(cfg, num_vectors=5, seed=seed)

    # Set sigma based on noise level
    sigma = cfg.noise_level * np.linalg.norm(B, 'fro')

    start = time.perf_counter()
    X, resid, grad, info = spg_mmv(A, B, sigma, verbosity=0)
    elapsed = (time.perf_counter() - start) * 1000

    return MacroBenchmarkResult(
        problem_type="mmv",
        size_category=cfg.name,
        m=cfg.m,
        n=cfg.n,
        sparsity=cfg.sparsity,
        wall_time_ms=elapsed,
        num_iterations=info["niters"],
        num_matvec_A=info["nprodA"],
        num_matvec_At=info["nprodAt"],
        final_objective=info["rnorm"],
        final_residual=float(np.linalg.norm(X - X_true, 'fro')),
        exit_flag=info["stat"],
        extra_info={"sigma": sigma, "num_vectors": 5},
    )


def bench_group(cfg: ProblemConfig, seed: int = 42) -> MacroBenchmarkResult:
    """Benchmark Group Sparsity (spg_group)."""
    num_groups = 10
    A, b, x_true, groups = generate_group_sparse_problem(
        cfg, num_groups=num_groups, seed=seed
    )

    # Adjust n to match generated problem
    n = groups.shape[1]

    # Set sigma based on noise level
    sigma = cfg.noise_level * np.linalg.norm(b)

    start = time.perf_counter()
    x, resid, grad, info = spg_group(A, b, groups, sigma, verbosity=0)
    elapsed = (time.perf_counter() - start) * 1000

    return MacroBenchmarkResult(
        problem_type="group",
        size_category=cfg.name,
        m=cfg.m,
        n=n,
        sparsity=cfg.sparsity,
        wall_time_ms=elapsed,
        num_iterations=info["niters"],
        num_matvec_A=info["nprodA"],
        num_matvec_At=info["nprodAt"],
        final_objective=info["rnorm"],
        final_residual=float(np.linalg.norm(x - x_true)),
        exit_flag=info["stat"],
        extra_info={"sigma": sigma, "num_groups": num_groups},
    )


def bench_hybrid(cfg: ProblemConfig, seed: int = 42) -> MacroBenchmarkResult:
    """Benchmark solver with hybrid mode enabled."""
    A, b, x_true = generate_sparse_problem(cfg, seed)

    # Set sigma based on noise level
    sigma = cfg.noise_level * np.linalg.norm(b)

    start = time.perf_counter()
    x, resid, grad, info = spg_bpdn(A, b, sigma, verbosity=0, isHybrid=True)
    elapsed = (time.perf_counter() - start) * 1000

    return MacroBenchmarkResult(
        problem_type="hybrid",
        size_category=cfg.name,
        m=cfg.m,
        n=cfg.n,
        sparsity=cfg.sparsity,
        wall_time_ms=elapsed,
        num_iterations=info["niters"],
        num_matvec_A=info["nprodA"],
        num_matvec_At=info["nprodAt"],
        final_objective=info["rnorm"],
        final_residual=float(np.linalg.norm(x - x_true)),
        exit_flag=info["stat"],
        extra_info={"sigma": sigma, "hybrid": True},
    )


# Registry of all macro-benchmarks
BENCHMARKS = {
    "bp": bench_bp,
    "bpdn": bench_bpdn,
    "lasso": bench_lasso,
    "mmv": bench_mmv,
    "group": bench_group,
    "hybrid": bench_hybrid,
}


def run_all_benchmarks(
    problems: list[str] | None = None,
    sizes: list[str] | None = None,
    num_runs: int = 3,
    output_file: str | None = None,
) -> list[MacroBenchmarkResult]:
    """Run all specified macro-benchmarks.

    Parameters
    ----------
    problems : list of str, optional
        Problem types to benchmark. If None, run all.
    sizes : list of str, optional
        Problem sizes to test. If None, use ["small", "medium"].
    num_runs : int
        Number of runs per configuration (takes median)
    output_file : str, optional
        Path to save CSV results

    Returns
    -------
    results : list of MacroBenchmarkResult
        All benchmark results
    """
    if problems is None:
        problems = list(BENCHMARKS.keys())
    if sizes is None:
        sizes = ["small", "medium"]

    results = []

    for size_name in sizes:
        if size_name not in PROBLEM_SIZES:
            print(f"Warning: Unknown size '{size_name}', skipping")
            continue
        cfg = PROBLEM_SIZES[size_name]

        print(f"\n{'='*60}")
        print(f"Problem size: {cfg.name} (m={cfg.m}, n={cfg.n})")
        print(f"{'='*60}")

        for prob_name in problems:
            if prob_name not in BENCHMARKS:
                print(f"Warning: Unknown problem '{prob_name}', skipping")
                continue

            bench_func = BENCHMARKS[prob_name]

            # Run multiple times and take median
            run_results = []
            for run in range(num_runs):
                try:
                    result = bench_func(cfg, seed=42 + run)
                    run_results.append(result)
                except Exception as e:
                    print(f"  {prob_name:15s}: ERROR - {e}")
                    break

            if run_results:
                # Take result with median time
                times = [r.wall_time_ms for r in run_results]
                median_idx = np.argsort(times)[len(times) // 2]
                result = run_results[median_idx]
                results.append(result)

                print(f"  {prob_name:15s}: {result.wall_time_ms:10.1f} ms "
                      f"({result.num_iterations} iters, "
                      f"{result.num_matvec_A + result.num_matvec_At} matvecs)")

    # Save results to CSV if requested
    if output_file:
        save_results_csv(results, output_file)
        print(f"\nResults saved to {output_file}")

    return results


def save_results_csv(results: list[MacroBenchmarkResult], filepath: str) -> None:
    """Save macro-benchmark results to CSV file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "problem_type",
            "size_category",
            "m",
            "n",
            "sparsity",
            "wall_time_ms",
            "num_iterations",
            "num_matvec_A",
            "num_matvec_At",
            "final_objective",
            "final_residual",
            "exit_flag",
        ])
        for r in results:
            writer.writerow([
                r.problem_type,
                r.size_category,
                r.m,
                r.n,
                r.sparsity,
                f"{r.wall_time_ms:.2f}",
                r.num_iterations,
                r.num_matvec_A,
                r.num_matvec_At,
                f"{r.final_objective:.6f}",
                f"{r.final_residual:.6f}",
                r.exit_flag,
            ])


def print_summary(results: list[MacroBenchmarkResult]) -> None:
    """Print a summary table of macro-benchmark results."""
    print("\n" + "="*80)
    print("MACRO-BENCHMARK SUMMARY (Python)")
    print("="*80)

    # Group by problem type
    probs = sorted(set(r.problem_type for r in results))
    sizes = sorted(set(r.size_category for r in results),
                   key=lambda s: PROBLEM_SIZES.get(s, ProblemConfig(s, 0, 0, 0)).n)

    # Header
    header = f"{'Problem':15s}"
    for size in sizes:
        header += f" | {size:>15s}"
    print(header)
    print("-" * len(header))

    # Data rows
    for prob in probs:
        row = f"{prob:15s}"
        for size in sizes:
            matching = [r for r in results
                        if r.problem_type == prob and r.size_category == size]
            if matching:
                row += f" | {matching[0].wall_time_ms:12.1f} ms"
            else:
                row += f" | {'N/A':>15s}"
        print(row)


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Run SPGL1 macro-benchmarks (Python)"
    )
    parser.add_argument(
        "--problems", "-p",
        type=str,
        default=None,
        help="Comma-separated list of problems: bp,bpdn,lasso,mmv,group,hybrid"
    )
    parser.add_argument(
        "--sizes", "-s",
        type=str,
        default="small,medium",
        help="Comma-separated list of sizes: tiny,small,medium,large"
    )
    parser.add_argument(
        "--runs", "-r",
        type=int,
        default=3,
        help="Number of runs per configuration (default: 3)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="benchmarks/results/macro_python.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available benchmarks and exit"
    )

    args = parser.parse_args()

    if args.list:
        print("Available problem types:")
        for name in sorted(BENCHMARKS.keys()):
            print(f"  - {name}")
        print("\nAvailable sizes:")
        for name, cfg in PROBLEM_SIZES.items():
            print(f"  - {name}: m={cfg.m}, n={cfg.n}")
        return

    problems = args.problems.split(",") if args.problems else None
    sizes = args.sizes.split(",")

    print("SPGL1 Macro-Benchmarks (Python)")
    print(f"Problems: {problems or 'all'}")
    print(f"Sizes: {sizes}")
    print(f"Runs: {args.runs}")

    results = run_all_benchmarks(
        problems=problems,
        sizes=sizes,
        num_runs=args.runs,
        output_file=args.output,
    )

    print_summary(results)


if __name__ == "__main__":
    main()
