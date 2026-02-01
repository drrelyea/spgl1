# SPGL1 Benchmark Results

Generated: 2026-01-31 20:05:16

## Summary

- **Python faster**: 38/38 (100.0%)
- **Median speedup**: 8.55x
- **Mean speedup**: 33.88x
- **Range**: 2.51x - 558.66x

## Detailed Results

| Function | Size | n | Python (ms) | Octave (ms) | Speedup | Status |
|----------|------|---|-------------|-------------|---------|--------|
| compute_sqrt_vectors | small | 1,000 | 0.01 | 0.08 | 14.90x |   faster |
| compute_sqrt_vectors | medium | 10,000 | 0.03 | 0.80 | 30.62x |   faster |
| find_lambda_star | small | 1,000 | 0.02 | 0.18 | 9.32x |   faster |
| find_lambda_star | medium | 10,000 | 0.57 | 1.46 | 2.54x |   faster |
| lbfgs_bprod | small | 1,000 | 0.00 | 0.37 | 558.66x |   faster |
| lbfgs_bprod | medium | 10,000 | 0.00 | 0.51 | 322.41x |   faster |
| lbfgs_hprod | small | 1,000 | 0.06 | 0.70 | 12.61x |   faster |
| lbfgs_hprod | medium | 10,000 | 0.30 | 0.93 | 3.05x |   faster |
| lbfgs_init | small | 1,000 | 0.00 | 0.08 | 25.03x |   faster |
| lbfgs_init | medium | 10,000 | 0.01 | 0.14 | 10.75x |   faster |
| lbfgs_update | small | 1,000 | 0.01 | 0.17 | 12.98x |   faster |
| lbfgs_update | medium | 10,000 | 0.02 | 0.22 | 12.86x |   faster |
| norm_l12_dual | small | 1,000 | 0.01 | 0.05 | 7.68x |   faster |
| norm_l12_dual | medium | 10,000 | 0.02 | 0.09 | 5.21x |   faster |
| norm_l12_primal | small | 1,000 | 0.01 | 0.05 | 7.00x |   faster |
| norm_l12_primal | medium | 10,000 | 0.02 | 0.08 | 5.20x |   faster |
| norm_l12_project | small | 1,000 | 0.10 | 0.55 | 5.63x |   faster |
| norm_l12_project | medium | 10,000 | 0.16 | 1.67 | 10.74x |   faster |
| norm_l1_dual | small | 1,000 | 0.00 | 0.04 | 13.77x |   faster |
| norm_l1_dual | medium | 10,000 | 0.01 | 0.22 | 32.67x |   faster |
| norm_l1_primal | small | 1,000 | 0.00 | 0.02 | 6.57x |   faster |
| norm_l1_primal | medium | 10,000 | 0.01 | 0.03 | 3.94x |   faster |
| norm_l1_primal_weighted | small | 1,000 | 0.00 | 0.02 | 7.65x |   faster |
| norm_l1_primal_weighted | medium | 10,000 | 0.01 | 0.03 | 3.78x |   faster |
| norm_l1_project | small | 1,000 | 0.04 | 0.30 | 7.87x |   faster |
| norm_l1_project | medium | 10,000 | 0.68 | 2.85 | 4.20x |   faster |
| oneprojector | small | 1,000 | 0.06 | 0.35 | 6.29x |   faster |
| oneprojector | medium | 10,000 | 0.70 | 2.79 | 3.98x |   faster |
| oneprojector_d | small | 1,000 | 0.03 | 1.36 | 41.71x |   faster |
| oneprojector_d | medium | 10,000 | 0.66 | 12.49 | 18.99x |   faster |
| oneprojector_i | small | 1,000 | 0.03 | 0.32 | 9.99x |   faster |
| oneprojector_i | medium | 10,000 | 0.65 | 2.67 | 4.11x |   faster |
| oneprojector_weighted | small | 1,000 | 0.05 | 1.39 | 26.01x |   faster |
| oneprojector_weighted | medium | 10,000 | 0.76 | 12.79 | 16.91x |   faster |
| product_b_forward | small | 1,000 | 0.00 | 0.02 | 7.37x |   faster |
| product_b_forward | medium | 10,000 | 0.01 | 0.03 | 2.53x |   faster |
| product_b_transpose | small | 1,000 | 0.00 | 0.02 | 9.24x |   faster |
| product_b_transpose | medium | 10,000 | 0.01 | 0.03 | 2.51x |   faster |
