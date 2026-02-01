# SPGL1 Benchmark Results

Generated: 2026-01-31 20:14:06

## Summary

- **Python faster**: 38/38 (100.0%)
- **Median speedup**: 8.18x
- **Mean speedup**: 31.34x
- **Range**: 2.32x - 515.67x

## Detailed Results

| Function | Size | n | Python (ms) | Octave (ms) | Speedup | Status |
|----------|------|---|-------------|-------------|---------|--------|
| compute_sqrt_vectors | small | 1,000 | 0.01 | 0.08 | 15.01x |   faster |
| compute_sqrt_vectors | medium | 10,000 | 0.03 | 0.89 | 30.76x |   faster |
| find_lambda_star | small | 1,000 | 0.03 | 0.17 | 6.70x |   faster |
| find_lambda_star | medium | 10,000 | 0.58 | 1.43 | 2.46x |   faster |
| lbfgs_bprod | small | 1,000 | 0.00 | 0.36 | 515.67x |   faster |
| lbfgs_bprod | medium | 10,000 | 0.00 | 0.47 | 291.24x |   faster |
| lbfgs_hprod | small | 1,000 | 0.06 | 0.63 | 11.19x |   faster |
| lbfgs_hprod | medium | 10,000 | 0.33 | 0.97 | 2.94x |   faster |
| lbfgs_init | small | 1,000 | 0.00 | 0.09 | 25.76x |   faster |
| lbfgs_init | medium | 10,000 | 0.01 | 0.13 | 10.08x |   faster |
| lbfgs_update | small | 1,000 | 0.01 | 0.16 | 11.65x |   faster |
| lbfgs_update | medium | 10,000 | 0.02 | 0.22 | 12.65x |   faster |
| norm_l12_dual | small | 1,000 | 0.01 | 0.05 | 7.30x |   faster |
| norm_l12_dual | medium | 10,000 | 0.02 | 0.08 | 5.06x |   faster |
| norm_l12_primal | small | 1,000 | 0.01 | 0.05 | 7.12x |   faster |
| norm_l12_primal | medium | 10,000 | 0.02 | 0.08 | 5.09x |   faster |
| norm_l12_project | small | 1,000 | 0.09 | 0.53 | 5.63x |   faster |
| norm_l12_project | medium | 10,000 | 0.15 | 1.71 | 11.22x |   faster |
| norm_l1_dual | small | 1,000 | 0.00 | 0.03 | 12.04x |   faster |
| norm_l1_dual | medium | 10,000 | 0.01 | 0.22 | 33.01x |   faster |
| norm_l1_primal | small | 1,000 | 0.00 | 0.02 | 6.09x |   faster |
| norm_l1_primal | medium | 10,000 | 0.01 | 0.03 | 4.01x |   faster |
| norm_l1_primal_weighted | small | 1,000 | 0.00 | 0.02 | 6.51x |   faster |
| norm_l1_primal_weighted | medium | 10,000 | 0.01 | 0.03 | 3.66x |   faster |
| norm_l1_project | small | 1,000 | 0.04 | 0.31 | 7.83x |   faster |
| norm_l1_project | medium | 10,000 | 0.68 | 2.81 | 4.15x |   faster |
| oneprojector | small | 1,000 | 0.04 | 0.36 | 9.81x |   faster |
| oneprojector | medium | 10,000 | 0.69 | 2.76 | 4.00x |   faster |
| oneprojector_d | small | 1,000 | 0.04 | 1.23 | 29.03x |   faster |
| oneprojector_d | medium | 10,000 | 0.64 | 12.49 | 19.40x |   faster |
| oneprojector_i | small | 1,000 | 0.04 | 0.31 | 8.23x |   faster |
| oneprojector_i | medium | 10,000 | 0.64 | 2.73 | 4.25x |   faster |
| oneprojector_weighted | small | 1,000 | 0.06 | 1.32 | 22.09x |   faster |
| oneprojector_weighted | medium | 10,000 | 0.76 | 12.60 | 16.66x |   faster |
| product_b_forward | small | 1,000 | 0.00 | 0.02 | 8.12x |   faster |
| product_b_forward | medium | 10,000 | 0.01 | 0.03 | 2.32x |   faster |
| product_b_transpose | small | 1,000 | 0.00 | 0.02 | 9.42x |   faster |
| product_b_transpose | medium | 10,000 | 0.01 | 0.03 | 2.58x |   faster |
