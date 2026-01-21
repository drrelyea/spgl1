# MATLAB vs Python SPGL1 Comparison

## Purpose
This document systematically compares the MATLAB and Python implementations to understand:
1. What exists in each codebase
2. What differs between them
3. What is missing from Python

**Approach**: Careful, methodical comparison without making assumptions.

---

## File Inventory

### MATLAB Files (~/code/matlab_spgl/)
```
Total: 39 .m files + C code

Public (root directory):
- spgl1.m                    [main solver - 1243 lines]
- spg_bp.m                   [BP wrapper]
- spg_bpdn.m                 [BPDN wrapper]
- spg_lasso.m                [LASSO wrapper]
- spg_mmv.m                  [MMV solver]
- spg_group.m                [Group sparse solver]
- spgSetParms.m              [Parameter configuration]
- spgsetup.m                 [MEX compilation]
- spgdemo.m                  [Demo/examples]

Norm functions:
- NormL1.m, NormL1_primal.m, NormL1_dual.m, NormL1_project.m
- NormL12.m, NormL12_primal.m, NormL12_dual.m, NormL12_project.m
- NormL1NN_primal.m, NormL1NN_dual.m, NormL1NN_project.m
- NormGroupL2.m, NormGroupL2_primal.m, NormGroupL2_dual.m, NormGroupL2_project.m
- NormObj.m                  [Base class]
- findLambdaStar.m

Private directory:
- lsqr.m
- oneProjector.m
- oneProjectorMex.m
- findLambdaStar.m
- ensure.m
- lbfgs*.m (7 files)

C code:
- oneProjectorCore.c/h
- oneProjectorMex.c
- productBMex.c
- heap.c/h
```

### Python Files
```
Total: 3 core Python files

Core module (spgl1/):
- spgl1.py                   [main solver + norms - 1663 lines]
- lsqr.py                    [LSQR wrapper]
- __init__.py                [package init]

Tests:
- pytests/test_spgl1.py

Examples:
- tutorials/spgl1s.py
- tutorials/mmvnn.py
- examples/*.ipynb
```

---

## What to Compare

Let's go through each major component systematically:

### 1. Main Solver: spgl1()
- [ ] Function signature
- [ ] Parameter names and defaults
- [ ] Algorithm structure
- [ ] Loop logic
- [ ] Exit conditions

### 2. Wrapper Functions
- [ ] spg_bp
- [ ] spg_bpdn
- [ ] spg_lasso
- [ ] spg_mmv

### 3. Projection Functions
- [ ] oneProjector vs oneprojector

### 4. Norm Functions
- [ ] L1 (primal, dual, project)
- [ ] L12 (primal, dual, project)
- [ ] L1NN (primal, dual, project)
- [ ] L12NN (primal, dual, project)

### 5. Missing in Python
- [ ] spg_group
- [ ] NormGroupL2 functions
- [ ] L-BFGS functions
- [ ] spgSetParms

---

## Next Steps

1. Read MATLAB spgl1.m carefully to understand structure
2. Read Python spgl1.py carefully to understand structure
3. Compare side-by-side specific sections
4. Document actual differences (not assumptions)
5. Only then decide what needs fixing/adding

