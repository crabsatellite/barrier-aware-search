# Barrier-Aware Search

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19489233.svg)](https://doi.org/10.5281/zenodo.19489233)

Replication code and experimental data for:

> **Cross-Dimensional Covariance Adaptation Induces Irreversible Drift Toward Deceptive Attractors in Barrier-Constrained Search**
>
> Alex Chengyu Li, 2026. DOI: [10.5281/zenodo.19489233](https://doi.org/10.5281/zenodo.19489233)

## Overview

This repository documents **covariance-induced irreversible drift**, a failure mode of CMA-ES in constrained optimization. On a barrier-constrained search problem with a non-smooth objective and binary feasibility constraints, full CMA-ES progressively amplifies redundant parameter dimensions through cross-dimensional covariance adaptation, trading constraint feasibility for marginal objective improvement.

Key findings (30-seed controlled comparison):
- Full CMA-ES drifts in **73%** of runs vs. 43% (sep-CMA-ES) and 16% (isotropic ES)
- Drift is **irreversible**: full CMA-ES escapes in only 1/10 trials; sep-CMA-ES in 5/5
- Drift separation is **monotonically non-decreasing** in population size, reaching complete separation (100% vs. 0%) at lambda=200

## Repository Structure

```
barrier-aware-search/
├── src/                    # Core modules
│   ├── search.py           # CMA-ES with barrier constraints
│   ├── measures.py         # Parameterized complexity measures
│   ├── dynamics.py         # Drift dynamics analysis
│   ├── mechanism.py        # Causal mechanism chain
│   ├── robustness.py       # Robustness checks
│   ├── barriers/           # Barrier constraint oracles
│   │   ├── natural_proof.py
│   │   ├── relativization.py
│   │   └── algebrization.py
│   └── ...
├── experiments/            # Experiment scripts
│   ├── exp_30seed.py       # 30-seed cross-optimizer comparison
│   ├── exp_ablation.py     # Structural condition ablations
│   ├── exp_lambda_escape.py # Population scaling + escape attempts
│   ├── exp_mitigation.py   # Adaptive mitigation experiment
│   └── ...
├── data/                   # Raw experimental results
│   └── jmlr_revision/      # 30-seed controlled comparison data (JSON)
├── figures/                # Generated visualizations (PDF)
├── logs/                   # Execution logs
├── checkpoints/            # CMA-ES state snapshots
├── configs/
│   └── default.json        # Default experiment configuration
└── paper/                  # LaTeX source and build script
    ├── covariance_drift_ecj.tex
    ├── references.bib
    └── build.bat
```

## Requirements

- Python 3.10+
- Dependencies: `pip install -r requirements.txt`

The experiments require precomputed Boolean function features (`data/search_features.npz`, included).

## Reproducing Results

### Full 30-seed replication

```bash
cd src
python -m experiments.run_all
```

This runs all experiments sequentially (cross-optimizer comparison, ablations, lambda scaling, escape attempts, adaptive mitigation). Total runtime is approximately 48 hours on an RTX 3090 + i7-12700K.

### Individual experiments

```bash
python -m experiments.exp_30seed          # Cross-optimizer comparison
python -m experiments.exp_ablation        # C1 (MSE) and C3 (soft penalty) ablations
python -m experiments.exp_lambda_escape   # Population scaling + escape attempts
python -m experiments.exp_mitigation      # Adaptive switching mitigation
```

### Verifying paper claims against raw data

```bash
cd paper
python verify_claims.py
python verify_final.py
```

Both scripts verify every numerical claim in the paper against the JSON result files and report any discrepancies.

### Building the paper

```bash
cd paper
build.bat
```

Produces `covariance_drift_ecj.pdf`.

## Experimental Domain

The search domain is Boolean function complexity measures that predict circuit size while satisfying three complexity-theoretic barrier constraints (Natural Proofs, Relativization, Algebrization). The domain combines:

1. A non-smooth, rank-based objective (Spearman correlation)
2. High-dimensional quadratic parameter interactions
3. Discontinuous binary constraint boundaries

## License

MIT License. See [LICENSE](LICENSE).

## Citation

```bibtex
@misc{li2026covariance,
  title={Cross-Dimensional Covariance Adaptation Induces Irreversible Drift
         Toward Deceptive Attractors in Barrier-Constrained Search},
  author={Li, Alex Chengyu},
  year={2026},
  doi={10.5281/zenodo.19489233},
  note={Preprint}
}
```
