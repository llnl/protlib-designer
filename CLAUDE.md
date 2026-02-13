# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

protlib-designer is a Python library for designing diverse protein libraries using Integer Linear Programming (ILP) seeded with deep mutational scanning data. It Pareto-optimizes multiple scoring objectives while enforcing diversity constraints. Primary application: antibody library design.

## Build & Development Commands

```bash
# Install for development
pip install -e .[dev]

# Install with all optional deps (PLM scoring, inverse folding, LLM reasoning)
pip install -e .[all]

# Lint
flake8 --ignore=E501,E203,W503 protlib_designer scripts

# Format (check only)
black --check -S -t py39 protlib_designer scripts

# Format (auto-fix)
black -S -t py39 protlib_designer scripts

# Run all tests
pytest

# Run a single test by name
pytest tests/test_run_protlib_designer.py -k "test_run_protlib_designer[args0]"
```

## Architecture

The core data flow is a pipeline:

```
CSV (score matrix) → DataLoader → ILPGenerator → GenerateAndRemoveSolver → SolutionManager → Output CSV
```

**Key modules:**

- `protlib_designer/dataloader.py` — Loads mutation score CSVs. Mutation format: `{WT}{Chain}{Position}{AA}` (e.g., `AH106C` = Ala→Cys at position 106, chain H).
- `protlib_designer/generator/ilp_generator.py` — Core ILP solver (~850 LOC). Creates binary decision variables per mutation, builds multi-objective optimization with PuLP (CBC MILP solver). Handles constraints: forbidden AAs, aromatic limits, scheduling, dissimilarity-based diversity, objective bounds.
- `protlib_designer/solver/generate_and_remove_solver.py` — Iterative loop: solve → collect solution → remove from feasible set → repeat until library size reached.
- `protlib_designer/solution_manager.py` — Collects solutions, writes output CSV.
- `protlib_designer/scorer/` — Abstract `Scorer` base class with two implementations:
  - `plm_scorer.py` — Protein Language Model scoring via HuggingFace transformers (ProtBERT, ESM2)
  - `ifold_scorer.py` — Inverse folding scoring via ProteinMPNN
- `protlib_designer/structure/` — `contact_graph.py` (antibody-antigen contact detection), `interface_profile.py` (H-bonds, salt bridges analysis)
- `protlib_designer/llm_reasoning.py` — Optional LLM-based guidance (via litellm) that generates structured JSON constraints for the ILP optimizer.

**Design patterns:** Abstract base classes for Generator, Filter, Scorer (strategy pattern). CLI entry points in `scripts/` use Click.

## CLI Entry Points (defined in setup.cfg)

- `protlib-designer` → `scripts/run_protlib_designer.py` (main optimizer)
- `protlib-plm-scorer` → `scripts/run_plm_scorer.py`
- `protlib-ifold-scorer` → `scripts/run_ifold_scorer.py`
- `protlib-pipeline` → `scripts/run_pipeline.py` (scoring + ILP end-to-end)
- `protlib-pipeline-llm` → `scripts/run_pipeline_with_llm.py` (scoring + LLM reasoning + ILP)

## Code Style

- Formatter: black with `-S` (skip string normalization) and `-t py39` target
- Linter: flake8, ignoring E501 (line length), E203, W503
- Python 3.10+ required

## Testing

Tests use Click's `CliRunner` for end-to-end CLI testing with parametrized test cases. Output goes to `tmp_path` fixtures. There are currently 3 parametrized test cases covering basic, advanced constraint, and objective constraint configurations.
