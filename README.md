<div align="left">
  <h2>
    <picture>
    <source media="(prefers-color-scheme: dark)" srcset="images/protlib-designer-logo-name-dark.png" width="350">
    <source media="(prefers-color-scheme: light)" srcset="images/protlib-designer-logo-name-light.png" width="350">
    <img alt="protlib-designer" src="images/protlib-designer-logo-name-light.png" width="350">
    </picture>
  </h2>
</div>

![Status](https://img.shields.io/badge/Status-Active-green.svg)
![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
[![Paper](https://img.shields.io/badge/Paper-Download-green.svg)](https://www.biorxiv.org/content/10.1101/2024.11.03.621763v1)
![CI](https://github.com/LLNL/protlib-designer/actions/workflows/ci.yml/badge.svg)

## Introduction

Welcome to the `protlib-designer` repository! This repository contains a lightweight python library for designing diverse protein libraries by seeding linear programming with deep mutational scanning data (or any other data that can be represented as a matrix of scores per single-point mutation). The software takes as input the score matrix, where each row corresponds to a mutation and each column corresponds to a different source of scores, and outputs a subset of mutations that **Pareto-minimizes the scores from the different sources while maximizing the diversity of the library**.

The paper [Combinatorial Optimization of Antibody Libraries via Constrained Integer Programming](https://www.biorxiv.org/content/10.1101/2024.11.03.621763v3) uses this software to design diverse antibody libraries by seeding linear programming with scores computed by Protein Language Models (PLMs) and Inverse Folding models (IFold). Check out the paper for more details on the method and its applications. Visit the blog post [here](https://landajuela.github.io/protlib_designer/) for a high-level overview of the method.

<figure>
<img src="https://raw.githubusercontent.com/LLNL/protlib-designer/main/images/method_diagram.png" alt="protlib-designer method diagram" width="100%">
<figcaption>
<p class="figure-caption text-center">
<em> protlib-designer designs diverse protein libraries by seeding linear programming with deep mutational scanning data. (a) The input to the method is target protein sequence and, if available, a structure of the protein or protein complex (in this case, the antibody trastuzumab in complex with the HER2 receptor). (b) We generate in silico deep mutational scanning data using protein language and inverse folding models. (c) The result is fed into a multi-objective linear programming solver. (d) The solver generates a library of antibodies that are co-optimized for the in silico scores while satisfying diversity constraints. 
</em>
</p>
</figcaption>
</figure>

## Getting Started

In this section, we provide instructions on how to install the software and run the code.

### Installation

You can `pip` install the package from [PyPI](https://pypi.org/project/protlib-designer/):

```bash
pip install protlib-designer
```

> **Note:** You can install all additional dependencies by running `pip install protlib-designer[all]`. This will install all the dependencies for the scoring functions, including the inverse folding and protein language model dependencies.

Alternatively, you can clone the repository and install the package from source. First, clone the repository and create an environment with Python >=3.10,<3.11. Then,
you can do:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you want a development environment, you can install the development dependencies:

```bash
pip install -e .[dev]
```

which will allow you to run the tests and the linter. You can run the linting with:

```bash
black -S -t py39 protlib_designer scripts && \
flake8 --ignore=E501,E203,W503 protlib_designer scripts
```

### Run the code

To run the code to create a diverse protein library of size 10 from the example data, run the following command:

```bash
protlib-designer ./example_data/trastuzumab_spm.csv 10
```

We provide a rich set of command-line arguments to customize the behavior of `protlib-designer`. For example, the following command runs `protlib-designer` with a range of 3 to 5 mutations per sequence, enforcing the interleaving of the mutant order and balancing the mutant order, allowing for each mutation to appear at most `1` time and a position to be mutated at most `4` times,
and using a weighted multi-objective optimization:

```bash
protlib-designer ./example_data/trastuzumab_spm.csv 10 \
  --min-mut 3 \
  --max-mut 5 \
  --interleave-mutant-order True \
  --force-mutant-order-balance True \
  --schedule 2 \
  --schedule-param '1,4' \
  --weighted-multi-objective True
```

For more information on the command-line arguments, run:

```bash
protlib-designer --help
```

## Input data

The input to the software is a matrix of per-mutation scores (the csv file `trastuzumab_spm.csv` in the example above). Typically, the score matrix is defined by *in silico* deep mutational scanning data, where each row corresponds to a mutation and each column corresponds to the score computed by a deep learning model. See the example data in the `example_data` directory for an example of the input data format. The structure of the input data is shown below:

| Mutation | score-1 | score-2 | ... | score-N |
|------------|--------|--------|-----|--------|
| AH106C     | -0.1    | 0.2    | ... | 0.3    |
| AH106D     | 0.2    | -0.3    | ... | -0.4    |
| ...        | ...    | ...    | ... | ...    |
| YH107A     | -0.3    | 0.4    | ... | -0.5    |
| ...        | ...    | ...    | ... | ...    |

Important notes about the input data:

• The `Mutation` column contains the mutation in the format : `WT_residue` + `chain` + `position_index` + `mutant_residue`. For example, `A+H+106+C = AH106C` represents the mutation of the residue at position 106 in chain H from alanine to cysteine.

• The `score-1`, `score-2`, ..., `score-N` columns contain the scores computed by the deep learning models for each mutation. Typically, the scores are the negative log-likelihoods ratios of the mutant residue and the wild-type residue, computed by the deep learning model: 

```math
s_{ij}^{\text{PLM}} =  -\log \left( \frac{p(x_i = a_j | w)}{p(x_i = w_i | w)} \right) =  -\log(p(x_i = a_j | w)) + \log(p(x_i = w_i | w)),
```

where $w$ is the wild-type sequence, and $p(x_i = a_j | w)$ is the probability of the mutant residue $a_j$ at position $i$ given the wild-type sequence $w$ as estimated by a Protein Language Model (PLM) or an Inverse Folding model (or any other deep learning model). For example, in [Antibody Library Design by Seeding Linear Programming with Inverse Folding and Protein Language Models](https://www.biorxiv.org/content/10.1101/2024.11.03.621763v1), we used the scores computed by the [ProtBert](https://pubmed.ncbi.nlm.nih.gov/34232869/) and [AntiFold](https://arxiv.org/abs/2405.03370) models.

### Computing Input Data using Protein Language Models

We provide a set of scoring functions that can be used to compute the scores for the input data. The scoring functions are defined in the `protlib_designer/scorer` module. To use this functionality, you need to install additional dependencies:

```bash
pip install -e .[plm]
```

After installing the dependencies, you can use the scoring functions to compute the scores for the input data. For example, we can compute the scores using `Rostlab/prot_bert` and `facebook/esm2_t6_8M_UR50D` models, and then, call `protlib-designer` to design a diverse protein library of size 10:

```bash
protlib-plm-scorer \
  EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS \
  WB99 GB100 GB101 DB102 GB103 FB104 YB105 AB106 MB107 DB108 \
  --models Rostlab/prot_bert \
  --models facebook/esm2_t6_8M_UR50D
```

After that, you can run `protlib-designer` with the generated scores:

```bash
protlib-designer plm_scores.csv 10 --weighted-multi-objective True
```

### Computing Input Data with Inverse Folding Models

We provide built-in scoring functions to evaluate your input structures using inverse‐folding methods. Currently, we support:

- *Robust deep learning–based protein sequence design using ProteinMPNN* (Dauparas et al. 2022) - [Paper](https://www.science.org/doi/abs/10.1126/science.add2187) - We adopt some of the open source code from [ProteinMPNN](https://github.com/dauparas/ProteinMPNN).

To enable inverse-folding scoring, install the extra dependencies:

```bash
pip install -e .[ifold]
```

> **Note:** This will automatically download the default ProteinMPNN model weights. If you already have the weights locally, skip the download by passing `--model-path` to the scorer (see below).

Following the example in the previous section, you can compute the scores using the inverse-folding model:

```bash
protlib-ifold-scorer \
  example_data/1n8z.pdb \
  WB99 GB100 GB101 DB102 GB103 FB104 YB105 AB106 MB107 DB108 \
```

## Putting it all together: Protlib Designer Pipeline

We provide a command-line interface to run the entire pipeline, which includes computing the scores using Protein Language Models and Inverse Folding models, and then designing a diverse protein library using `protlib-designer`. This is done using the `protlib-pipeline` command.
You can run the pipeline with the following command:

```bash
protlib-pipeline \
  WB99 GB100 GB101 DB102 GB103 FB104 YB105 AB106 MB107 DB108 \
  --pdb-path ./example_data/1n8z.pdb \
  --plm-model-names facebook/esm2_t6_8M_UR50D
```

> **Note:** You can pass a placeholder position `*{chain}{{start-end}}` to the `protlib-pipeline` command to specify a range of positions to mutate. For example, `*B{99-108}` will have the same effect as `WB99 GB100 GB101 DB102 GB103 FB104 YB105 AB106 MB107 DB108`. You can also pass a placeholder of the form `*{chain}*` to mutate all positions in the chain. For example, `*B*` will mutate all positions in chain B.

## Contributing

Please read [CONTRIBUTING.md](./CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Citation

If you use this software in your research, please cite the following paper:

```bibtex
@article{Hayes2024.11.03.621763,
  author       = {Hayes, Conor F. and Goncalves, Andre R. and Magana-Zook, Steven and Pettit, Jacob and Solak, Ahmet Can and Faissol, Daniel and Landajuela, Mikel},
  title        = {Combinatorial Optimization of Antibody Libraries via Constrained Integer Programming},
  journal      = {bioRxiv},
  year         = {2024},
  elocation-id = {2024.11.03.621763},
  doi          = {10.1101/2024.11.03.621763},
  publisher    = {Cold Spring Harbor Laboratory},
  url          = {https://www.biorxiv.org/content/early/2025/12/31/2024.11.03.621763},
  eprint       = {https://www.biorxiv.org/content/early/2025/12/31/2024.11.03.621763.full.pdf}
}
```

## License

`protlib-designer` is released under an MIT license. For more details, please see the
[LICENSE](./LICENSE) and [RELEASE](./RELEASE) files. All new contributions must be made under the MIT license.

SPDX-License-Identifier: MIT

LLNL-CODE-2001645