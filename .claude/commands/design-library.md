---
description: Run protlib-designer with provided score matrix and args.
---

# Design Library

Run `protlib-designer` to design a diverse protein library from a score matrix CSV.

## Arguments

The user may provide:
- A path to the input CSV file (defaults to `./example_data/trastuzumab_spm.csv`)
- A library size (defaults to `10`)
- Any additional CLI flags

## Steps

1. Run protlib-designer with the provided arguments. If no arguments are given, run with the example data:
   ```bash
   protlib-designer ./example_data/trastuzumab_spm.csv 10
   ```
2. Report the output, including the number of sequences generated and the output location.
3. If the run fails, analyze the error and suggest fixes.
