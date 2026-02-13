---
description: Run protlib-pipeline end-to-end with provided args.
---

# Run Full Pipeline

Run the `protlib-pipeline` end-to-end: score mutations with PLM and/or inverse folding models, then design a diverse protein library with ILP.

## Arguments

The user may provide:
- Mutation positions (e.g., `WB99 GB100 GB101 DB102 GB103 FB104 YB105 AB106 MB107 DB108`)
- `--pdb-path` for the structure file
- `--plm-model-names` for PLM models
- Any additional CLI flags

## Steps

1. If the user provides arguments, pass them directly. Otherwise, run with example data:
   ```bash
   protlib-pipeline \
     WB99 GB100 GB101 DB102 GB103 FB104 YB105 AB106 MB107 DB108 \
     --pdb-path ./example_data/1n8z.pdb \
     --plm-model-names facebook/esm2_t6_8M_UR50D
   ```
2. Report the output, including scores computed and library generated.
3. If the run fails, analyze the error. Common issues include missing dependencies (`pip install -e .[all]`).
