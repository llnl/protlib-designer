---
description: Review staged/unstaged changes for correctness and quality.
---

# Code Review

Review the current staged or unstaged changes for code quality, correctness, and adherence to project conventions.

## Steps

1. Check for changes to review:
   ```bash
   git diff
   git diff --staged
   ```
2. Review the changes for:
   - **Correctness**: Logic errors, edge cases, off-by-one errors
   - **Style**: Adherence to black formatting (`-S -t py39`) and flake8 rules (ignoring E501, E203, W503)
   - **Architecture**: Consistency with the project's pipeline pattern (DataLoader -> ILPGenerator -> Solver -> SolutionManager)
   - **Testing**: Whether new functionality has corresponding tests
   - **Security**: No hardcoded credentials, no injection vulnerabilities
3. Provide a structured review with:
   - A summary of the changes
   - Issues found (if any), categorized by severity
   - Suggestions for improvement
