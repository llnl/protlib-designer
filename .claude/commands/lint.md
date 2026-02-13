---
description: Run black and flake8, fixing lint/format issues.
---

# Lint and Format

Run the project's linting and formatting checks. Fix any issues found.

## Steps

1. Run black formatter in check mode first to see what needs changing:
   ```bash
   black --check -S -t py39 protlib_designer scripts
   ```
2. If there are formatting issues, run black to auto-fix:
   ```bash
   black -S -t py39 protlib_designer scripts
   ```
3. Run flake8 linter:
   ```bash
   flake8 --ignore=E501,E203,W503 protlib_designer scripts
   ```
4. If flake8 reports issues, fix them in the source files and re-run until clean.
5. Report a summary of what was fixed.
