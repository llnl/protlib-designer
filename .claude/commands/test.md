---
description: Run pytest suite or a targeted test.
---

# Run Tests

Run the project's test suite using pytest.

## Steps

1. Run the full test suite:
   ```bash
   pytest
   ```
2. If any tests fail, analyze the failures and report a summary including:
   - Which tests failed
   - The error messages
   - Suggested fixes
3. If the user provides a specific test name or pattern, run only that test:
   ```bash
   pytest tests/test_run_protlib_designer.py -k "<pattern>"
   ```
