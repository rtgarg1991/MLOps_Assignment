# TESTING.md — Unit Test Documentation (MLOps Assignment)

This project uses **Pytest** to validate critical parts of the ML pipeline and API in a way that is:
- **Deterministic** (no dependence on live cloud services)
- **CI-friendly** (works on GitHub Actions runners without GCP credentials)
- **Fast** (runs in a few seconds)

All tests live in: `tests/`

---

## How to run tests

From the repo root:

```bash
source venv/bin/activate
pytest -v

