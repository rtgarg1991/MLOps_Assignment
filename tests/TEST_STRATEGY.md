# Test Strategy Summary

## Objective
The objective of this test strategy is to ensure correctness, stability, and reproducibility of the
end-to-end MLOps pipeline, including data processing, feature engineering, model training,
and API inference.

The tests are designed to be:
- Deterministic
- Cloud-independent
- CI/CD friendly
- Fast to execute

---

## Scope of Testing

### In Scope
- Data ingestion logic (mocked external data source)
- Data preprocessing and cleaning
- Feature engineering and encoding
- Model training logic
- API health and prediction endpoints


## Types of Tests Implemented

### 1. Unit Tests
Validate individual functions in isolation:
- `clean_data`
- `one_hot_encode_features`
- `train_model`
- `ingest_data`

External dependencies (UCI repo, GCP SDKs) are mocked.

---

### 2. API Contract Tests
Validate FastAPI behavior:
- `/health` endpoint availability
- `/predict` request/response structure
- Input validation errors

A dummy model is used when the real model artifact is unavailable.

---

## Mocking Strategy

To ensure tests run without cloud credentials:
- `google.cloud.storage` is mocked
- `google.cloud.bigquery` is mocked
- `ucimlrepo.fetch_ucirepo` is mocked

This allows tests to run in:
- Local environments
- GitHub Actions
- CI/CD pipelines

---

## Test Data Strategy

- Small synthetic DataFrames are used
- Balanced class distributions are ensured where stratified splitting is required
- No production data is used in tests

---

## Expected Outcomes

- All tests pass with a clean environment
- Failures indicate real regressions in logic or contracts
- CI pipeline fails fast on breaking changes

---

## Tools Used

- **pytest** – test runner
- **unittest.mock** – mocking external dependencies
- **FastAPI TestClient** – API testing

---

## Conclusion

This testing strategy ensures that the core ML pipeline and API are production-ready, reproducible,
and safe to deploy through automated CI/CD pipelines without requiring cloud dependencies.
