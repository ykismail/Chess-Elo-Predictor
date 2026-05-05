# Test Commands Quick Reference

## Quick Start

```bash
# Run all tests with coverage
make test

# Run only integration tests
make test-integration

# Generate test and coverage reports
make test-report
```

## Detailed Commands

### Test Execution
```bash
# All tests with coverage
make test
# Output: Coverage HTML at reports/coverage/html/index.html

# Unit tests only (excludes integration)
make test-unit

# Integration tests only
make test-integration

# Coverage report generation
make test-coverage
```

### Report Generation
```bash
# Generate HTML + Markdown reports
make test-report
# Output: reports/test-results/
```

### Direct pytest Commands
```bash
# All tests with detailed output
poetry run pytest tests/ -v --tb=short

# Integration tests only with print statements visible
poetry run pytest tests/test_integration_pipeline.py -v -s

# Specific test class
poetry run pytest tests/test_integration_pipeline.py::TestParsingConsistency -v

# Specific test
poetry run pytest tests/test_integration_pipeline.py::TestParsingConsistency::test_pgn_parsing_produces_required_columns -v

# With coverage
poetry run pytest tests/ --cov=src --cov-report=html

# Stop on first failure
poetry run pytest tests/ -x

# Run last failed tests
poetry run pytest tests/ --lf

# Run failed tests first, then others
poetry run pytest tests/ --ff
```

## Understanding Test Output

### Success
```
tests/test_integration_pipeline.py::TestParsingConsistency::test_pgn_parsing_produces_required_columns PASSED
```

### Failure
```
tests/test_integration_pipeline.py::TestParsingConsistency::test_pgn_parsing_produces_required_columns FAILED

AssertionError: Missing critical column: white_elo
```

## Coverage Report Locations

After running tests:
- **Interactive HTML**: `reports/coverage/html/index.html` (open in browser)
- **Machine-readable XML**: `reports/coverage/coverage.xml`
- **Terminal output**: Shows coverage % and missing lines
- **Test results**: `reports/junit-results.xml` (JUnit format)

## Common Issues

### Import errors in tests
```bash
# Make sure you're in the right directory
cd "d:\as3b folder\last_term\Data science\chess project\Chess-Elo-Predictor"

# Install dependencies
poetry install
```

### Fixtures not found
```bash
# Run with current directory as root
pytest tests/ --rootdir=.
```

### Tests timeout
```bash
# Increase timeout (in seconds)
pytest tests/ --timeout=60
```

## Test File Organization

```
tests/
├── test_data_loading.py           # Unit tests for Phase 1
├── test_transform.py              # Unit tests for Phase 3
├── test_build_features.py         # Unit tests for Phase 5
├── test_validate.py               # Unit tests for validation
├── test_integration_pipeline.py   # NEW: Integration tests (28 tests)
├── INTEGRATION_TESTING_STRATEGY.md # Documentation
├── TEST_COMMANDS.md               # This file
└── __init__.py
```

## Test Classes & Coverage

| Test Suite | Count | Focus |
|-----------|-------|-------|
| `TestParsingConsistency` | 4 | Phase 1 parsing validation |
| `TestMergeIntegrity` | 3 | Phase 3 merge operations |
| `TestDataLeakagePrevention` | 2 | Critical: prevent target leakage |
| `TestPreprocessingPipeline` | 3 | Phase 4 preprocessing logic |
| `TestTrainTestSeparation` | 2 | Phase 4 dataset splitting |
| `TestFinalDatasetQuality` | 3 | Final dataset validation |
| `TestReproducibility` | 2 | Determinism verification |
| **Total** | **28** | **7 risk areas** |

## Expected Test Execution Time

- Unit tests only: ~2-5 seconds
- Unit + Integration: ~5-10 seconds
- With coverage: ~10-15 seconds
- Full report generation: ~20-30 seconds

## CI/CD Integration

Tests run automatically on push:
1. Unit tests on specific test files
2. Integration tests on all scenarios
3. Coverage report generated
4. Reports uploaded as artifacts

View results at: GitHub Actions > Test Reports artifact

---

For detailed test justification, see [INTEGRATION_TESTING_STRATEGY.md](INTEGRATION_TESTING_STRATEGY.md)
