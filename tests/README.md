# Chess Elo Predictor — Unit Tests

Comprehensive unit tests for all 4 pipeline phases.

## Test Files

### 1. `test_data_loading.py` — Phase 1 Data Loading
Tests for data acquisition, parsing PGN/UCI files, and ECO database download.

**Test Classes:**
- `TestParsePGN`: PGN file parsing functionality
  - Returns DataFrame with correct structure
  - Proper data types (numeric ELO, string result/moves)
  - No unexpected missing values
  - Reproducibility: identical output for same input

- `TestParseUCI`: UCI file parsing
  - Correct column structure
  - Event ID tracking for joins
  - Reproducibility

- `TestECODatabase`: ECO opening database download
  - Network-aware tests (skip if unavailable)
  - Required columns present
  - File persistence

- `TestDataConsistency`: Cross-function consistency
  - Event IDs joinable across formats
  - ELO values in valid ranges (0-4000)
  - Result values valid chess outcomes (1-0, 0-1, 1/2-1/2)

**Key Metrics:**
- ✅ Reproducibility: Same input → Same output
- ✅ Data types: Numeric columns properly typed
- ✅ No missing data: Critical columns have no NaN
- ✅ Value ranges: ELO 0-4000, Results valid

---

### 2. `test_transform.py` — Phase 3 Transformation & Merge
Tests for ECO matching, Stockfish extraction, and join operations.

**Test Classes:**
- `TestECOMatching`: ECO opening code matching
  - Lookup dictionary structure
  - Match returns (code, name, family) tuple
  - Handles unmatched sequences
  - Reproducible matching

- `TestStockfishExtraction`: Stockfish feature extraction
  - DataFrame output format
  - Required columns (white_acl, black_acl)
  - Numeric data types

- `TestMergeOperations`: Join/merge operations
  - Preserves all records in inner join (when all present)
  - All source columns present in output
  - No unintended NaN introduction
  - No duplicate records created
  - Reproducible merges

- `TestFeatureValueRanges`: Feature value validation
  - ACL (accuracy) 0-100
  - Blunder counts non-negative integers

- `TestTransformConsistency`: Data integrity through transform
  - Event IDs preserved correctly
  - Original values unchanged (ELO, results)

**Key Metrics:**
- ✅ No data loss: All rows preserved in joins
- ✅ Reproducibility: Same merge → Same result
- ✅ Value ranges: ACL 0-100, no invalid values
- ✅ Consistency: Event IDs and original values preserved

---

### 3. `test_build_features.py` — Phase 4 Feature Engineering
Tests for feature engineering, target encoding, and dataset integration.

**Test Classes:**
- `TestEngineerFeatures`: Feature engineering pipeline
  - Elo bucket columns created
  - Valid bucket labels: Beginner, Intermediate, Advanced, Expert, Master
  - Ordinal encoding 0-4
  - Correct ELO → bucket classification
  - ACL gap calculated correctly (white_acl - black_acl)
  - Winner binary: 1=White, 0=Black (NaN for draws)
  - Winner multiclass: 0=Black, 1=Draw, 2=White
  - No unexpected NaN in critical features
  - Reproducibility

- `TestFeatureDataTypes`: Data type validation
  - Categorical columns are object dtype
  - Ordinal columns are numeric dtype

- `TestFeatureConsistency`: Feature integrity
  - No duplicate rows after engineering
  - Event IDs preserved
  - Original columns preserved

- `TestValidationReport`: Validation report generation
  - Runs without error
  - Handles missing columns gracefully

- `TestFeatureIntegration`: Modeling readiness
  - All required features present for models
  - No infinite values in numeric features

**Key Metrics:**
- ✅ Correct encoding: Elo buckets 0-4, Results 0/1/2
- ✅ No data loss: Rows preserved, duplicates detected
- ✅ Value ranges: ACL gap in valid range
- ✅ Reproducibility: Same input → Same engineered features
- ✅ Modeling ready: All required features, no Inf/NaN anomalies

---

### 4. `test_validate.py` — Phase 2 Data Validation
Tests for data quality checks and validation reporting.

**Test Classes:**
- `TestValidationFunctions`: Validation function execution
  - Error handling for missing files
  - Graceful degradation

- `TestValidationReport`: Report generation
  - Executes without error
  - Includes shape information
  - Includes data types section
  - Includes missing values detection
  - Includes duplicate detection

- `TestDataQualityChecks`: Specific anomaly detection
  - Negative ELO detection
  - Missing critical columns
  - Invalid game results
  - ACL out of range (0-100)

- `TestValidationReportStructure`: Report completeness
  - Multiple sections present
  - Summary statistics included

- `TestValidationConsistency`: Reproducibility
  - Same DataFrame → Same report

- `TestValidationEdgeCases`: Edge case handling
  - Empty DataFrames
  - Single row
  - All NaN columns
  - Single column

- `TestValidationAnomalies`: Anomaly detection
  - Outlier detection (IQR method)
  - Duplicate events
  - All identical values (data quality issues)

**Key Metrics:**
- ✅ Anomaly detection: Outliers, duplicates, invalid values
- ✅ Reproducibility: Same data → Same validation
- ✅ Robustness: Handles edge cases gracefully
- ✅ Completeness: Multiple validation checks

---

## Running Tests

### Run all tests:
```bash
make test
```

### Run specific test file:
```bash
poetry run pytest tests/test_data_loading.py -v
```

### Run specific test class:
```bash
poetry run pytest tests/test_build_features.py::TestEngineerFeatures -v
```

### Run specific test:
```bash
poetry run pytest tests/test_transform.py::TestMergeOperations::test_merge_preserves_all_records -v
```

### Run with coverage report:
```bash
poetry run pytest tests/ --cov=src --cov-report=html
```

### Run with detailed output:
```bash
poetry run pytest tests/ -vv --tb=long
```

---

## Test Levels (Level 2 Criteria)

All tests follow **Level 2** testing criteria:

### Custom Functions ✅
- `parse_pgn()`, `parse_uci()` — data parsing
- `build_eco_lookup()`, `match_eco()` — ECO matching
- `extract_stockfish_features()` — feature extraction
- `merge_datasets()` — join operations
- `engineer_features()` — feature engineering
- `validation_report()` — data quality reporting

### Data Shapes & Types ✅
- Input/output row/column counts
- Data type validation (numeric, string, categorical)
- Column presence and structure
- DataFrame consistency

### Performance Metrics ✅
- ELO value ranges (0-4000)
- ACL accuracy ranges (0-100)
- Blunder counts (non-negative)
- Result values (valid chess outcomes)

### Reproducibility ✅
- Same input → identical output
- No random seeds needed (deterministic operations)
- Consistency across multiple runs

### Value Ranges & Bounds ✅
- ELO bucket ordinal encoding (0-4)
- Winner encoding (0, 1, 2)
- ACL gap calculation
- Feature value limits

### Missing Data Detection ✅
- No unexpected NaN in critical columns
- Duplicate row detection
- Missing column detection

### Data Consistency ✅
- Event ID preservation through joins
- No data loss in merges
- Original values unchanged
- Joins produce expected counts

### Validation & Anomaly Detection ✅
- Outlier detection (IQR method)
- Invalid value detection
- Duplicate detection
- Missing values tracking

---

## Test Coverage

**Current Coverage:**
- Phase 1 (Data Loading): 15 tests
- Phase 3 (Transform): 18 tests
- Phase 4 (Features): 27 tests
- Phase 2 (Validation): 24 tests

**Total: 84 tests**

---

## Dependencies

Required packages (in `pyproject.toml`):
- pytest
- pandas
- numpy

Install with:
```bash
poetry install
```

---

## Contributing

When adding new tests:
1. Follow naming convention: `test_<functionality>.py`
2. Organize into test classes by feature area
3. Use descriptive test names: `test_<what_is_tested>_<expected_result>`
4. Document test purpose in docstring
5. Include both happy path and edge cases
6. Add to appropriate test file (data_loading, transform, build_features, validate)

---

## Troubleshooting

### Network-related test failures
Some tests that download data (ECO database) are skipped if network unavailable:
```
SKIPPED [1] tests/test_data_loading.py:123: Network error
```

### Missing module errors
Ensure `src/` directory is in Python path. Tests bootstrap this automatically.

### Assertion failures
Review test error messages for specific data type or value range mismatches.

---

## Next Steps

Additional test types to consider:
- **Level 3**: Integration tests (full pipeline)
- **Level 4**: Acceptance tests (with real Kaggle data)
- **Model tests**: Training, overfitting, performance (when models added)

