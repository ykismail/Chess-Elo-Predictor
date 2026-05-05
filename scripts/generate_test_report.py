"""
Test Coverage and Results Report Generator
===========================================

Generates comprehensive testing reports including:
- Test execution results
- Code coverage metrics
- Critical failure point analysis
- Untested/undercovered code identification
- Integration test justification

Usage
-----
    python scripts/generate_test_report.py
"""

import os
import sys
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# ── Path bootstrap ────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, ".."))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)


class TestReportGenerator:
    """Generate comprehensive test execution and coverage reports."""

    def __init__(self, project_root: str = project_dir):
        self.project_root = Path(project_root)
        self.reports_dir = self.project_root / "reports"
        self.coverage_dir = self.reports_dir / "coverage"
        self.test_results_dir = self.reports_dir / "test-results"
        
        # Create directories
        for d in [self.reports_dir, self.coverage_dir, self.test_results_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def parse_coverage_xml(self) -> Dict:
        """
        Parse coverage.xml from pytest coverage report.
        
        Returns
        -------
        dict with coverage statistics:
            {
                "total_lines": int,
                "covered_lines": int,
                "coverage_percent": float,
                "packages": [...],
                "uncovered_critical": [...]
            }
        """
        coverage_xml = self.coverage_dir / "coverage.xml"
        
        if not coverage_xml.exists():
            return {"error": "coverage.xml not found"}
        
        try:
            tree = ET.parse(coverage_xml)
            root = tree.getroot()
            
            # Get overall statistics
            sources = root.findall(".//package")
            
            total_lines = int(root.get("lines-valid", 0))
            covered_lines = int(root.get("lines-covered", 0))
            coverage_pct = (covered_lines / total_lines * 100) if total_lines > 0 else 0
            
            # Identify critical uncovered modules
            critical_modules = ["data", "models", "features"]
            uncovered_critical = []
            
            for source in sources:
                for cls in source.findall(".//class"):
                    filename = cls.get("filename", "")
                    coverage = float(cls.get("complexity-coverage", 0)) if cls.get("complexity-coverage") else 0
                    
                    # Flag critical modules with low coverage
                    if any(m in filename for m in critical_modules) and coverage < 80:
                        uncovered_critical.append({
                            "file": filename,
                            "coverage": coverage
                        })
            
            return {
                "total_lines": total_lines,
                "covered_lines": covered_lines,
                "coverage_percent": round(coverage_pct, 2),
                "uncovered_critical": sorted(
                    uncovered_critical,
                    key=lambda x: x["coverage"]
                ),
                "status": "✓ PASS" if coverage_pct >= 80 else "✗ FAIL"
            }
        
        except Exception as e:
            return {"error": f"Failed to parse coverage: {str(e)}"}

    def parse_junit_results(self) -> Dict:
        """
        Parse JUnit XML test results.
        
        Returns
        -------
        dict with test statistics:
            {
                "total_tests": int,
                "passed": int,
                "failed": int,
                "errors": int,
                "skipped": int,
                "failed_tests": [...]
            }
        """
        junit_xml = self.reports_dir / "junit-results.xml"
        
        if not junit_xml.exists():
            return {"error": "junit-results.xml not found"}
        
        try:
            tree = ET.parse(junit_xml)
            root = tree.getroot()
            
            total_tests = int(root.get("tests", 0))
            failed = int(root.get("failures", 0))
            errors = int(root.get("errors", 0))
            skipped = int(root.get("skipped", 0))
            passed = total_tests - failed - errors - skipped
            
            # Extract failed test details
            failed_tests = []
            for testcase in root.findall(".//testcase"):
                failure = testcase.find("failure")
                error = testcase.find("error")
                
                if failure is not None or error is not None:
                    failed_tests.append({
                        "name": testcase.get("name"),
                        "classname": testcase.get("classname"),
                        "message": (failure or error).get("message", "No message")
                    })
            
            return {
                "total_tests": total_tests,
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "skipped": skipped,
                "pass_rate": round(passed / total_tests * 100, 2) if total_tests > 0 else 0,
                "failed_tests": failed_tests,
                "status": "✓ PASS" if failed == 0 and errors == 0 else "✗ FAIL"
            }
        
        except Exception as e:
            return {"error": f"Failed to parse junit: {str(e)}"}

    def generate_test_justification_report(self) -> str:
        """
        Generate markdown report justifying test case selection
        based on critical failure points and business logic.
        """
        report = """# Integration Test Justification Report

## Executive Summary
This integration test suite validates the complete Chess Elo Predictor pipeline,
focusing on critical failure points that could invalidate model training or
introduce subtle data quality issues.

## Test Strategy

### 1. PARSING CONSISTENCY (Test Suite 1)
**Why This Matters**: Parsing is the first step; errors here propagate through
the entire pipeline and are hard to detect downstream.

**Critical Failure Points**:
- Invalid Elo ratings (< 300 or > 2900) indicate parsing bugs
- Non-standard result values break model training
- Missing move sequences prevent feature computation
- Schema mismatches in parse output

**Test Cases**:
- ✓ PGN parsing produces required columns (white_elo, black_elo, result, moves_san)
- ✓ Elo ratings are within valid range [300, 2900]
- ✓ Result values are in {"1-0", "0-1", "1/2-1/2"}
- ✓ UCI parsing extracts non-empty move sequences

---

### 2. PHASE TRANSITIONS & MERGE INTEGRITY (Test Suite 2)
**Why This Matters**: Merges between phases silently lose data if schemas don't match.
Users won't notice until model performance degrades mysteriously.

**Critical Failure Points**:
- Inner joins dropping games without Stockfish evaluations (data loss)
- Column name mismatches (e.g., "Event" vs "event_id") causing merge failures
- Duplicate games appearing after merge (inflated dataset size)
- Schema incompatibility between PGN, UCI, and Stockfish formats

**Test Cases**:
- ✓ Merge preserves game count (no silent data loss)
- ✓ No duplicate merge suffix columns (_x, _y)
- ✓ ECO matching covers majority of games (> 50%)
- ✓ Column naming consistency across formats

---

### 3. DATA LEAKAGE DETECTION (Test Suite 3)
**Why This Matters**: Using white_elo or elo_gap to predict elo_bucket_white
makes the model worthless in production (we won't have these values for new games).
This is a critical business logic bug.

**Critical Failure Points**:
- Target-derived columns (white_elo, avg_elo) in feature matrix
- Leakage columns removed AFTER encoding instead of BEFORE
- Accidentally including game metadata (event_id, source) in features

**Test Cases**:
- ✓ Leakage columns removed before categorical encoding
- ✓ Target-derived features (avg_elo, elo_gap) absent from final matrix
- ✓ Feature independence verified (no circular dependencies)

---

### 4. PREPROCESSING & FEATURE ENGINEERING (Test Suite 4)
**Why This Matters**: Bugs here produce models with incorrect learned weights.

**Critical Failure Points**:
- Inconsistent categorical encoding (same input → different output)
- Scaling applied to wrong dataset (data leakage via fitted scaler)
- NaN values in Stockfish features propagating unchecked
- Missing value imputation using wrong statistics

**Test Cases**:
- ✓ Categorical encoding is deterministic (same input → same output)
- ✓ Scaling normalizes train set to mean≈0, std≈1
- ✓ No null values in required columns after preprocessing
- ✓ Feature value ranges are within expected bounds

---

### 5. TRAIN/TEST SPLIT INTEGRITY (Test Suite 5)
**Why This Matters**: Overlapping games between train and test lead to overfitting
and unreliable evaluation metrics.

**Critical Failure Points**:
- Duplicate games in train AND test sets
- Skewed class distributions (test has 70% Expert, train has 30%)
- Test set leakage (contaminated with training data)
- Stratification failure (class imbalance not preserved)

**Test Cases**:
- ✓ Train and test sets have zero overlap
- ✓ Class distributions similar across splits (stratified split validated)
- ✓ No duplicate games within each set

---

### 6. OUTPUT DATASET QUALITY (Test Suite 6)
**Why This Matters**: Final dataset is the input to model training.
Quality issues here directly impact model performance.

**Critical Failure Points**:
- Null values in required columns (breaks sklearn)
- Feature values outside expected ranges (data corruption)
- Missing classes in training data (model can't learn them)
- Extreme class imbalance (Expert 38.9%, Beginner 0.6%)

**Test Cases**:
- ✓ No null values in essential columns
- ✓ Feature value ranges are realistic (e.g., 5-300 moves per game)
- ✓ All 5 classes present in training data
- ✓ Class imbalance is acknowledged and handled

---

### 7. REPRODUCIBILITY & STABILITY (Test Suite 7)
**Why This Matters**: Non-determinism breaks model validation and makes debugging
impossible.

**Critical Failure Points**:
- Random seeds not set (different results each run)
- Parsing non-deterministic (floating point precision issues)
- Hash-based operations (dict iteration order in Python < 3.7)

**Test Cases**:
- ✓ Parsing is deterministic (same file → identical DataFrame)
- ✓ Encoding is reproducible (fixed seed produces same encodings)

---

## Test Coverage by Component

### Data Loading (Phase 1)
- Parsing functions: `parse_pgn()`, `parse_uci()` → **Tested**
- Data acquisition: `download_eco_database()` → **Not tested** (requires network)
- ZIP extraction → **Not tested** (file system dependent)

### Data Validation (Phase 2)
- Quality checks: `validation_report()` → **Tested implicitly**
- Schema validation → **Tested**

### Transformation (Phase 3)
- ECO matching: `match_eco()` → **Tested**
- Stockfish extraction → **Tested (mocked)**
- Dataset merge → **Tested**

### Preprocessing (Phase 4)
- Train/test split → **Tested**
- Categorical encoding → **Tested**
- Feature scaling → **Tested**
- Missing value imputation → **Tested**

### Feature Engineering (Phase 5)
- Feature computation → **Tested**
- Target encoding → **Tested**

### Model Training (Phase 6)
- Model initialization → **Not tested** (unit level)
- Model evaluation → **Not tested** (requires full pipeline)

---

## Coverage Target Analysis

### Critical Paths (Must be ≥ 95% coverage)
- `src/data/preprocess.py` — leakage prevention, encoding
- `src/data/validate.py` — quality checks
- `src/features/build_features.py` — target computation

### Important Paths (Must be ≥ 85% coverage)
- `src/data/load_data.py` — parsing, merging
- `src/data/transform.py` — ECO matching

### Testing Challenges
- **Network dependencies**: ECO database download requires GitHub/API access
- **Large file handling**: Full pipeline requires 100MB+ of data
- **Stockfish integration**: Requires valid stockfish.csv format

**Solution**: Integration tests use synthetic data fixtures.

---

## Known Untested Areas

1. **Data Acquisition**
   - Kaggle API authentication
   - GitHub ECO database download
   - ZIP file recursion edge cases
   
2. **Full Pipeline Execution**
   - End-to-end run with real data
   - Memory usage with large datasets
   - Performance benchmarks

3. **Model Training**
   - GridSearchCV convergence
   - Class weight application
   - Cross-validation stability

---

## Recommendations

1. **Increase Coverage**
   - Add mocking for Kaggle API calls
   - Create synthetic large-file tests
   - Add smoke test for full pipeline

2. **Continuous Monitoring**
   - Track coverage trends
   - Alert on coverage drops > 5%
   - Require coverage ≥ 80% for all pull requests

3. **Performance Testing**
   - Benchmark pipeline execution time
   - Track memory usage
   - Monitor model training convergence

---

Generated: {timestamp}
"""
        return report.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def generate_html_report(self, junit_data: Dict, coverage_data: Dict) -> str:
        """
        Generate HTML report combining test results and coverage.
        """
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Chess Elo Predictor - Test Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; }}
        .header {{ background: #f5f5f5; padding: 20px; border-radius: 4px; margin-bottom: 20px; }}
        .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007bff; }}
        .pass {{ border-left-color: #28a745; }}
        .fail {{ border-left-color: #dc3545; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #007bff; color: white; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
        .metric-value {{ font-size: 24px; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Chess Elo Predictor - Test Execution Report</h1>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>

    <div class="section pass" id="junit">
        <h2>Test Execution Results</h2>
"""
        
        if "error" in junit_data:
            html += f"<p><strong>Error:</strong> {junit_data['error']}</p>"
        else:
            pass_rate = junit_data.get("pass_rate", 0)
            status_class = "pass" if pass_rate >= 95 else "fail"
            
            html += f"""
        <div class="metric">
            <div>Total Tests</div>
            <div class="metric-value">{junit_data.get('total_tests', 0)}</div>
        </div>
        <div class="metric">
            <div>Passed</div>
            <div class="metric-value" style="color: #28a745;">{junit_data.get('passed', 0)}</div>
        </div>
        <div class="metric">
            <div>Failed</div>
            <div class="metric-value" style="color: #dc3545;">{junit_data.get('failed', 0)}</div>
        </div>
        <div class="metric">
            <div>Pass Rate</div>
            <div class="metric-value">{pass_rate}%</div>
        </div>

        <h3>Status: <span style="color: {'green' if pass_rate >= 95 else 'red'};">{junit_data.get('status', 'UNKNOWN')}</span></h3>
"""
            
            if junit_data.get("failed_tests"):
                html += "<h3>Failed Tests</h3><table><tr><th>Test</th><th>Message</th></tr>"
                for test in junit_data["failed_tests"]:
                    html += f"<tr><td>{test['classname']}::{test['name']}</td><td>{test['message']}</td></tr>"
                html += "</table>"
        
        html += "</div>"
        
        # Coverage section
        html += '<div class="section" id="coverage">\n<h2>Code Coverage</h2>\n'
        
        if "error" in coverage_data:
            html += f"<p><strong>Error:</strong> {coverage_data['error']}</p>"
        else:
            cov_pct = coverage_data.get("coverage_percent", 0)
            status_class = "pass" if cov_pct >= 80 else "fail"
            
            html += f"""
        <div class="metric">
            <div>Coverage</div>
            <div class="metric-value">{cov_pct}%</div>
        </div>
        <div class="metric">
            <div>Covered Lines</div>
            <div class="metric-value">{coverage_data.get('covered_lines', 0)}</div>
        </div>
        <div class="metric">
            <div>Total Lines</div>
            <div class="metric-value">{coverage_data.get('total_lines', 0)}</div>
        </div>

        <h3>Status: <span style="color: {'green' if cov_pct >= 80 else 'red'};">{coverage_data.get('status', 'UNKNOWN')}</span></h3>
"""
            
            if coverage_data.get("uncovered_critical"):
                html += "<h3>Critical Modules with Low Coverage</h3><table><tr><th>Module</th><th>Coverage</th></tr>"
                for module in coverage_data["uncovered_critical"]:
                    html += f"<tr><td>{module['file']}</td><td>{module['coverage']:.1f}%</td></tr>"
                html += "</table>"
        
        html += "</div>\n</body>\n</html>"
        return html

    def generate_all_reports(self) -> None:
        """Generate all report types."""
        print("=" * 70)
        print("CHESS ELO PREDICTOR - TEST REPORT GENERATOR")
        print("=" * 70)
        
        # Parse existing results
        junit_data = self.parse_junit_results()
        coverage_data = self.parse_coverage_xml()
        
        print("\n[1/3] Parsing Test Results...")
        print(f"  Tests: {junit_data.get('total_tests', 0)}")
        print(f"  Passed: {junit_data.get('passed', 0)}")
        print(f"  Failed: {junit_data.get('failed', 0)}")
        print(f"  Pass Rate: {junit_data.get('pass_rate', 0)}%")
        print(f"  Status: {junit_data.get('status', 'UNKNOWN')}")
        
        print("\n[2/3] Parsing Coverage Data...")
        print(f"  Total Lines: {coverage_data.get('total_lines', 0)}")
        print(f"  Covered Lines: {coverage_data.get('covered_lines', 0)}")
        print(f"  Coverage: {coverage_data.get('coverage_percent', 0)}%")
        print(f"  Status: {coverage_data.get('status', 'UNKNOWN')}")
        
        # Generate markdown report
        print("\n[3/3] Generating Reports...")
        
        justification_report = self.generate_test_justification_report()
        justif_path = self.test_results_dir / "TEST_JUSTIFICATION.md"
        justif_path.write_text(justification_report)
        print(f"  ✓ {justif_path.relative_to(self.project_root)}")
        
        # Generate HTML report
        html_report = self.generate_html_report(junit_data, coverage_data)
        html_path = self.test_results_dir / "test-report.html"
        html_path.write_text(html_report)
        print(f"  ✓ {html_path.relative_to(self.project_root)}")
        
        # Generate JSON summary
        summary = {
            "generated": datetime.now().isoformat(),
            "junit": junit_data,
            "coverage": coverage_data
        }
        json_path = self.test_results_dir / "test-summary.json"
        json_path.write_text(json.dumps(summary, indent=2))
        print(f"  ✓ {json_path.relative_to(self.project_root)}")
        
        print("\n" + "=" * 70)
        print("REPORTS GENERATED SUCCESSFULLY")
        print("=" * 70)
        
        return summary


if __name__ == "__main__":
    generator = TestReportGenerator()
    generator.generate_all_reports()
