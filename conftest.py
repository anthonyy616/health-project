"""
Unified Test Logger Plugin
--------------------------
Captures pytest results and writes them to a timestamped JSON file.
Each run produces a unique file. The file includes:
  - Session metadata (start/end time, duration, test selection)
  - Per-test results (name, outcome, duration, failure info)
  - Summary counts

File naming: test_logs/test_results_YYYY-MM-DD_HH-MM-SS.json
"""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest


# Feature detection map: substrings in test file names -> feature labels
FEATURE_MAP = {
    "age_dataset": "age-dataset",
    "age_model": "age-model",
    "age_onnx": "age-onnx",
    "pipeline": "pipeline",
    "pupil": "pupil-detection",
    "respiration": "respiration",
    "rppg": "rppg-heart-rate",
}


def _detect_features(testnames: list[str]) -> list[str]:
    """Infer which features are being tested from collected test names."""
    features = set()
    for name in testnames:
        name_lower = name.lower()
        for keyword, label in FEATURE_MAP.items():
            if keyword in name_lower:
                features.add(label)
    return sorted(features)


class TestResultLog:
    """Accumulates test results across a session, then writes JSON."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.results: list[dict] = []
        self.features: list[str] = []
        self.cmd_args: list[str] = []

    def as_dict(self) -> dict:
        return {
            "meta": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "start_time": self.start_time,
                "end_time": self.end_time,
                "duration_seconds": (
                    round(self.end_time_ts - self.start_time_ts, 2)
                    if self.start_time_ts and self.end_time_ts
                    else None
                ),
                "features_tested": self.features,
                "cmd_args": self.cmd_args,
            },
            "summary": self._summary(),
            "results": self.results,
        }

    def _summary(self) -> dict:
        counts = {"passed": 0, "failed": 0, "error": 0, "skipped": 0, "total": 0}
        for r in self.results:
            outcome = r.get("outcome", "")
            if outcome in counts:
                counts[outcome] += 1
            counts["total"] += 1
        return counts


_log = TestResultLog()
_output_path: str = ""


def pytest_configure(config):
    global _output_path
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(config.rootpath) / "test_logs"
    log_dir.mkdir(exist_ok=True)

    # Resolve features after collection; for now use a temp name
    _output_path = str(log_dir / f"test_results_{timestamp}.json")

    # Store start time (all UTC)
    _log.start_time = now.isoformat()
    _log.start_time_ts = time.monotonic()

    # Capture actual pytest args (the test paths/patterns the user passed)
    _log.cmd_args = config.args if hasattr(config, 'args') and config.args else []

    # Hook into pytest to capture results
    config.pluginmanager.register(_ResultsHook(), "test_result_logger")


class _ResultsHook:
    """Internal plugin that captures per-test outcomes."""

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(item, call):
        outcome = yield
        report = outcome.get_result()
        if report.when == "call" or (report.when == "setup" and report.failed):
            outcome_label = "passed"
            if report.skipped:
                outcome_label = "skipped"
            elif report.failed:
                if report.when == "setup":
                    outcome_label = "error"
                else:
                    outcome_label = "failed"

            entry = {
                "node_id": report.nodeid,
                "outcome": outcome_label,
                "duration_seconds": round(report.duration, 4) if report.duration else 0,
                "when": report.when,
            }

            # Attach failure long repr if present
            if report.failed and report.longreprtext:
                entry["failure_info"] = report.longreprtext

            _log.results.append(entry)

    def pytest_collection_modifyitems(config, items):
        all_names = [item.nodeid for item in items]
        _log.features = _detect_features(all_names)

        # Update filename to include features if specific
        if _log.features and _log.features != sorted(FEATURE_MAP.values()):
            feature_tag = "+".join(_log.features)
            # Rename output path to include features
            global _output_path
            p = Path(_output_path)
            _output_path = str(p.parent / f"{p.stem}_{feature_tag}{p.suffix}")

    def pytest_sessionfinish(session, exitstatus):
        _log.end_time = datetime.now(timezone.utc).isoformat()
        _log.end_time_ts = time.monotonic()
        data = _log.as_dict()
        with open(_output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        # Print a short notice so the user sees it
        print(f"\n[LOG] Test log written to: {_output_path}")
