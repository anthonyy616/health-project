"""
Evaluation Reports
==================

Generates structured evaluation reports combining metrics, plots, and
metadata into JSON and optional Markdown summaries.

Usage:
    from evaluation.reports import generate_report, save_report
    report = generate_report(hr_results=..., age_results=..., ...)
    save_report(report, "results/evaluation_report.json")
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    """Container for a complete evaluation report."""
    timestamp: str
    project_version: str
    dataset_info: Dict[str, Any]
    hr_results: Dict[str, Any]
    respiration_results: Optional[Dict[str, Any]] = None
    age_results: Optional[Dict[str, Any]] = None
    pupil_results: Optional[Dict[str, Any]] = None
    pipeline_results: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def generate_report(
    hr_results: Optional[Dict] = None,
    respiration_results: Optional[Dict] = None,
    age_results: Optional[Dict] = None,
    pupil_results: Optional[Dict] = None,
    pipeline_results: Optional[Dict] = None,
    dataset_info: Optional[Dict] = None,
    project_version: str = "0.1.0",
    metadata: Optional[Dict] = None,
) -> EvaluationReport:
    """
    Generate a structured evaluation report.

    Args:
        hr_results: Heart rate evaluation results (method_name -> HREvaluationResult)
        respiration_results: Respiration evaluation results dict
        age_results: Age evaluation results dict
        pupil_results: Pupil detection evaluation results dict
        pipeline_results: Full pipeline evaluation results dict
        dataset_info: Dataset metadata (name, version, n_subjects, etc.)
        project_version: Project version string
        metadata: Additional metadata (git hash, environment, etc.)

    Returns:
        EvaluationReport instance
    """
    # Serialize HR results if they're dataclass instances
    hr_serialized = {}
    if hr_results:
        for method_name, result in hr_results.items():
            if hasattr(result, '__dataclass_fields__'):
                hr_serialized[method_name] = asdict(result)
            else:
                hr_serialized[method_name] = result

    return EvaluationReport(
        timestamp=datetime.now().isoformat(),
        project_version=project_version,
        dataset_info=dataset_info or {},
        hr_results=hr_serialized,
        respiration_results=respiration_results,
        age_results=age_results,
        pupil_results=pupil_results,
        pipeline_results=pipeline_results,
        metadata=metadata or {},
    )


def save_report(report: EvaluationReport, output_path: str) -> str:
    """
    Save an evaluation report to JSON.

    Args:
        report: EvaluationReport instance
        output_path: Path to save the JSON report

    Returns:
        Path to saved file
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        json.dump(asdict(report), f, indent=2, default=str)

    logger.info(f"Evaluation report saved to: {output}")
    return str(output)


def generate_markdown_summary(report: EvaluationReport) -> str:
    """
    Generate a Markdown summary of an evaluation report.

    Args:
        report: EvaluationReport instance

    Returns:
        Markdown-formatted string
    """
    lines = [
        f"# Evaluation Report",
        f"",
        f"**Timestamp:** {report.timestamp}",
        f"**Project version:** {report.project_version}",
        f"",
    ]

    # Dataset info
    if report.dataset_info:
        lines.append("## Dataset")
        for key, value in report.dataset_info.items():
            lines.append(f"- **{key}:** {value}")
        lines.append("")

    # HR results
    if report.hr_results:
        lines.append("## Heart Rate Evaluation")
        lines.append("")
        for method_name, results in report.hr_results.items():
            if isinstance(results, dict):
                lines.append(f"### {method_name.upper()}")
                lines.append(f"- MAE: {results.get('mae', 'N/A'):.2f} BPM")
                lines.append(f"- RMSE: {results.get('rmse', 'N/A'):.2f} BPM")
                lines.append(f"- Pearson r: {results.get('pearson_r', 'N/A'):.3f}")
                lines.append(f"- R²: {results.get('r2', 'N/A'):.3f}")
                lines.append(f"- Within ±5 BPM: {results.get('percentage_within_5bpm', 'N/A'):.1%}")
                lines.append(f"- Within ±10 BPM: {results.get('percentage_within_10bpm', 'N/A'):.1%}")
                lines.append(f"- Failure rate: {results.get('failure_rate', 'N/A'):.1%}")
                lines.append(f"- N valid: {results.get('n_valid', 'N/A')} / {results.get('n_total', 'N/A')}")
                lines.append("")

    # Respiration results
    if report.respiration_results:
        lines.append("## Respiration Rate Evaluation")
        lines.append("")
        if isinstance(report.respiration_results, dict):
            for key, value in report.respiration_results.items():
                if isinstance(value, (int, float)):
                    lines.append(f"- **{key}:** {value}")
        lines.append("")

    # Age results
    if report.age_results:
        lines.append("## Age Estimation Evaluation")
        lines.append("")
        if isinstance(report.age_results, dict):
            overall = report.age_results.get("overall", {})
            if overall:
                lines.append(f"- **MAE:** {overall.get('mae', 'N/A'):.2f} years")
                lines.append(f"- **RMSE:** {overall.get('rmse', 'N/A'):.2f} years")
                lines.append(f"- **Pearson r:** {overall.get('pearson_r', 'N/A'):.3f}")
                lines.append(f"- **R²:** {overall.get('r2', 'N/A'):.3f}")
                lines.append(f"- **±3 years:** {overall.get('percentage_within_3yr', 'N/A'):.1%}")
                lines.append(f"- **±5 years:** {overall.get('percentage_within_5yr', 'N/A'):.1%}")
            per_group = report.age_results.get("per_age_group", {})
            if per_group:
                lines.append("")
                lines.append("### Per-Age-Group Breakdown")
                lines.append("")
                for group_name, group_data in per_group.items():
                    mae = group_data.get("mae")
                    n = group_data.get("n", 0)
                    if mae is not None:
                        lines.append(f"- **{group_name}:** n={n}, MAE={mae:.2f}")
                    else:
                        lines.append(f"- **{group_name}:** n={n} (insufficient)")
        lines.append("")

    # Pupil results
    if report.pupil_results:
        lines.append("## Pupil Detection Evaluation")
        lines.append("")
        if isinstance(report.pupil_results, dict):
            for key, value in report.pupil_results.items():
                if isinstance(value, (int, float)):
                    lines.append(f"- **{key}:** {value}")
        lines.append("")

    # Pipeline results
    if report.pipeline_results:
        lines.append("## Full Pipeline Evaluation")
        lines.append("")
        if isinstance(report.pipeline_results, dict):
            for key, value in report.pipeline_results.items():
                if isinstance(value, (int, float)):
                    lines.append(f"- **{key}:** {value}")
        lines.append("")

    # Metadata
    if report.metadata:
        lines.append("## Metadata")
        lines.append("")
        for key, value in report.metadata.items():
            lines.append(f"- **{key}:** {value}")
        lines.append("")

    return "\n".join(lines)


def save_markdown_summary(report: EvaluationReport, output_path: str) -> str:
    """
    Save a Markdown summary of the evaluation report.

    Args:
        report: EvaluationReport instance
        output_path: Path to save the Markdown file

    Returns:
        Path to saved file
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    markdown = generate_markdown_summary(report)
    with open(output, "w") as f:
        f.write(markdown)

    logger.info(f"Markdown summary saved to: {output}")
    return str(output)
