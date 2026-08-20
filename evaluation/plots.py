"""
Evaluation Plots
================

Generates publication-quality plots for vital signs evaluation:
- Predicted vs Ground Truth scatter plots
- Bland-Altman plots
- Error histograms
- Confidence vs Error correlation
- Per-subject MAE
- Latency distributions
- Lighting/distance condition comparisons

All plots save to specified output directories with consistent styling.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

# Use non-interactive backend for saving figures
matplotlib.use('Agg')

# Consistent plot styling
PLOT_STYLE = {
    'figure.figsize': (10, 8),
    'figure.dpi': 150,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
}


def _setup_style():
    """Apply consistent plot styling"""
    plt.rcParams.update(PLOT_STYLE)


def plot_predicted_vs_ground_truth(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    title: str = "Predicted vs Ground Truth",
    xlabel: str = "Ground Truth (BPM)",
    ylabel: str = "Predicted (BPM)",
    save_path: Optional[str] = None,
    method_name: str = "",
    show_identity_line: bool = True,
    show_regression_line: bool = True,
) -> str:
    """
    Create a scatter plot of predicted vs ground truth values.
    
    Args:
        predicted: Array of predicted values
        ground_truth: Array of ground truth values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save the figure (if None, auto-generated)
        method_name: Name of the method (for legend)
        show_identity_line: Show y=x line (perfect agreement)
        show_regression_line: Show linear regression fit
    
    Returns:
        Path to saved figure
    """
    _setup_style()
    
    predicted = np.asarray(predicted)
    ground_truth = np.asarray(ground_truth)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(ground_truth, predicted, alpha=0.6, s=50, c='steelblue', edgecolors='white', linewidth=0.5)
    
    # Identity line (perfect agreement)
    if show_identity_line:
        min_val = min(ground_truth.min(), predicted.min())
        max_val = max(ground_truth.max(), predicted.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Agreement')
    
    # Regression line
    if show_regression_line and len(ground_truth) > 1:
        coeffs = np.polyfit(ground_truth, predicted, 1)
        poly = np.poly1d(coeffs)
        x_line = np.linspace(ground_truth.min(), ground_truth.max(), 100)
        ax.plot(x_line, poly(x_line), 'r-', linewidth=2, 
                label=f'Regression (slope={coeffs[0]:.2f})')
    
    # Labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Equal aspect ratio for better visualization
    ax.set_aspect('equal', adjustable='box')
    
    # Save
    if save_path is None:
        save_path = f"results/predicted_vs_ground_truth_{method_name}.png"
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved predicted vs ground truth plot to {save_path}")
    return save_path


def plot_bland_altman(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    title: str = "Bland-Altman Plot",
    xlabel: str = "Mean of Two Measurements (BPM)",
    ylabel: str = "Difference (Predicted - Reference) (BPM)",
    save_path: Optional[str] = None,
    method_name: str = "",
) -> str:
    """
    Create a Bland-Altman plot showing agreement between methods.
    
    The Bland-Altman plot displays:
    - Each data point as (mean, difference)
    - Mean bias line (solid)
    - 95% limits of agreement (dashed)
    
    Args:
        predicted: Array of predicted values (new method)
        ground_truth: Array of ground truth values (reference)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save the figure
        method_name: Name of the method
    
    Returns:
        Path to saved figure
    """
    _setup_style()
    
    predicted = np.asarray(predicted)
    ground_truth = np.asarray(ground_truth)
    
    # Calculate differences and means
    differences = predicted - ground_truth
    means = (predicted + ground_truth) / 2.0
    
    # Statistical measures
    mean_bias = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    lower_loa = mean_bias - 1.96 * std_diff
    upper_loa = mean_bias + 1.96 * std_diff
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(means, differences, alpha=0.6, s=50, c='steelblue', edgecolors='white', linewidth=0.5)
    
    # Mean bias line
    ax.axhline(y=mean_bias, color='green', linewidth=2, label=f'Mean Bias: {mean_bias:.2f} BPM')
    
    # Limits of agreement
    ax.axhline(y=lower_loa, color='red', linewidth=2, linestyle='--', 
               label=f'Lower LoA: {lower_loa:.2f} BPM')
    ax.axhline(y=upper_loa, color='red', linewidth=2, linestyle='--',
               label=f'Upper LoA: {upper_loa:.2f} BPM')
    
    # Zero line
    ax.axhline(y=0, color='gray', linewidth=1, linestyle=':')
    
    # Labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add text box with statistics
    stats_text = (f'n = {len(differences)}\n'
                  f'Bias: {mean_bias:.2f} BPM\n'
                  f'SD: {std_diff:.2f} BPM\n'
                  f'LoA: [{lower_loa:.2f}, {upper_loa:.2f}]')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.95, 0.05, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    # Save
    if save_path is None:
        save_path = f"results/bland_altman_{method_name}.png"
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved Bland-Altman plot to {save_path}")
    return save_path


def plot_error_histogram(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    title: str = "Error Distribution",
    xlabel: str = "Error (Predicted - Ground Truth) (BPM)",
    ylabel: str = "Frequency",
    save_path: Optional[str] = None,
    method_name: str = "",
    bins: int = 30,
) -> str:
    """
    Create a histogram of prediction errors.
    
    Args:
        predicted: Array of predicted values
        ground_truth: Array of ground truth values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save the figure
        method_name: Name of the method
        bins: Number of histogram bins
    
    Returns:
        Path to saved figure
    """
    _setup_style()
    
    errors = np.asarray(predicted) - np.asarray(ground_truth)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Histogram
    n, bins_edges, patches = ax.hist(errors, bins=bins, edgecolor='black', 
                                      linewidth=0.5, alpha=0.7, color='steelblue')
    
    # Add vertical lines for statistics
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    ax.axvline(x=mean_error, color='green', linewidth=2, linestyle='-', 
               label=f'Mean: {mean_error:.2f} BPM')
    ax.axvline(x=mean_error - std_error, color='red', linewidth=1, linestyle='--',
               label=f'±1 SD: {std_error:.2f} BPM')
    ax.axvline(x=mean_error + std_error, color='red', linewidth=1, linestyle='--')
    
    # Labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Save
    if save_path is None:
        save_path = f"results/error_histogram_{method_name}.png"
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved error histogram to {save_path}")
    return save_path


def plot_confidence_vs_error(
    confidences: np.ndarray,
    errors: np.ndarray,
    title: str = "Confidence vs Error",
    xlabel: str = "Confidence (0-1)",
    ylabel: str = "Absolute Error (BPM)",
    save_path: Optional[str] = None,
    method_name: str = "",
) -> str:
    """
    Create a scatter plot showing relationship between confidence and error.
    
    A good SQI should show negative correlation: higher confidence = lower error.
    
    Args:
        confidences: Array of confidence values (0-1)
        errors: Array of absolute errors
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save the figure
        method_name: Name of the method
    
    Returns:
        Path to saved figure
    """
    _setup_style()
    
    confidences = np.asarray(confidences)
    errors = np.asarray(np.abs(errors))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(confidences, errors, alpha=0.6, s=50, c='steelblue', 
               edgecolors='white', linewidth=0.5)
    
    # Add trend line
    if len(confidences) > 1:
        # Bin by confidence and show mean error per bin
        n_bins = min(10, len(confidences) // 5)
        if n_bins > 1:
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_means = []
            bin_centers = []
            
            for i in range(n_bins):
                mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
                if np.sum(mask) > 0:
                    bin_means.append(np.mean(errors[mask]))
                    bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            
            if bin_centers:
                ax.plot(bin_centers, bin_means, 'r-o', linewidth=2, markersize=8,
                        label='Mean error per confidence bin')
    
    # Labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save
    if save_path is None:
        save_path = f"results/confidence_vs_error_{method_name}.png"
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved confidence vs error plot to {save_path}")
    return save_path


def plot_per_subject_mae(
    subject_ids: np.ndarray,
    maes: np.ndarray,
    title: str = "Per-Subject MAE",
    xlabel: str = "Subject ID",
    ylabel: str = "MAE (BPM)",
    save_path: Optional[str] = None,
    method_name: str = "",
) -> str:
    """
    Create a bar chart showing MAE for each subject.
    
    Args:
        subject_ids: Array of subject identifiers
        maes: Array of MAE values per subject
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save the figure
        method_name: Name of the method
    
    Returns:
        Path to saved figure
    """
    _setup_style()
    
    subject_ids = np.asarray(subject_ids)
    maes = np.asarray(maes)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Bar chart
    bars = ax.bar(range(len(subject_ids)), maes, color='steelblue', edgecolor='black', linewidth=0.5)
    
    # Highlight subjects with high MAE
    mean_mae = np.mean(maes)
    for i, mae in enumerate(maes):
        if mae > mean_mae * 1.5:
            bars[i].set_color('salmon')
    
    # Add mean line
    ax.axhline(y=mean_mae, color='green', linewidth=2, linestyle='--', 
               label=f'Mean: {mean_mae:.2f} BPM')
    
    # Labels
    ax.set_xticks(range(len(subject_ids)))
    ax.set_xticklabels(subject_ids, rotation=45, ha='right')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Save
    if save_path is None:
        save_path = f"results/per_subject_mae_{method_name}.png"
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved per-subject MAE plot to {save_path}")
    return save_path


def plot_latency_distribution(
    latencies_ms: np.ndarray,
    title: str = "Latency Distribution",
    xlabel: str = "Latency (ms)",
    ylabel: str = "Frequency",
    save_path: Optional[str] = None,
    method_name: str = "",
) -> str:
    """
    Create a histogram of inference latencies.
    
    Args:
        latencies_ms: Array of latency values in milliseconds
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save the figure
        method_name: Name of the method
    
    Returns:
        Path to saved figure
    """
    _setup_style()
    
    latencies_ms = np.asarray(latencies_ms)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Histogram
    n, bins, patches = ax.hist(latencies_ms, bins=30, edgecolor='black',
                                linewidth=0.5, alpha=0.7, color='steelblue')
    
    # Add vertical lines for percentiles
    p50 = np.percentile(latencies_ms, 50)
    p95 = np.percentile(latencies_ms, 95)
    p99 = np.percentile(latencies_ms, 99)
    
    ax.axvline(x=p50, color='green', linewidth=2, linestyle='-', 
               label=f'P50: {p50:.1f} ms')
    ax.axvline(x=p95, color='orange', linewidth=2, linestyle='--',
               label=f'P95: {p95:.1f} ms')
    ax.axvline(x=p99, color='red', linewidth=2, linestyle='--',
               label=f'P99: {p99:.1f} ms')
    
    # Labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Save
    if save_path is None:
        save_path = f"results/latency_distribution_{method_name}.png"
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved latency distribution plot to {save_path}")
    return save_path


def plot_method_comparison(
    methods: List[str],
    metrics: Dict[str, List[float]],
    title: str = "Method Comparison",
    save_path: Optional[str] = None,
) -> str:
    """
    Create a grouped bar chart comparing multiple methods across metrics.
    
    Args:
        methods: List of method names
        metrics: Dictionary mapping metric names to lists of values per method
        title: Plot title
        save_path: Path to save the figure
    
    Returns:
        Path to saved figure
    """
    _setup_style()
    
    n_methods = len(methods)
    n_metrics = len(metrics)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(n_metrics)
    width = 0.8 / n_methods
    
    colors = ['steelblue', 'salmon', 'lightgreen', 'gold', 'plum']
    
    for i, method in enumerate(methods):
        offset = (i - n_methods / 2 + 0.5) * width
        values = [metrics[m][i] for m in metrics.keys()]
        bars = ax.bar(x + offset, values, width, label=method, 
                      color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)
    
    # Labels
    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics.keys()))
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Save
    if save_path is None:
        save_path = "results/method_comparison.png"
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved method comparison plot to {save_path}")
    return save_path


def plot_lighting_conditions(
    results_by_condition: Dict[str, Dict[str, float]],
    metric: str = "mae",
    title: str = "Performance by Lighting Condition",
    xlabel: str = "Lighting Condition",
    ylabel: str = "MAE (BPM)",
    save_path: Optional[str] = None,
) -> str:
    """
    Create a bar chart comparing performance across lighting conditions.
    
    Args:
        results_by_condition: Dictionary mapping condition names to metric results
        metric: Which metric to plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save the figure
    
    Returns:
        Path to saved figure
    """
    _setup_style()
    
    conditions = list(results_by_condition.keys())
    values = [results_by_condition[c].get(metric, 0) for c in conditions]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    bars = ax.bar(conditions, values, color='steelblue', edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Save
    if save_path is None:
        save_path = f"results/lighting_{metric}.png"
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved lighting condition plot to {save_path}")
    return save_path


def plot_distance_conditions(
    results_by_distance: Dict[str, Dict[str, float]],
    metric: str = "mae",
    title: str = "Performance by Distance",
    xlabel: str = "Distance (m)",
    ylabel: str = "MAE (BPM)",
    save_path: Optional[str] = None,
) -> str:
    """
    Create a line plot showing performance vs distance.
    
    Args:
        results_by_distance: Dictionary mapping distance strings to metric results
        metric: Which metric to plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save the figure
    
    Returns:
        Path to saved figure
    """
    _setup_style()
    
    distances = list(results_by_distance.keys())
    values = [results_by_distance[d].get(metric, 0) for d in distances]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Line plot with markers
    ax.plot(distances, values, 'o-', color='steelblue', linewidth=2, markersize=8)
    
    # Add value labels
    for x, y in zip(distances, values):
        ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=10)
    
    # Labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Save
    if save_path is None:
        save_path = f"results/distance_{metric}.png"
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved distance condition plot to {save_path}")
    return save_path
