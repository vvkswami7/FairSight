"""
Visualization service for bias metrics.
Generates base64-encoded charts for Demographic Parity and Equalized Odds.
"""

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import io
import base64
from typing import Dict, List

# Use non-interactive backend to avoid display issues in headless environments
matplotlib.use('Agg')
sns.set_style("darkgrid")

# Color palette for consistency
PALETTE = {
    "primary": "#7c6af7",
    "success": "#22d3a0", 
    "warning": "#f59e0b",
    "danger": "#f87171",
    "info": "#60a5fa"
}


def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='#12121a', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def generate_demographic_parity_chart(group_rates: Dict[str, float], attribute: str) -> str:
    """
    Generate a bar chart showing selection rates by demographic group.
    
    DP = |P(Ŷ=1|A=a) - P(Ŷ=1|A=b)|
    
    Args:
        group_rates: Dict mapping group names to positive outcome rates (0-1)
        attribute: Name of the sensitive attribute (e.g., "gender", "race")
    
    Returns:
        Base64-encoded PNG of the chart
    """
    groups = list(group_rates.keys())
    rates = [group_rates[g] for g in groups]
    
    if not groups:
        return ""
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Color bars based on disparity (further from mean = more red)
    mean_rate = sum(rates) / len(rates)
    colors = [
        PALETTE["danger"] if abs(r - mean_rate) > 0.1 else 
        PALETTE["warning"] if abs(r - mean_rate) > 0.05 else 
        PALETTE["success"]
        for r in rates
    ]
    
    bars = ax.bar(groups, rates, color=colors, edgecolor='rgba(255,255,255,0.2)', linewidth=1.5, alpha=0.85)
    
    # Add value labels on bars
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1%}',
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='#f0eff8')
    
    # Add mean line
    ax.axhline(y=mean_rate, color=PALETTE["info"], linestyle='--', linewidth=2, 
               label=f'Average: {mean_rate:.1%}', alpha=0.7)
    
    ax.set_xlabel('Group', fontsize=12, fontweight='bold', color='#f0eff8')
    ax.set_ylabel('Positive Outcome Rate', fontsize=12, fontweight='bold', color='#f0eff8')
    ax.set_title(f'Demographic Parity: {attribute}', fontsize=14, fontweight='bold', color='#f0eff8', pad=20)
    ax.set_ylim(0, max(rates) * 1.15 if rates else 1)
    ax.legend(loc='upper right', framealpha=0.9, facecolor='#1a1a26', edgecolor='rgba(255,255,255,0.1)')
    
    # Styling
    ax.set_facecolor('#0a0a0f')
    fig.patch.set_facecolor('#12121a')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('rgba(255,255,255,0.1)')
    ax.spines['bottom'].set_color('rgba(255,255,255,0.1)')
    ax.tick_params(colors='#8887a0')
    
    return fig_to_base64(fig)


def generate_equalized_odds_chart(tpr_per_group: Dict[str, float], fpr_per_group: Dict[str, float], attribute: str) -> str:
    """
    Generate a grouped bar chart comparing TPR and FPR across groups.
    
    EO: Ensure TPR and FPR are equal across groups.
    TPR_disparity = |P(Ŷ=1|A=a,Y=1) - P(Ŷ=1|A=b,Y=1)|
    
    Args:
        tpr_per_group: Dict mapping group names to True Positive Rates
        fpr_per_group: Dict mapping group names to False Positive Rates
        attribute: Name of the sensitive attribute
    
    Returns:
        Base64-encoded PNG of the chart
    """
    groups = list(tpr_per_group.keys())
    
    if not groups:
        return ""
    
    tpr_vals = [tpr_per_group[g] for g in groups]
    fpr_vals = [fpr_per_group[g] for g in groups]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = range(len(groups))
    width = 0.35
    
    # Plot TPR and FPR side by side
    bars1 = ax.bar([i - width/2 for i in x], tpr_vals, width, label='True Positive Rate (TPR)',
                   color=PALETTE["success"], edgecolor='rgba(255,255,255,0.2)', linewidth=1.5, alpha=0.85)
    bars2 = ax.bar([i + width/2 for i in x], fpr_vals, width, label='False Positive Rate (FPR)',
                   color=PALETTE["danger"], edgecolor='rgba(255,255,255,0.2)', linewidth=1.5, alpha=0.85)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}',
                    ha='center', va='bottom', fontsize=10, color='#f0eff8', fontweight='bold')
    
    ax.set_xlabel('Group', fontsize=12, fontweight='bold', color='#f0eff8')
    ax.set_ylabel('Rate', fontsize=12, fontweight='bold', color='#f0eff8')
    ax.set_title(f'Equalized Odds: {attribute}', fontsize=14, fontweight='bold', color='#f0eff8', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, max(tpr_vals + fpr_vals) * 1.15 if (tpr_vals + fpr_vals) else 1)
    ax.legend(loc='upper right', framealpha=0.9, facecolor='#1a1a26', edgecolor='rgba(255,255,255,0.1)')
    
    # Styling
    ax.set_facecolor('#0a0a0f')
    fig.patch.set_facecolor('#12121a')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('rgba(255,255,255,0.1)')
    ax.spines['bottom'].set_color('rgba(255,255,255,0.1)')
    ax.tick_params(colors='#8887a0')
    
    return fig_to_base64(fig)


def generate_fairness_score_heatmap(bias_analysis: Dict) -> str:
    """
    Generate a heatmap of fairness scores across all sensitive attributes.
    
    Args:
        bias_analysis: Dict mapping attribute names to bias analysis results
    
    Returns:
        Base64-encoded PNG of the heatmap
    """
    if not bias_analysis:
        return ""
    
    attributes = list(bias_analysis.keys())
    scores = [bias_analysis[attr].get("fairness_score", 0) for attr in attributes]
    
    fig, ax = plt.subplots(figsize=(max(8, len(attributes) * 1.2), 3))
    
    # Create heatmap-like visualization with colors
    colors = [
        PALETTE["danger"] if s < 40 else
        PALETTE["warning"] if s < 60 else
        PALETTE["primary"] if s < 80 else
        PALETTE["success"]
        for s in scores
    ]
    
    bars = ax.barh(attributes, scores, color=colors, edgecolor='rgba(255,255,255,0.2)', linewidth=1.5, alpha=0.85)
    
    # Add score labels
    for bar, score in zip(bars, scores):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f' {score}/100',
                ha='left', va='center', fontsize=11, fontweight='bold', color='#f0eff8')
    
    ax.set_xlabel('Fairness Score', fontsize=12, fontweight='bold', color='#f0eff8')
    ax.set_title('Fairness Score by Attribute', fontsize=14, fontweight='bold', color='#f0eff8', pad=20)
    ax.set_xlim(0, 105)
    
    # Styling
    ax.set_facecolor('#0a0a0f')
    fig.patch.set_facecolor('#12121a')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('rgba(255,255,255,0.1)')
    ax.spines['bottom'].set_color('rgba(255,255,255,0.1)')
    ax.tick_params(colors='#8887a0')
    
    return fig_to_base64(fig)


def generate_all_visualizations(bias_results: Dict) -> Dict[str, str]:
    """
    Generate all visualizations for a bias analysis result.
    
    Args:
        bias_results: Complete bias analysis dictionary from bias_engine
    
    Returns:
        Dict mapping visualization names to base64-encoded PNGs
    """
    charts = {}
    bias_data = bias_results.get("bias_analysis", {})
    
    # Overall fairness heatmap
    charts["overall_fairness"] = generate_fairness_score_heatmap(bias_data)
    
    # Per-attribute charts
    for attr, analysis in bias_data.items():
        dp = analysis.get("demographic_parity", {})
        eo = analysis.get("equalized_odds", {})
        
        if dp:
            charts[f"dp_{attr}"] = generate_demographic_parity_chart(
                dp.get("group_rates", {}), 
                attr
            )
        
        if eo:
            charts[f"eo_{attr}"] = generate_equalized_odds_chart(
                eo.get("tpr_per_group", {}),
                eo.get("fpr_per_group", {}),
                attr
            )
    
    return charts
