"""
Tab 8: Correlation Analysis
=============================
Correlations between metrics, biological parameters, and model comparisons.
Boxplots and rankings to answer: Which method works best under which condition?
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List
from scipy import stats

from analysis import (
    compute_per_replicate_metrics,
    compute_metric_correlations,
    compute_correlation_with_biological_params,
    prepare_correlation_heatmap_data,
    compute_condition_ranking,
    compute_model_advantage_matrix,
    compute_summary_with_ci,
    analyze_original_by_tool,
)


# Biological parameters for each condition
CONDITION_PARAMS = {
    'C1': {'μ': 2.5e-5, 'r': 0.005, 'μ_label': '2.5×10⁻⁵', 'r_label': '0.005'},
    'C2': {'μ': 2.5e-5, 'r': 0.010, 'μ_label': '2.5×10⁻⁵', 'r_label': '0.010'},
    'C3': {'μ': 2.5e-5, 'r': 0.020, 'μ_label': '2.5×10⁻⁵', 'r_label': '0.020'},
    'C4': {'μ': 1.0e-4, 'r': 0.005, 'μ_label': '1.0×10⁻⁴', 'r_label': '0.005'},
    'C5': {'μ': 1.0e-4, 'r': 0.010, 'μ_label': '1.0×10⁻⁴', 'r_label': '0.010'},
    'C6': {'μ': 1.0e-4, 'r': 0.020, 'μ_label': '1.0×10⁻⁴', 'r_label': '0.020'},
    'C7': {'μ': 2.0e-4, 'r': 0.005, 'μ_label': '2.0×10⁻⁴', 'r_label': '0.005'},
    'C8': {'μ': 2.0e-4, 'r': 0.010, 'μ_label': '2.0×10⁻⁴', 'r_label': '0.010'},
    'C9': {'μ': 2.0e-4, 'r': 0.020, 'μ_label': '2.0×10⁻⁴', 'r_label': '0.020'},
}


@st.cache_data(ttl=3600, show_spinner=False)
def _prepare_correlation_data(all_data: Dict) -> Dict:
    """Prepare all data needed for correlation analysis."""
    conditions = sorted(all_data.keys())
    models = sorted(set(m for d in all_data.values() for m in d.keys()))
    
    # Per-replicate data
    per_rep = {}
    for c in conditions:
        per_rep[c] = {}
        for m in models:
            original_df, result_df = all_data[c][m]
            if result_df is not None:
                orig_stats = analyze_original_by_tool(original_df) if original_df is not None else {}
                per_rep[c][m] = compute_per_replicate_metrics(result_df, orig_stats, list(range(1, 101)))
            else:
                per_rep[c][m] = None
    
    # Summary DataFrame with biological params
    summary_rows = []
    for c in conditions:
        params = CONDITION_PARAMS.get(c, {})
        for m in models:
            df = per_rep[c].get(m)
            if df is not None and not df.empty:
                row = {
                    'Condition': c,
                    'Model': m,
                    'μ': params.get('μ', np.nan),
                    'r': params.get('r', np.nan),
                    'μ_label': params.get('μ_label', '?'),
                    'r_label': params.get('r_label', '?'),
                }
                for metric in ['Accuracy', 'FPR', 'BP_Distance']:
                    if metric in df.columns:
                        row[metric] = df[metric].mean()
                        row[f'{metric}_SD'] = df[metric].std()
                summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Combined per-replicate data for metric correlations
    all_rep_data = []
    for c in conditions:
        params = CONDITION_PARAMS.get(c, {})
        for m in models:
            df = per_rep[c].get(m)
            if df is not None and not df.empty:
                dc = df.copy()
                dc['Condition'] = c
                dc['Model'] = m
                dc['μ'] = params.get('μ', np.nan)
                dc['r'] = params.get('r', np.nan)
                all_rep_data.append(dc)
    
    combined_rep = pd.concat(all_rep_data, ignore_index=True) if all_rep_data else pd.DataFrame()
    
    return {
        'per_rep': per_rep,
        'summary_df': summary_df,
        'combined_rep': combined_rep,
        'conditions': conditions,
        'models': models,
    }


def render(all_data: Dict, summary: Dict = None, show_raw: bool = False):
    """Main render for correlation analysis tab."""
    
    st.markdown("## 🔬 Correlation & Comparative Analysis")
    st.caption("Which method works best under which condition?")
    
    data = _prepare_correlation_data(all_data)
    
    if data['summary_df'].empty:
        st.warning("No data available.")
        return
    
    # Sub-tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Metric Correlations",
        "🧬 Biological Parameters",
        "🏆 Model Rankings",
        "📈 Boxplots",
        "📝 Summary"
    ])
    
    with tab1:
        _render_metric_correlations(data)
    
    with tab2:
        _render_biological_correlations(data)
    
    with tab3:
        _render_model_rankings(data)
    
    with tab4:
        _render_boxplots(data)
    
    with tab5:
        _render_summary(data)


def _render_metric_correlations(data: Dict):
    """Render correlations between Accuracy, FPR, and BP Distance."""
    st.markdown("### 📊 Correlations Between Metrics")
    st.caption("How do Accuracy, FPR, and BP Distance relate to each other?")
    
    combined_rep = data['combined_rep']
    conditions = data['conditions']
    
    # Overall correlation
    corr_results = compute_metric_correlations(combined_rep)
    
    if 'error' not in corr_results:
        st.markdown("#### Overall Correlation Matrix (All Data Combined)")
        st.caption("Spearman rank correlation (non-parametric)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if corr_results['spearman']['matrix'] is not None:
                fig = px.imshow(
                    corr_results['spearman']['matrix'],
                    text_auto='.3f',
                    aspect='auto',
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1,
                    title="Spearman ρ"
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if corr_results['spearman']['p_values'] is not None:
                # Format p-values
                p_display = corr_results['spearman']['p_values'].copy()
                for col in p_display.columns:
                    p_display[col] = p_display[col].apply(
                        lambda p: f"{p:.4f}" + ("*" if p < 0.05 else "") + ("**" if p < 0.01 else "") + ("***" if p < 0.001 else "")
                    )
                
                st.dataframe(p_display, use_container_width=True)
                st.caption("* p<.05  ** p<.01  *** p<.001")
        
        # Scatter plots
        st.markdown("---")
        st.markdown("#### Pairwise Scatter Plots")
        
        metrics = ['Accuracy', 'FPR', 'BP_Distance']
        available = [m for m in metrics if m in combined_rep.columns]
        
        if len(available) >= 2:
            for i, m1 in enumerate(available):
                for m2 in available[i+1:]:
                    r, p = stats.spearmanr(
                        combined_rep[m1].dropna(),
                        combined_rep[m2].dropna()
                    )
                    
                    fig = px.scatter(
                        combined_rep, x=m1, y=m2,
                        color='Model',
                        color_discrete_map={'DT': '#4285f4', 'LR': '#34a853', 'NN': '#fbbc04'},
                        opacity=0.6,
                        trendline='ols',
                        title=f"{m1.replace('_', ' ')} vs {m2.replace('_', ' ')}  (ρ = {r:.3f}, p = {p:.4f})"
                    )
                    fig.update_layout(template="plotly_white", title_x=0.5, height=350)
                    st.plotly_chart(fig, use_container_width=True)
    
    # Per-condition correlation
    st.markdown("---")
    st.markdown("#### Correlation by Condition")
    
    cond_corrs = []
    for c in conditions:
        c_data = combined_rep[combined_rep['Condition'] == c]
        if len(c_data) >= 10:
            for m1, m2 in [('Accuracy', 'FPR'), ('Accuracy', 'BP_Distance'), ('FPR', 'BP_Distance')]:
                if m1 in c_data.columns and m2 in c_data.columns:
                    r, p = stats.spearmanr(c_data[m1].dropna(), c_data[m2].dropna())
                    cond_corrs.append({
                        'Condition': c,
                        'Pair': f'{m1} vs {m2}',
                        'Spearman_ρ': r,
                        'p_value': p,
                        'Significant': '✅' if p < 0.05 else '❌',
                    })
    
    if cond_corrs:
        corr_df = pd.DataFrame(cond_corrs)
        pivot = corr_df.pivot(index='Condition', columns='Pair', values='Spearman_ρ')
        
        fig = px.imshow(
            pivot, text_auto='.3f', aspect='auto',
            color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
            title="Spearman ρ by Condition"
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)


def _render_biological_correlations(data: Dict):
    """Render correlations with biological parameters μ and r."""
    st.markdown("### 🧬 Correlation with Biological Parameters")
    st.caption("How do mutation rate (μ) and recombination rate (r) affect performance?")
    
    summary_df = data['summary_df']
    models = data['models']
    
    # Correlation with biological params
    bio_corr = compute_correlation_with_biological_params(summary_df)
    
    if bio_corr:
        st.markdown("#### Spearman Correlation: Performance vs Biological Parameters")
        
        # Build display table
        rows = []
        for model in models:
            model_corr = bio_corr.get(model, {})
            for metric in ['Accuracy', 'FPR', 'BP_Distance']:
                corr_mu = model_corr.get(f'{metric}_vs_μ', {})
                corr_r = model_corr.get(f'{metric}_vs_r', {})
                
                rows.append({
                    'Model': model,
                    'Metric': metric.replace('_', ' '),
                    'ρ vs μ': f"{corr_mu.get('spearman_r', 0):.3f}",
                    'p (μ)': f"{corr_mu.get('p_value', 1):.4f}",
                    'Sig (μ)': '✅' if corr_mu.get('significant', False) else '❌',
                    'ρ vs r': f"{corr_r.get('spearman_r', 0):.3f}",
                    'p (r)': f"{corr_r.get('p_value', 1):.4f}",
                    'Sig (r)': '✅' if corr_r.get('significant', False) else '❌',
                })
        
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    
    # Heatmap of all correlations
    st.markdown("---")
    st.markdown("#### Correlation Heatmap: Metrics × Biological Parameters")
    
    corr_matrix = prepare_correlation_heatmap_data(summary_df)
    
    if not corr_matrix.empty:
        fig = px.imshow(
            corr_matrix, text_auto='.3f', aspect='auto',
            color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
            title="Spearman Correlation Matrix"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Scatter: Performance vs μ and r
    st.markdown("---")
    st.markdown("#### Performance vs Biological Parameters")
    
    metric = st.selectbox(
        "Select Metric:", ['Accuracy', 'FPR', 'BP_Distance'],
        key="bio_metric"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            summary_df, x='μ', y=metric, color='Model',
            color_discrete_map={'DT': '#4285f4', 'LR': '#34a853', 'NN': '#fbbc04'},
            trendline='ols', size_max=8,
            title=f"{metric.replace('_', ' ')} vs Mutation Rate (μ)",
            labels={'μ': 'Mutation Rate', metric: metric.replace('_', ' ')}
        )
        fig.update_layout(template="plotly_white", title_x=0.5, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            summary_df, x='r', y=metric, color='Model',
            color_discrete_map={'DT': '#4285f4', 'LR': '#34a853', 'NN': '#fbbc04'},
            trendline='ols', size_max=8,
            title=f"{metric.replace('_', ' ')} vs Recombination Rate (r)",
            labels={'r': 'Recombination Rate', metric: metric.replace('_', ' ')}
        )
        fig.update_layout(template="plotly_white", title_x=0.5, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    st.markdown("---")
    st.markdown("#### Key Findings")
    
    # Compute correlations for interpretation
    for model in models:
        model_data = summary_df[summary_df['Model'] == model]
        st.markdown(f"**{model}:**")
        
        for metric in ['Accuracy', 'FPR', 'BP_Distance']:
            if metric in model_data.columns and len(model_data) >= 3:
                r_mu, p_mu = stats.spearmanr(model_data['μ'], model_data[metric])
                r_r, p_r = stats.spearmanr(model_data['r'], model_data[metric])
                
                mu_effect = "↑" if r_mu > 0 else "↓"
                r_effect = "↑" if r_r > 0 else "↓"
                
                st.markdown(
                    f"- **{metric.replace('_', ' ')}**: "
                    f"μ: ρ={r_mu:.3f} (p={p_mu:.3f}) {mu_effect} | "
                    f"r: ρ={r_r:.3f} (p={p_r:.3f}) {r_effect}"
                )


def _render_model_rankings(data: Dict):
    """Render model rankings per condition."""
    st.markdown("### 🏆 Model Rankings by Condition")
    st.caption("Which classifier performs best in each condition?")
    
    conditions = data['conditions']
    models = data['models']
    per_rep = data['per_rep']
    
    # Metric selector
    metric = st.selectbox(
        "Select Metric:",
        ['Accuracy', 'FPR', 'BP_Distance'],
        key="ranking_metric"
    )
    
    # Compute rankings
    ranking_df = compute_condition_ranking(per_rep, conditions, models, metric)
    
    if not ranking_df.empty:
        st.markdown(f"#### Rankings: {metric.replace('_', ' ')}")
        st.caption(
            "Rank 1 = Best" if metric != 'FPR' else "Rank 1 = Lowest FPR (Best)"
        )
        
        # Pivot to show rankings
        pivot_rank = ranking_df.pivot(index='Condition', columns='Model', values='Rank')
        
        # Color the best (rank 1) in green
        def highlight_best(val):
            return 'background-color: #d4edda; font-weight: bold' if val == 1 else ''
        
        st.dataframe(
            pivot_rank.style.map(highlight_best),
            use_container_width=True
        )
        
        # Bar chart with rankings
        st.markdown("---")
        st.markdown("#### Performance by Condition (with Rankings)")
        
        pivot_mean = ranking_df.pivot(index='Condition', columns='Model', values='Mean')
        
        fig = go.Figure()
        colors = {'DT': '#4285f4', 'LR': '#34a853', 'NN': '#fbbc04'}
        
        for model in models:
            if model in pivot_mean.columns:
                fig.add_trace(go.Bar(
                    name=model,
                    x=pivot_mean.index,
                    y=pivot_mean[model],
                    marker_color=colors.get(model, '#888'),
                    text=[f"Rank {int(pivot_rank.loc[c, model])}" if c in pivot_rank.index and model in pivot_rank.columns else ''
                          for c in pivot_mean.index],
                    textposition='outside',
                ))
        
        fig.update_layout(
            title=f"{metric.replace('_', ' ')} by Condition and Model",
            xaxis_title="Condition",
            yaxis_title=metric.replace('_', ' '),
            template="plotly_white",
            title_x=0.5,
            height=450,
            barmode='group',
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Win count
        st.markdown("---")
        st.markdown("#### Win Count: Best Model per Condition")
        
        win_counts = ranking_df[ranking_df['Rank'] == 1]['Model'].value_counts()
        
        win_df = pd.DataFrame({
            'Model': models,
            'Wins': [win_counts.get(m, 0) for m in models],
            'Percentage': [win_counts.get(m, 0) / len(conditions) * 100 for m in models],
        })
        
        fig = px.bar(
            win_df, x='Model', y='Wins', color='Model',
            color_discrete_map=colors,
            text='Wins',
            title=f"Number of Conditions Where Model is Best ({metric.replace('_', ' ')})"
        )
        fig.update_layout(template="plotly_white", title_x=0.5, height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(win_df, use_container_width=True, hide_index=True)


def _render_boxplots(data: Dict):
    """Render boxplots comparing models across conditions."""
    st.markdown("### 📈 Distribution Analysis: Boxplots")
    st.caption("Performance distributions across 100 replicates per condition")
    
    combined_rep = data['combined_rep']
    conditions = data['conditions']
    models = data['models']
    
    # Metric selector
    metric = st.selectbox(
        "Select Metric:",
        ['Accuracy', 'FPR', 'BP_Distance'],
        key="boxplot_metric"
    )
    
    # All conditions boxplot
    st.markdown(f"#### {metric.replace('_', ' ')} Distribution: All Conditions")
    
    fig = px.box(
        combined_rep, x='Condition', y=metric, color='Model',
        color_discrete_map={'DT': '#4285f4', 'LR': '#34a853', 'NN': '#fbbc04'},
        title=f"{metric.replace('_', ' ')} Across All Conditions",
    )
    fig.update_layout(
        template="plotly_white", title_x=0.5, height=500,
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Individual condition boxplots
    st.markdown("---")
    st.markdown("#### Per-Condition Boxplots")
    
    n_cols = 3
    n_rows = (len(conditions) + n_cols - 1) // n_cols
    
    for row in range(n_rows):
        cols = st.columns(n_cols)
        for col_idx in range(n_cols):
            cond_idx = row * n_cols + col_idx
            if cond_idx < len(conditions):
                c = conditions[cond_idx]
                with cols[col_idx]:
                    c_data = combined_rep[combined_rep['Condition'] == c]
                    
                    if not c_data.empty and metric in c_data.columns:
                        fig = px.box(
                            c_data, x='Model', y=metric, color='Model',
                            color_discrete_map={'DT': '#4285f4', 'LR': '#34a853', 'NN': '#fbbc04'},
                            title=f"{c}",
                        )
                        fig.update_layout(
                            template="plotly_white", title_x=0.5,
                            height=250, showlegend=False,
                            margin=dict(t=30, b=10, l=10, r=10),
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    # Faceted boxplot by μ and r groups
    st.markdown("---")
    st.markdown("#### Boxplots Grouped by Biological Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Group by μ
        combined_rep['μ_group'] = combined_rep['μ'].map({
            2.5e-5: 'μ = 2.5×10⁻⁵ (Low)',
            1.0e-4: 'μ = 1.0×10⁻⁴ (Mid)',
            2.0e-4: 'μ = 2.0×10⁻⁴ (High)',
        })
        
        fig = px.box(
            combined_rep.dropna(subset=['μ_group']),
            x='μ_group', y=metric, color='Model',
            color_discrete_map={'DT': '#4285f4', 'LR': '#34a853', 'NN': '#fbbc04'},
            title=f"{metric.replace('_', ' ')} by Mutation Rate (μ)"
        )
        fig.update_layout(template="plotly_white", title_x=0.5, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Group by r
        combined_rep['r_group'] = combined_rep['r'].map({
            0.005: 'r = 0.005 (Rare)',
            0.010: 'r = 0.010 (Moderate)',
            0.020: 'r = 0.020 (Frequent)',
        })
        
        fig = px.box(
            combined_rep.dropna(subset=['r_group']),
            x='r_group', y=metric, color='Model',
            color_discrete_map={'DT': '#4285f4', 'LR': '#34a853', 'NN': '#fbbc04'},
            title=f"{metric.replace('_', ' ')} by Recombination Rate (r)"
        )
        fig.update_layout(template="plotly_white", title_x=0.5, height=400)
        st.plotly_chart(fig, use_container_width=True)


def _render_summary(data: Dict):
    """Render summary answering the main research question."""
    st.markdown("### 📝 Summary: Which Method Works Best?")
    
    summary_df = data['summary_df']
    models = data['models']
    conditions = data['conditions']
    per_rep = data['per_rep']
    
    # Overall winner
    st.markdown("#### Overall Performance Comparison")
    
    overall = summary_df.groupby('Model').agg({
        'Accuracy': 'mean',
        'FPR': 'mean',
        'BP_Distance': 'mean',
    }).round(4)
    
    # Rank each metric
    for metric in ['Accuracy', 'FPR', 'BP_Distance']:
        if metric in overall.columns:
            reverse = metric != 'FPR'
            overall[f'{metric}_Rank'] = overall[metric].rank(ascending=not reverse)
    
    st.dataframe(overall, use_container_width=True)
    
    # Per-condition winner
    st.markdown("---")
    st.markdown("#### Best Model per Condition")
    
    winners = []
    for condition in conditions:
        cond_data = summary_df[summary_df['Condition'] == condition]
        if not cond_data.empty:
            # Find best for each metric
            best_acc = cond_data.loc[cond_data['Accuracy'].idxmax(), 'Model']
            best_fpr = cond_data.loc[cond_data['FPR'].idxmin(), 'Model']
            best_bp = cond_data.loc[cond_data['BP_Distance'].idxmin(), 'Model']
            
            winners.append({
                'Condition': condition,
                'Best Accuracy': best_acc,
                'Best FPR': best_fpr,
                'Best BP Distance': best_bp,
            })
    
    if winners:
        winners_df = pd.DataFrame(winners)
        
        # Count overall wins
        all_winners = []
        for _, row in winners_df.iterrows():
            all_winners.extend([row['Best Accuracy'], row['Best FPR'], row['Best BP Distance']])
        
        win_counts = pd.Series(all_winners).value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(winners_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**Total Wins (across all metrics):**")
            for model in models:
                wins = win_counts.get(model, 0)
                st.metric(f"{model}", f"{wins}/27", f"{wins/27*100:.0f}%")
    
    # Answer the question
    st.markdown("---")
    st.markdown("#### 🎯 Answer: Which Method Works Best?")
    
    # Determine overall best
    overall_wins = overall.filter(like='_Rank').mean(axis=1)
    best_model = overall_wins.idxmin()
    best_model_rank = overall_wins.min()
    
    st.success(f"""
    ### {best_model} is the best overall classifier
    
    **Average rank across metrics:** {best_model_rank:.1f} (lower = better)
    
    **Key advantages:**
    - Highest Accuracy in {len(winners_df[winners_df['Best Accuracy'] == best_model])}/9 conditions
    - Lowest FPR in {len(winners_df[winners_df['Best FPR'] == best_model])}/9 conditions  
    - Best BP Distance in {len(winners_df[winners_df['Best BP Distance'] == best_model])}/9 conditions
    """)
    
    # Detailed breakdown
    st.markdown("**Condition-specific recommendations:**")
    
    for condition in conditions:
        cond_winners = winners_df[winners_df['Condition'] == condition]
        if not cond_winners.empty:
            row = cond_winners.iloc[0]
            
            # Determine overall best for this condition
            cond_data = summary_df[summary_df['Condition'] == condition]
            if not cond_data.empty:
                cond_scores = cond_data.copy()
                # Normalize and combine
                for metric in ['Accuracy', 'FPR', 'BP_Distance']:
                    if metric in cond_scores.columns:
                        if metric == 'FPR':
                            cond_scores[f'{metric}_score'] = 1 - (cond_scores[metric] - cond_scores[metric].min()) / (cond_scores[metric].max() - cond_scores[metric].min() + 1e-10)
                        else:
                            cond_scores[f'{metric}_score'] = (cond_scores[metric] - cond_scores[metric].min()) / (cond_scores[metric].max() - cond_scores[metric].min() + 1e-10)
                
                score_cols = [c for c in cond_scores.columns if c.endswith('_score')]
                if score_cols:
                    cond_scores['Overall_Score'] = cond_scores[score_cols].mean(axis=1)
                    best_for_cond = cond_scores.loc[cond_scores['Overall_Score'].idxmax(), 'Model']
                    
                    st.markdown(f"- **{condition}**: Use **{best_for_cond}** (Acc: {row['Best Accuracy']}, FPR: {row['Best FPR']}, BP: {row['Best BP Distance']})")