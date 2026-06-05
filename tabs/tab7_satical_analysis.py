"""
Tab 7: Statistical Analysis
=============================
Statistical validation following strict protocol:
- Friedman test (global, per metric)
- Wilcoxon signed-rank post-hoc (paired, per metric pair)
- Holm-Bonferroni correction
- Rank-biserial correlation (effect size)
- Descriptive stats with 95% CI
Displays ALL results automatically without user selection.
Auto-computes on load with caching.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from typing import Dict, List, Optional, Tuple
from itertools import combinations

from analysis import (
    compute_per_replicate_metrics,
    compute_summary_with_ci,
    analyze_original_by_tool,
    format_p_value,
)



def _compute_all_per_replicate(all_data: Dict) -> Dict:
    """Cached computation of all per-replicate metrics."""
    per_rep = {}
    conditions = list(all_data.keys())
    
    models = set()
    for cond_data in all_data.values():
        models.update(cond_data.keys())
    models = sorted(models)
    
    for condition in conditions:
        per_rep[condition] = {}
        for model in models:
            original_df, result_df = all_data[condition][model]
            if result_df is not None:
                orig_stats = analyze_original_by_tool(original_df) if original_df is not None else {}
                per_rep[condition][model] = compute_per_replicate_metrics(
                    result_df, orig_stats, list(range(1, 101))
                )
            else:
                per_rep[condition][model] = None
    return per_rep


# ============================================================================
# STRICT STATISTICAL FUNCTIONS
# ============================================================================

def _holm_bonferroni(p_values: List[float]) -> List[float]:
    """Apply Holm-Bonferroni correction to a list of p-values."""
    n = len(p_values)
    if n == 0:
        return []
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]
    corrected = np.zeros(n)
    
    for k, (idx, p) in enumerate(zip(sorted_idx, sorted_p)):
        corrected[idx] = min(p * (n - k), 1.0)
        if k > 0:
            corrected[idx] = max(corrected[idx], corrected[sorted_idx[k-1]])
    
    return corrected.tolist()


def _rank_biserial_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute rank-biserial correlation (effect size for Wilcoxon).
    Formula: r = 1 - (2*U)/(n1*n2)
    Range: -1 to 1 (0 = no effect)
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    
    if len(x) < 3:
        return np.nan
    
    diff = x - y
    diff = diff[diff != 0]
    
    if len(diff) == 0:
        return 0.0
    
    ranks = stats.rankdata(np.abs(diff))
    W_pos = np.sum(ranks[diff > 0])
    W_neg = np.sum(ranks[diff < 0])
    
    r = (W_pos - W_neg) / (W_pos + W_neg) if (W_pos + W_neg) > 0 else 0.0
    return r


def _interpret_effect_size(r: float) -> str:
    """Interpret rank-biserial correlation effect size."""
    ar = abs(r)
    if ar < 0.1:
        return "negligible"
    elif ar < 0.3:
        return "small"
    elif ar < 0.5:
        return "medium"
    else:
        return "large"


def friedman_test_per_metric(
    per_rep_data: Dict, 
    conditions: List[str], 
    models: List[str],
    metric: str
    ) -> Dict:
    """Friedman test (non-parametric repeated measures ANOVA)."""
    block_data = {m: [] for m in models}
    
    for condition in conditions:
        for model in models:
            df = per_rep_data.get(condition, {}).get(model)
            if df is not None and not df.empty and metric in df.columns:
                values = df[metric].dropna()
                if len(values) > 0:
                    block_data[model].append(values.mean())
                else:
                    block_data[model].append(np.nan)
            else:
                block_data[model].append(np.nan)
    
    valid_blocks = []
    for i in range(len(conditions)):
        if all(not np.isnan(block_data[m][i]) for m in models):
            valid_blocks.append(i)
    
    if len(valid_blocks) < 3:
        return {'test': 'insufficient_data', 'error': f'Only {len(valid_blocks)} complete blocks'}
    
    groups = [np.array([block_data[m][i] for i in valid_blocks]) for m in models]
    
    try:
        statistic, p_value = stats.friedmanchisquare(*groups)
        n = len(valid_blocks)
        k = len(models)
        W = statistic / (n * (k - 1)) if n > 0 and k > 1 else 0
        
        if W < 0.1:
            w_interp = "negligible"
        elif W < 0.3:
            w_interp = "small"
        elif W < 0.5:
            w_interp = "medium"
        else:
            w_interp = "large"
        
        return {
            'test': 'friedman',
            'metric': metric,
            'n_blocks': n,
            'n_models': k,
            'statistic': statistic,
            'p_value': p_value,
            'p_value_formatted': format_p_value(p_value),
            'significant': p_value < 0.05,
            'kendall_w': W,
            'w_interpretation': w_interp,
            'interpretation': (
                f"Significant difference among classifiers" if p_value < 0.05 
                else "No significant difference among classifiers"
            )
        }
    except Exception as e:
        return {'test': 'error', 'error': str(e)}


def wilcoxon_posthoc_all_metrics(
    per_rep_data: Dict,
    conditions: List[str],
    models: List[str]
    ) -> Dict[str, pd.DataFrame]:
    """
    Post-hoc pairwise Wilcoxon signed-rank tests for ALL metrics.
    Returns dict of DataFrames keyed by metric.
    """
    all_results = {}
    
    for metric in ['Accuracy', 'FPR', 'BP_Distance']:
        results = []
        
        for m1, m2 in combinations(models, 2):
            pair_data = []
            
            for condition in conditions:
                df1 = per_rep_data.get(condition, {}).get(m1)
                df2 = per_rep_data.get(condition, {}).get(m2)
                
                if (df1 is not None and not df1.empty and metric in df1.columns and
                    df2 is not None and not df2.empty and metric in df2.columns):
                    
                    merged = pd.merge(
                        df1[['Replicate', metric]].rename(columns={metric: 'm1'}),
                        df2[['Replicate', metric]].rename(columns={metric: 'm2'}),
                        on='Replicate',
                        how='inner'
                    )
                    
                    for _, row in merged.iterrows():
                        if not (pd.isna(row['m1']) or pd.isna(row['m2'])):
                            pair_data.append({'m1_val': row['m1'], 'm2_val': row['m2']})
            
            if len(pair_data) >= 3:
                x = np.array([p['m1_val'] for p in pair_data])
                y = np.array([p['m2_val'] for p in pair_data])
                
                try:
                    statistic, p_value = stats.wilcoxon(x, y, alternative='two-sided')
                    r = _rank_biserial_correlation(x, y)
                    
                    results.append({
                        'Comparison': f"{m1} vs {m2}",
                        'Test': 'Wilcoxon signed-rank',
                        'N_Pairs': len(x),
                        'Mean_Diff': np.mean(x - y),
                        'Median_Diff': np.median(x - y),
                        'Statistic': statistic,
                        'P_Value_Raw': p_value,
                        'Effect_Size_r': r,
                        'Effect_Interpretation': _interpret_effect_size(r),
                    })
                except Exception as e:
                    results.append({
                        'Comparison': f"{m1} vs {m2}",
                        'Test': 'Wilcoxon signed-rank',
                        'N_Pairs': 0,
                        'Mean_Diff': np.nan,
                        'Median_Diff': np.nan,
                        'Statistic': np.nan,
                        'P_Value_Raw': np.nan,
                        'Effect_Size_r': np.nan,
                        'Effect_Interpretation': 'N/A',
                        'Error': str(e),
                    })
        
        if results:
            results_df = pd.DataFrame(results)
            
            # Holm-Bonferroni correction
            valid_p = results_df['P_Value_Raw'].dropna()
            if len(valid_p) > 0:
                corrected_p = _holm_bonferroni(valid_p.tolist())
                p_corrected_map = {}
                for i, idx in enumerate(valid_p.index):
                    p_corrected_map[idx] = corrected_p[i]
                
                results_df['P_Value_Holm'] = results_df.index.map(
                    lambda i: p_corrected_map.get(i, np.nan)
                )
                results_df['P_Value_Formatted'] = results_df['P_Value_Holm'].apply(
                    lambda p: format_p_value(p) if not pd.isna(p) else 'N/A'
                )
                results_df['Significant (Holm-Bonferroni)'] = results_df['P_Value_Holm'].apply(
                    lambda p: p < 0.05 if not pd.isna(p) else False
                )
            else:
                results_df['P_Value_Holm'] = np.nan
                results_df['P_Value_Formatted'] = 'N/A'
                results_df['Significant (Holm-Bonferroni)'] = False
            
            all_results[metric] = results_df
    
    return all_results


def descriptive_stats_all(
    per_rep_data: Dict,
    conditions: List[str],
    models: List[str]
    ) -> Dict[str, pd.DataFrame]:
    """
    Compute descriptive statistics for ALL conditions × models × metrics.
    Returns dict of DataFrames keyed by metric.
    """
    all_stats = {}
    
    for metric in ['Accuracy', 'FPR', 'BP_Distance']:
        rows = []
        
        for condition in conditions:
            for model in models:
                df = per_rep_data.get(condition, {}).get(model)
                if df is not None and not df.empty and metric in df.columns:
                    s = compute_summary_with_ci(df, metric)
                    values = df[metric].dropna()
                    rows.append({
                        'Condition': condition,
                        'Model': model,
                        'N': s['n'],
                        'Mean': s['mean'],
                        'SD': s['std'],
                        'SEM': s['sem'],
                        'CI_95_Lower': s['ci_95_lower'],
                        'CI_95_Upper': s['ci_95_upper'],
                        'Median': s['median'],
                        'Q1': s['q1'],
                        'Q3': s['q3'],
                        'IQR': s['iqr'],
                        'Min': values.min(),
                        'Max': values.max(),
                    })
                else:
                    rows.append({
                        'Condition': condition,
                        'Model': model,
                        'N': 0,
                        'Mean': np.nan,
                        'SD': np.nan,
                        'SEM': np.nan,
                        'CI_95_Lower': np.nan,
                        'CI_95_Upper': np.nan,
                        'Median': np.nan,
                        'Q1': np.nan,
                        'Q3': np.nan,
                        'IQR': np.nan,
                        'Min': np.nan,
                        'Max': np.nan,
                    })
        
        all_stats[metric] = pd.DataFrame(rows)
    
    return all_stats


# ============================================================================
# RENDER FUNCTIONS
# ============================================================================

def render(all_data: Dict, summary: Dict = None, show_raw: bool = False):
    """Main render - displays ALL results automatically."""
    
    st.markdown("## 📊 Statistical Validation")
    st.markdown("""
    **Protocol:** Friedman test (global) → Wilcoxon signed-rank (post-hoc) → Holm-Bonferroni correction → Rank-biserial correlation (effect size)
    """)
    
    conditions = sorted(all_data.keys())
    models = sorted(set(m for d in all_data.values() for m in d.keys()))
    
    with st.spinner("Computing per-replicate statistics..."):
        per_rep_data = _compute_all_per_replicate(all_data)
    st.success(f"✅ {len(conditions)} conditions × {len(models)} models × 100 replicates")
    
    # Pre-compute ALL results
    with st.spinner("Running statistical tests..."):
        friedman_results = {}
        for metric in ['Accuracy', 'FPR', 'BP_Distance']:
            friedman_results[metric] = friedman_test_per_metric(per_rep_data, conditions, models, metric)
        
        posthoc_results = wilcoxon_posthoc_all_metrics(per_rep_data, conditions, models)
        desc_results = descriptive_stats_all(per_rep_data, conditions, models)
    
    # Combine all data for global use
    dfs = []
    for c in conditions:
        for m in models:
            df = per_rep_data.get(c, {}).get(m)
            if df is not None and not df.empty:
                dc = df.copy(); dc['Condition'] = c; dc['Model'] = m
                dfs.append(dc)
    gdf = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    # Sub-tabs
    subtab1, subtab2, subtab3, subtab4 = st.tabs([
        "🌐 1. Global (Friedman)", 
        "🔄 2. Post-Hoc (Wilcoxon)", 
        "📋 3. Descriptive Stats", 
        "📝 4. Full Report"
    ])
    
    with subtab1:
        _render_friedman_all(friedman_results)
    
    with subtab2:
        _render_posthoc_all(posthoc_results, models)
    
    with subtab3:
        _render_descriptive_all(desc_results, conditions, models)
    
    with subtab4:
        _render_full_report(friedman_results, posthoc_results, desc_results, 
                           conditions, models, gdf)


def _render_friedman_all(friedman_results: Dict):
    """Render ALL Friedman test results."""
    st.markdown("### 🌐 Global Test: Friedman (Repeated Measures ANOVA)")
    st.caption("H₀: All classifiers have equal performance | H₁: At least one differs")
    
    metrics = ['Accuracy', 'FPR', 'BP_Distance']
    
    # Summary table
    st.markdown("#### All Metrics Summary")
    summary_rows = []
    for metric in metrics:
        res = friedman_results.get(metric, {})
        if res.get('test') == 'friedman':
            summary_rows.append({
                'Metric': metric.replace('_', ' '),
                'χ²': f"{res['statistic']:.4f}",
                'df': res['n_models'] - 1,
                'p-value': res['p_value_formatted'],
                "Kendall's W": f"{res['kendall_w']:.4f}",
                'Effect Size': res['w_interpretation'],
                'Significant (α=0.05)': '✅ YES' if res['significant'] else '❌ NO',
            })
    
    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    
    # Detailed results per metric
    st.markdown("---")
    st.markdown("#### Detailed Results by Metric")
    
    for metric in metrics:
        res = friedman_results.get(metric, {})
        if res.get('test') == 'friedman':
            st.markdown(f"##### {metric.replace('_', ' ')}")
            
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.metric("χ²", f"{res['statistic']:.4f}")
            with c2:
                st.metric("df", res['n_models'] - 1)
            with c3:
                st.metric("p-value", res['p_value_formatted'])
            with c4:
                st.metric("Kendall's W", f"{res['kendall_w']:.4f}")
            with c5:
                st.metric("Blocks (n)", res['n_blocks'])
            
            if res['significant']:
                st.success(f"✅ **{res['interpretation']}** (p = {res['p_value']:.4f} < 0.05)")
                st.info("→ Significant post-hoc results shown in Tab 2 (Post-Hoc Wilcoxon)")
            else:
                st.info(f"ℹ️ **{res['interpretation']}** (p = {res['p_value']:.4f} ≥ 0.05)")
        elif res.get('test') == 'insufficient_data':
            st.warning(f"⚠️ **{metric}**: {res.get('error', 'Insufficient data')}")
        elif res.get('test') == 'error':
            st.error(f"❌ **{metric}**: {res.get('error', 'Unknown error')}")
        
        st.markdown("---")


def _render_posthoc_all(posthoc_results: Dict[str, pd.DataFrame], models: List[str]):
    """Render ALL post-hoc Wilcoxon results for all metrics."""
    st.markdown("### 🔄 Post-Hoc: Wilcoxon Signed-Rank Tests")
    st.caption("Paired comparisons with Holm-Bonferroni correction")
    st.caption(f"Comparisons: DT vs LR | DT vs NN | LR vs NN")
    
    metrics = ['Accuracy', 'FPR', 'BP_Distance']
    
    # Combined summary table
    st.markdown("#### All Comparisons Summary")
    
    all_rows = []
    for metric in metrics:
        df = posthoc_results.get(metric)
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                all_rows.append({
                    'Metric': metric.replace('_', ' '),
                    'Comparison': row['Comparison'],
                    'N_Pairs': row['N_Pairs'],
                    'Mean_Diff': f"{row['Mean_Diff']:.4f}",
                    'Median_Diff': f"{row['Median_Diff']:.4f}",
                    'Statistic': f"{row['Statistic']:.1f}" if not pd.isna(row['Statistic']) else 'N/A',
                    'p (Holm)': row['P_Value_Formatted'],
                    'Effect r': f"{row['Effect_Size_r']:.3f}" if not pd.isna(row['Effect_Size_r']) else 'N/A',
                    'Effect Size': row['Effect_Interpretation'],
                    'Significant': '✅ YES' if row['Significant (Holm-Bonferroni)'] else '❌ NO',
                })
    
    if all_rows:
        summary_df = pd.DataFrame(all_rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Detailed per metric
    st.markdown("---")
    
    for metric in metrics:
        df = posthoc_results.get(metric)
        if df is None or df.empty:
            st.warning(f"No post-hoc results for {metric}")
            continue
        
        st.markdown(f"#### {metric.replace('_', ' ')}")
        
        # Display table
        display_cols = [
            'Comparison', 'N_Pairs', 'Mean_Diff', 'Median_Diff',
            'Statistic', 'P_Value_Formatted', 'Effect_Size_r',
            'Effect_Interpretation', 'Significant (Holm-Bonferroni)'
        ]
        
        st.dataframe(
            df[display_cols].style.apply(
                lambda x: ['background: #d4edda' if v else '' for v in x],
                subset=['Significant (Holm-Bonferroni)']
            ),
            use_container_width=True,
            hide_index=True,
        )
        
        # Effect size visualization
        st.markdown("**Effect Sizes:**")
        fig = go.Figure()
        for _, row in df.iterrows():
            color = '#34a853' if row['Significant (Holm-Bonferroni)'] else '#ea4335'
            r_val = row['Effect_Size_r']
            fig.add_trace(go.Bar(
                x=[abs(r_val) if not pd.isna(r_val) else 0],
                y=[row['Comparison']],
                orientation='h',
                marker_color=color,
                text=f"r = {r_val:.3f}" if not pd.isna(r_val) else 'N/A',
                textposition='outside',
                showlegend=False,
            ))
        
        for thresh, label, color in [(0.1, 'negligible', 'gray'), (0.3, 'small', 'orange'), (0.5, 'medium', 'red')]:
            fig.add_vline(x=thresh, line_dash="dash", line_color=color, opacity=0.5,
                         annotation_text=label, annotation_position="top")
        
        max_r = max([abs(r) for r in df['Effect_Size_r'].dropna()], default=0.5)
        fig.update_layout(
            title=f"Effect Sizes: {metric.replace('_', ' ')}",
            xaxis_title="|Rank-Biserial Correlation|",
            template="plotly_white",
            title_x=0.5,
            height=200,
            xaxis=dict(range=[0, max(1.0, max_r * 1.3)])
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Significance summary
        sig_count = df['Significant (Holm-Bonferroni)'].sum()
        if sig_count > 0:
            sig_pairs = df[df['Significant (Holm-Bonferroni)']]['Comparison'].tolist()
            st.success(f"**{sig_count}/3 significant:** {', '.join(sig_pairs)}")
        else:
            st.info("No comparisons survived Holm-Bonferroni correction.")
        
        st.markdown("---")
    
    # Effect size interpretation
    st.markdown("#### Effect Size Interpretation")
    st.caption("Rank-Biserial Correlation: |r| < 0.1 = negligible | 0.1–0.3 = small | 0.3–0.5 = medium | > 0.5 = large")


def _render_descriptive_all(desc_results: Dict[str, pd.DataFrame], 
                            conditions: List[str], models: List[str]):
    """Render ALL descriptive statistics."""
    st.markdown("### 📋 Descriptive Statistics")
    st.caption(f"All {len(conditions)} conditions × {len(models)} models × 100 replicates")
    
    metrics = ['Accuracy', 'FPR', 'BP_Distance']
    
    # Per metric display
    for metric in metrics:
        df = desc_results.get(metric)
        if df is None or df.empty:
            continue
        
        st.markdown(f"#### {metric.replace('_', ' ')}")
        
        # Formatted table
        display_df = df.copy()
        display_df['Mean ± SD'] = display_df.apply(
            lambda r: f"{r['Mean']:.4f} ± {r['SD']:.4f}" if not pd.isna(r['Mean']) else 'N/A', axis=1
        )
        display_df['95% CI'] = display_df.apply(
            lambda r: f"[{r['CI_95_Lower']:.4f}, {r['CI_95_Upper']:.4f}]" if not pd.isna(r['CI_95_Lower']) else 'N/A', axis=1
        )
        display_df['Median [IQR]'] = display_df.apply(
            lambda r: f"{r['Median']:.4f} [{r['Q1']:.4f}, {r['Q3']:.4f}]" if not pd.isna(r['Median']) else 'N/A', axis=1
        )
        
        st.dataframe(
            display_df[['Condition', 'Model', 'N', 'Mean ± SD', '95% CI', 'Median [IQR]', 'Min', 'Max']],
            use_container_width=True,
            hide_index=True,
            height=350,
        )
        
        # Heatmap
        st.markdown("**Heatmap:**")
        pvt = df.pivot(index='Condition', columns='Model', values='Mean')
        fig = px.imshow(
            pvt, text_auto='.4f', aspect='auto',
            color_continuous_scale='RdYlGn' if metric != 'FPR' else 'RdYlGn_r',
            title=f"Mean {metric.replace('_', ' ')}"
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Box plot: all conditions
        st.markdown("**Distribution by Condition:**")
        plot_data = []
        for _, row in df.iterrows():
            plot_data.append({
                'Condition': row['Condition'],
                'Model': row['Model'],
                'Mean': row['Mean'],
            })
        
        if plot_data:
            pdf = pd.DataFrame(plot_data)
            fig = px.box(
                pdf, x='Condition', y='Mean', color='Model',
                color_discrete_map={'DT': '#4285f4', 'LR': '#34a853', 'NN': '#fbbc04'},
                title=f"{metric.replace('_', ' ')} Across Conditions"
            )
            fig.update_layout(template="plotly_white", title_x=0.5, height=350, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
    
    # Download all
    st.markdown("### Downloads")
    for metric in metrics:
        df = desc_results.get(metric)
        if df is not None:
            st.download_button(
                f"📥 {metric}.csv",
                df.to_csv(index=False),
                f"descriptive_{metric}.csv",
                key=f"dl_desc_{metric}"
            )


def _render_full_report(friedman_results: Dict, posthoc_results: Dict, 
                        desc_results: Dict, conditions: List[str], 
                        models: List[str], gdf: pd.DataFrame):
    """Generate complete statistical report."""
    st.markdown("### 📝 Complete Statistical Report")
    
    metrics = ['Accuracy', 'FPR', 'BP_Distance']
    
    rep = []
    rep.append("# Statistical Validation Report")
    rep.append(f"**Generated:** {pd.Timestamp.now():%Y-%m-%d %H:%M}")
    rep.append(f"**Design:** {len(conditions)} conditions × {len(models)} models × 100 replicates")
    rep.append(f"**Conditions:** {', '.join(conditions)}")
    rep.append(f"**Models:** {', '.join(models)}")
    rep.append("")
    rep.append("---")
    rep.append("")
    
    # Protocol
    rep.append("## Statistical Protocol")
    rep.append("")
    rep.append("1. **Friedman test** — Global test per metric (non-parametric repeated measures ANOVA)")
    rep.append("2. **Wilcoxon signed-rank** — Post-hoc pairwise comparisons (paired design)")
    rep.append("3. **Holm-Bonferroni correction** — Multiple comparison correction")
    rep.append("4. **Rank-biserial correlation (r)** — Effect size")
    rep.append("   - |r| < 0.1: negligible | 0.1–0.3: small | 0.3–0.5: medium | > 0.5: large")
    rep.append("")
    rep.append("---")
    rep.append("")
    
    # 1. Friedman
    rep.append("## 1. Global Tests (Friedman)")
    rep.append("")
    rep.append("| Metric | χ² | df | p | Kendall's W | Effect | Significant? |")
    rep.append("|---|---|---|---|---|---|---|")
    
    for metric in metrics:
        res = friedman_results.get(metric, {})
        if res.get('test') == 'friedman':
            sig = "✅ YES" if res['significant'] else "❌ NO"
            rep.append(
                f"| {metric.replace('_', ' ')} | {res['statistic']:.4f} | {res['n_models']-1} | "
                f"{res['p_value_formatted']} | {res['kendall_w']:.4f} | {res['w_interpretation']} | {sig} |"
            )
    rep.append("")
    rep.append("---")
    rep.append("")
    
    # 2. Post-hoc
    rep.append("## 2. Post-Hoc Tests (Wilcoxon Signed-Rank)")
    rep.append("")
    rep.append("All p-values Holm-Bonferroni corrected.")
    rep.append("")
    
    for metric in metrics:
        rep.append(f"### {metric.replace('_', ' ')}")
        rep.append("")
        df = posthoc_results.get(metric)
        if df is not None and not df.empty:
            rep.append("| Comparison | N | Mean Diff | Statistic | p (Holm) | Effect r | Effect | Significant? |")
            rep.append("|---|---|---|---|---|---|---|---|")
            for _, row in df.iterrows():
                sig = "✅ YES" if row['Significant (Holm-Bonferroni)'] else "❌ NO"
                rep.append(
                    f"| {row['Comparison']} | {row['N_Pairs']} | {row['Mean_Diff']:.4f} | "
                    f"{row['Statistic']:.1f} | {row['P_Value_Formatted']} | "
                    f"{row['Effect_Size_r']:.3f} | {row['Effect_Interpretation']} | {sig} |"
                )
            rep.append("")
        else:
            rep.append("No results available.")
            rep.append("")
    
    rep.append("---")
    rep.append("")
    
    # 3. Descriptive
    rep.append("## 3. Descriptive Statistics")
    rep.append("")
    
    for metric in metrics:
        rep.append(f"### {metric.replace('_', ' ')}")
        rep.append("")
        rep.append("| Condition | Model | N | Mean ± SD | 95% CI | Median [IQR] |")
        rep.append("|---|---|---|---|---|---|")
        
        df = desc_results.get(metric)
        if df is not None:
            for _, row in df.iterrows():
                if not pd.isna(row['Mean']):
                    rep.append(
                        f"| {row['Condition']} | {row['Model']} | {int(row['N'])} | "
                        f"{row['Mean']:.4f} ± {row['SD']:.4f} | "
                        f"[{row['CI_95_Lower']:.4f}, {row['CI_95_Upper']:.4f}] | "
                        f"{row['Median']:.4f} [{row['Q1']:.4f}, {row['Q3']:.4f}] |"
                    )
        rep.append("")
    
    rep.append("---")
    rep.append("")
    
    # 4. Summary
    rep.append("## 4. Summary of Findings")
    rep.append("")
    
    for metric in metrics:
        res = friedman_results.get(metric, {})
        if res.get('test') == 'friedman':
            if res['significant']:
                df = posthoc_results.get(metric)
                if df is not None:
                    sig_pairs = df[df['Significant (Holm-Bonferroni)']]
                    if not sig_pairs.empty:
                        pairs_str = ', '.join(sig_pairs['Comparison'].tolist())
                        rep.append(f"**{metric.replace('_', ' ')}**: {res['interpretation']}. "
                                  f"Significant differences: {pairs_str}.")
                    else:
                        rep.append(f"**{metric.replace('_', ' ')}**: {res['interpretation']}, "
                                  f"but no pairwise differences survived correction.")
            else:
                rep.append(f"**{metric.replace('_', ' ')}**: {res['interpretation']}.")
        rep.append("")
    
    txt = "\n".join(rep)
    
    # Preview & Download
    with st.expander("📄 Preview Full Report"):
        st.markdown(txt)
    
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("📥 Report (.txt)", txt, "statistical_report.txt", key="dl_rep")
    with c2:
        if not gdf.empty:
            st.download_button("📥 All Data (.csv)", gdf.to_csv(index=False), "all_metrics.csv", key="dl_data")