"""UI display components for Streamlit."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from analysis import (
    analyze_original_by_tool, 
    extract_tag_level_stats,
    compute_comparison_stats,
    compute_summary_totals,
    get_missing_tags_list,
    compute_study_metrics,
    analyze_breakpoint_distances,
    compute_study_summary_all_conditions,


)
from visualization import (
    create_tag_comparison_bar_chart,
    create_match_rate_scatter,
    create_summary_metrics_chart
)






# ============================================================================
# STUDY METRICS DISPLAY FUNCTIONS
# ============================================================================


# ============================================================================
# EXISTING DISPLAY FUNCTIONS (Keep all your existing functions below)
# ============================================================================

# [Keep all your existing functions: display_original_analysis, 
#  display_result_tag_analysis, display_comparison_analysis, display_css]
# They remain exactly as you have them

# ============================================================================
# STUDY METRICS DISPLAY FUNCTIONS
# ============================================================================

def display_study_metrics_section(result_df: pd.DataFrame, original_df: pd.DataFrame,
                                   condition: str, model: str):
    """
    Display the special study section with three key metrics.
    """
    from analysis import compute_study_metrics, analyze_original_by_tool, analyze_breakpoint_distances
    
    st.markdown("---")
    st.markdown("## 🔬 Study Metrics Summary")
    st.markdown("*Key performance indicators for recombination detection analysis*")
    
    # Compute metrics
    orig_stats = analyze_original_by_tool(original_df) if original_df is not None else {}
    metrics = compute_study_metrics(result_df, orig_stats)
    
    # ===== (1) FALSE POSITIVE RATE =====
    st.markdown("### 📊 (1) False Positive Rate")
    st.markdown("*Proportion of RDP-inferred events that could not be matched to SANTA events*")
    st.latex(r"FPR = \frac{\text{Unmatched RDP}}{\text{Total Original RDP}}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("False Positive Rate", f"{metrics['false_positive_rate']:.3%}")
    with col2:
        st.metric("Total Original RDP", f"{metrics['total_original_rdp']:,}")
    with col3:
        st.metric("Matched RDP", f"{metrics['matched_rdp']:,}")
    with col4:
        st.metric("Unmatched RDP (FPs)", f"{metrics['unmatched_rdp']:,}")
    
    st.info(f"**Calculation:** {metrics['unmatched_rdp']} / {metrics['total_original_rdp']} = {metrics['false_positive_rate']:.3%}")
    
    # ===== (2) RECOMBINANT ACCURACY =====
    st.markdown("---")
    st.markdown("### 🧬 (2) Recombinant Identification Accuracy")
    st.markdown("*Step 7 = parental misidentification*")

    st.latex(r"\text{Accuracy} = 1 - \frac{\text{Step 7 Rows}}{\text{Total Matched RDP}}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Matched RDP", f"{metrics['total_matched_rdp']:,}")
    with col2:
        st.metric("✅ Correct Recombinant", f"{metrics['correct_recombinant']:,}")
    with col3:
        st.metric("❌ Step 7 Count", f"{metrics['step7_count']:,}")
    with col4:
        st.metric("Accuracy", f"{metrics['recombinant_accuracy']:.3%}")

    st.success(f"**Calculation:** 1 - ({metrics['step7_count']} / {metrics['total_matched_rdp']}) = 1 - {metrics['incorrect_parental_rate']:.3%} = **{metrics['recombinant_accuracy']:.3%}**")

    if metrics['total_matched_rdp'] > 0:
        fig = go.Figure(data=[go.Pie(
            labels=['✅ Correct Recombinant', '❌ Step 7 (Incorrect Parental)'],
            values=[metrics['correct_recombinant'], metrics['step7_count']],
            marker_colors=['#11998e', '#f5576c']
        )])
        fig.update_layout(
            title=f"Recombinant Identification<br><sub>Step 7 rows = {metrics['step7_count']} | Total Matched RDP = {metrics['total_matched_rdp']:,}</sub>",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True, key=f"recombinant_pie_{condition}_{model}")
    
    # ===== (3) BREAKPOINT DISTANCE =====
    st.markdown("---")
    st.markdown("### 📍 (3) Breakpoint Distance Analysis")
    st.markdown("*Circular distance between inferred and simulated breakpoints (genome = 10,000 bp)*")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Start Distance", f"{metrics['mean_start_distance']:.1f} bp" if metrics['mean_start_distance'] > 0 else 'N/A')
    with col2:
        st.metric("Mean End Distance", f"{metrics['mean_end_distance']:.1f} bp" if metrics['mean_end_distance'] > 0 else 'N/A')
    with col3:
        st.metric("Mean Breakpoint Dist", f"{metrics['mean_breakpoint_distance']:.1f} bp" if metrics['mean_breakpoint_distance'] > 0 else 'N/A')
    with col4:
        st.metric("Breakpoints Analyzed", f"{metrics['total_breakpoints_analyzed']:,}")
    
    bp_distances = analyze_breakpoint_distances(result_df)
    
    if bp_distances['total_distances']:
        fig = px.histogram(
            x=bp_distances['total_distances'],
            nbins=30,
            title="Breakpoint Distance Distribution (Circular)",
            labels={'x': 'Distance (bp)', 'y': 'Frequency'},
            color_discrete_sequence=['#11998e']
        )
        fig.update_layout(template="plotly_white", title_x=0.5)
        st.plotly_chart(fig, use_container_width=True, key=f"bp_hist_{condition}_{model}")
    
    # ===== SUMMARY BOX =====
    st.markdown("---")
    st.markdown("### 📋 Study Summary")
    
    summary_text = f"""
    <strong>Condition {condition} - Model {model}</strong><br><br>
    
    <strong>(1) False Positive Rate:</strong> {metrics['false_positive_rate']:.3%}<br>
    - Formula: Unmatched RDP / Total Original RDP<br>
    - {metrics['unmatched_rdp']:,} / {metrics['total_original_rdp']:,} = {metrics['false_positive_rate']:.3%}<br><br>
    
    <strong>(2) Recombinant Accuracy:</strong> {metrics['recombinant_accuracy']:.3%}<br>
    - Total matches: {metrics['total_matches']:,}<br>
    - Correct recombinant (all except Step 7): {metrics['correct_recombinant']:,}<br>
    - Incorrect parental (Step 7): {metrics['incorrect_parental']:,}<br><br>
    
    <strong>(3) Breakpoint Distance:</strong> Mean = {metrics['mean_breakpoint_distance']:.1f} bp<br>
    - Circular distance between inferred and simulated breakpoints
    """
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                color: white; padding: 1.5rem; border-radius: 10px; font-size: 1rem;">
        {summary_text}
    </div>
    """, unsafe_allow_html=True)
    
    return metrics


def display_study_summary_all_conditions():
    """Display study metrics summary for all 9 conditions."""
    
    st.markdown("*Comparison of key metrics across all 9 simulation conditions*")
    
    summary_df = compute_study_summary_all_conditions()
    
    if not summary_df.empty:
        st.dataframe(summary_df, use_container_width=True, height=500)
        
        # Create comparison visualizations
        st.markdown("---")
        st.subheader("📊 Cross-Condition Comparison")
        
        # Prepare data for plotting
        plot_data = []
        for _, row in summary_df.iterrows():
            try:
                fpr = float(str(row['FPR']).strip('%')) / 100 if 'FPR' in row else 0
                accuracy = float(str(row['Accuracy']).strip('%')) / 100 if 'Accuracy' in row else 0
            except:
                fpr = 0
                accuracy = 0
            
            plot_data.append({
                'Condition': f"{row['Condition']} ({row['Model']})",
                'Condition_Only': row['Condition'],
                'Model': row['Model'],
                'FPR': fpr,
                'Accuracy': accuracy,
            })
        
        plot_df = pd.DataFrame(plot_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                plot_df,
                x='Condition',
                y='FPR',
                color='Model',
                barmode='group',
                title="False Positive Rate by Condition",
                labels={'FPR': 'False Positive Rate'}
            )
            fig.update_layout(template="plotly_white", title_x=0.5, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True, key="study_fpr_bar")
        
        with col2:
            fig = px.bar(
                plot_df,
                x='Condition',
                y='Accuracy',
                color='Model',
                barmode='group',
                title="Recombinant Accuracy by Condition",
                labels={'Accuracy': 'Accuracy'}
            )
            fig.update_layout(template="plotly_white", title_x=0.5, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True, key="study_accuracy_bar")
        
        # Download button
        csv = summary_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Study Summary (All Conditions)",
            data=csv,
            file_name="study_metrics_all_conditions.csv",
            mime="text/csv"
        )


# ============================================================================
# EXISTING DISPLAY FUNCTIONS
# ============================================================================

def display_original_analysis(original_df: pd.DataFrame, condition: str, model: str):
    """Display original dataset analysis split by tool."""
    st.markdown("---")
    st.subheader("📋 Original Dataset Analysis (Split by Tool)")
    
    if original_df is None:
        st.warning("No original dataset available.")
        return
    
    stats = analyze_original_by_tool(original_df)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", stats['total_rows'])
    with col2:
        st.metric("Unique Tags", stats['tag_count'])
    with col3:
        st.metric("Total RDP Events", f"{stats['total_rdp_events']:,}")
    with col4:
        st.metric("Total SANTA Events", f"{stats['total_santa_events']:,}")
    
    # Tag statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tags with RDP", len(stats['tags_with_rdp']))
    with col2:
        st.metric("Tags with SANTA", len(stats['tags_with_santa']))
    with col3:
        st.metric("Tags with Both", len(stats['tags_with_both']))
    with col4:
        st.metric("Tags with Neither", len(stats['tags_with_neither']))
    
    # Warnings for zero events
    if stats['zero_rdp_tags']:
        st.warning(f"⚠️ **{len(stats['zero_rdp_tags'])} tags have ZERO RDP events**")
    if stats['zero_santa_tags']:
        st.warning(f"⚠️ **{len(stats['zero_santa_tags'])} tags have ZERO SANTA events**")
    
    # Missing tags
    if stats.get('missing_tags'):
        missing = stats['missing_tags']
        st.info(f"📋 **{len(missing)} tags missing from original dataset**")
        if len(missing) <= 20:
            st.write(f"Missing tags: {missing}")
    
    # Per-tag summary table
    if stats['per_tag']:
        with st.expander("📊 Per-Tag Event Summary (Original Dataset)"):
            tag_data = []
            for tag, data in stats['per_tag'].items():
                tag_data.append({
                    'Tag': tag,
                    'RDP Count': data['rdp_count'],
                    'Santa Count': data['santa_count'],
                    'Total': data['total'],
                    'Has RDP': '✅' if data['has_rdp'] else '❌',
                    'Has Santa': '✅' if data['has_santa'] else '❌',
                })
            
            tag_df = pd.DataFrame(tag_data)
            tag_df = tag_df.sort_values('Total', ascending=False)
            st.dataframe(tag_df, use_container_width=True, height=400)
            
            csv = tag_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Original Per-Tag Summary",
                data=csv,
                file_name=f"{condition}_{model}_original_per_tag.csv",
                mime="text/csv"
            )




def display_comparison_analysis(original_df: pd.DataFrame, result_df: pd.DataFrame, 
                                condition: str, model: str):
    """Display combined comparison analysis."""
    st.markdown("---")
    st.subheader("🔬 Combined Comparison Analysis")
    
    # Analyze original (if available)
    if original_df is not None:
        orig_stats = analyze_original_by_tool(original_df)
    else:
        orig_stats = {
            'per_tag': {},
            'total_rdp_events': 0,
            'total_santa_events': 0,
            'total_events': 0,
            'tag_count': 0,
            'missing_tags': list(range(1, 101)),
            'zero_rdp_tags': [],
            'zero_santa_tags': [],
            'total_rows': 0,
            'tags_with_rdp': [],
            'tags_with_santa': [],
            'tags_with_both': [],
            'tags_with_neither': [],
        }
    
    # Extract result tag stats
    result_tag_df = extract_tag_level_stats(result_df)
    
    # Create comparison
    comparison_df = compute_comparison_stats(orig_stats, result_tag_df)
    
    if comparison_df.empty:
        st.warning("Could not create comparison analysis.")
        return
    
    # Compute totals - NOW USES orig_stats for true totals
    totals = compute_summary_totals(comparison_df, orig_stats)
    
    # Get values from totals (includes ALL original tags)
    total_orig_rdp = totals.get('total_original_rdp', 0)
    total_orig_santa = totals.get('total_original_santa', 0)
    total_matched_rdp = totals.get('total_result_rdp', 0)
    total_matched_santa = totals.get('total_result_santa', 0)
    total_remaining_rdp = totals.get('total_unmatched_rdp', 0)
    total_remaining_santa = totals.get('total_unmatched_santa', 0)
    
    # Display summary metrics
    st.markdown("#### 📈 Summary Totals (Includes ALL Original Tags)")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Original RDP", f"{total_orig_rdp:,}")
    with col2:
        st.metric("Total Original Santa", f"{total_orig_santa:,}")
    with col3:
        st.metric("Total Original Events", f"{total_orig_rdp + total_orig_santa:,}")
    with col4:
        st.metric("Tags in Results", f"{totals.get('tags_in_results', 0)}/{totals.get('tags_in_original', 0)}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Matched RDP", f"{total_matched_rdp:,}")
    with col2:
        st.metric("Total Matched Santa", f"{total_matched_santa:,}")
    with col3:
        st.metric("Total Matched Events", f"{total_matched_rdp + total_matched_santa:,}")
    with col4:
        match_rate = (total_matched_rdp + total_matched_santa) / (total_orig_rdp + total_orig_santa) * 100 if (total_orig_rdp + total_orig_santa) > 0 else 0
        st.metric("Overall Match Rate", f"{match_rate:.1f}%")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Remaining RDP", f"{total_remaining_rdp:,}")
    with col2:
        st.metric("Remaining Santa", f"{total_remaining_santa:,}")
    with col3:
        st.metric("Total Remaining", f"{total_remaining_rdp + total_remaining_santa:,}")
    with col4:
        fpr = total_remaining_rdp / total_matched_rdp * 100 if total_matched_rdp > 0 else 0
        st.metric("False Positive Rate", f"{fpr:.1f}%")
    
    # Show missing tags warning with count of missing events
    missing_list = get_missing_tags_list(comparison_df)
    if missing_list.get('missing_in_results'):
        missing = missing_list['missing_in_results']
        
        # Calculate missing events from those tags
        missing_rdp = 0
        missing_santa = 0
        for tag in missing:
            if tag in orig_stats.get('per_tag', {}):
                missing_rdp += orig_stats['per_tag'][tag].get('rdp_count', 0)
                missing_santa += orig_stats['per_tag'][tag].get('santa_count', 0)
        
        st.warning(f"⚠️ **{len(missing)} tags missing in results** (Missing RDP: {missing_rdp}, Missing Santa: {missing_santa})")
        if len(missing) <= 10:
            st.write(f"Missing tags: {missing}")
    
    # Show zero event tags
    if totals.get('zero_rdp_tags'):
        st.info(f"📊 **{len(totals['zero_rdp_tags'])} tags have ZERO RDP events**")
    if totals.get('zero_santa_tags'):
        st.info(f"📊 **{len(totals['zero_santa_tags'])} tags have ZERO SANTA events**")
    
    # Comparison table
    st.markdown("#### 📊 Per-Tag Comparison")
    
    if 'Match_Status' in comparison_df.columns:
        status_counts = comparison_df['Match_Status'].value_counts()
        st.write("**Match Status Summary:**")
        cols = st.columns(min(len(status_counts), 5))
        for i, (status, count) in enumerate(status_counts.items()):
            if i < 5:
                with cols[i]:
                    st.metric(status, count)
    
    st.dataframe(comparison_df, use_container_width=True, height=400)
    
    # Download buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        csv_comparison = comparison_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Comparison Table",
            data=csv_comparison,
            file_name=f"{condition}_{model}_comparison.csv",
            mime="text/csv",
            key=f"download_comp_{condition}_{model}"
        )
    with col2:
        if not result_tag_df.empty:
            csv_tag = result_tag_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Per-Tag Table",
                data=csv_tag,
                file_name=f"{condition}_{model}_per_tag.csv",
                mime="text/csv",
                key=f"download_tag_{condition}_{model}"
            )
    with col3:
        summary_data = {
            'Metric': [
                'Total Original RDP', 'Total Original Santa', 'Total Original Events',
                'Total Matched RDP', 'Total Matched Santa', 'Total Matched Events',
                'Remaining RDP', 'Remaining Santa', 'Total Remaining',
                'Overall Match Rate (%)', 'False Positive Rate (%)',
                'Tags in Original', 'Tags in Results', 'Missing Tags'
            ],
            'Value': [
                total_orig_rdp, total_orig_santa, total_orig_rdp + total_orig_santa,
                total_matched_rdp, total_matched_santa, total_matched_rdp + total_matched_santa,
                total_remaining_rdp, total_remaining_santa, total_remaining_rdp + total_remaining_santa,
                f"{match_rate:.1f}", f"{fpr:.1f}",
                totals.get('tags_in_original', 0), totals.get('tags_in_results', 0), 
                len(missing_list.get('missing_in_results', []))
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        csv_summary = summary_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Summary Totals",
            data=csv_summary,
            file_name=f"{condition}_{model}_totals.csv",
            mime="text/csv",
            key=f"download_summary_{condition}_{model}"
        )


def display_css():
    """Display custom CSS for the app."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * { font-family: 'Inter', sans-serif; }
        
        .main-header {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .main-header h1 {
            color: white;
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        .main-header p {
            color: #e0e7ff;
            font-size: 1.1rem;
            margin-top: 0.5rem;
        }
        
        .condition-card {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            margin: 0.5rem 0;
            border-left: 4px solid #2a5298;
        }
        
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #f0f2f6;
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 500;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        [data-testid="stDataFrame"] {
            max-height: 500px !important;
            overflow-y: auto !important;
        }
        
        .download-section {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)