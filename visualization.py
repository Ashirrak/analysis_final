"""Visualization functions using Plotly."""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def create_tag_comparison_bar_chart(comparison_df: pd.DataFrame, top_n: int = 20):
    """Create bar chart comparing original vs result counts per tag."""
    if comparison_df.empty:
        return None
    
    # Sort by original total and take top N
    df_sorted = comparison_df.nlargest(top_n, 'Original_Total')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Original RDP',
        x=df_sorted['Tag'].astype(str),
        y=df_sorted['Original_RDP'],
        marker_color='#667eea'
    ))
    
    fig.add_trace(go.Bar(
        name='Original Santa',
        x=df_sorted['Tag'].astype(str),
        y=df_sorted['Original_Santa'],
        marker_color='#11998e'
    ))
    
    if 'Result_RDP' in df_sorted.columns:
        fig.add_trace(go.Bar(
            name='Result RDP',
            x=df_sorted['Tag'].astype(str),
            y=df_sorted['Result_RDP'],
            marker_color='#764ba2'
        ))
    
    fig.update_layout(
        title=f"Top {top_n} Tags - Event Count Comparison",
        xaxis_title="Tag",
        yaxis_title="Event Count",
        barmode='group',
        template="plotly_white",
        height=500
    )
    
    return fig


def create_match_rate_scatter(comparison_df: pd.DataFrame):
    """Create scatter plot of RDP vs Santa match rates."""
    if comparison_df.empty:
        return None
    
    df_valid = comparison_df.dropna(subset=['RDP_Match_Rate', 'Santa_Match_Rate'])
    
    if df_valid.empty:
        return None
    
    fig = px.scatter(
        df_valid,
        x='RDP_Match_Rate',
        y='Santa_Match_Rate',
        hover_data=['Tag'],
        title="RDP vs Santa Match Rates per Tag",
        labels={'RDP_Match_Rate': 'RDP Match Rate', 'Santa_Match_Rate': 'Santa Match Rate'}
    )
    
    # Add diagonal line
    max_val = max(df_valid['RDP_Match_Rate'].max(), df_valid['Santa_Match_Rate'].max())
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        name='y = x',
        line=dict(color='gray', dash='dash')
    ))
    
    fig.update_layout(template="plotly_white", height=400)
    
    return fig


def create_summary_metrics_chart(totals: dict):
    """Create a summary metrics indicator chart."""
    fig = go.Figure()
    
    metrics = [
        ('False Positive Rate', totals.get('false_positive_rate', 0) * 100, '%'),
        ('RDP Match Rate', totals.get('avg_rdp_match_rate', 0) * 100, '%'),
        ('Santa Match Rate', totals.get('avg_santa_match_rate', 0) * 100, '%'),
        ('Tags in Results', totals.get('tags_in_results', 0) / max(totals.get('total_tags', 1), 1) * 100, '%'),
    ]
    
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=metrics[0][1],
        title={'text': metrics[0][0]},
        gauge={'axis': {'range': [0, 100]}},
        domain={'row': 0, 'column': 0}
    ))
    
    # Use bar chart instead for simplicity
    fig = px.bar(
        x=[m[0] for m in metrics],
        y=[m[1] for m in metrics],
        title="Key Performance Metrics",
        labels={'x': 'Metric', 'y': 'Value (%)'},
        color_discrete_sequence=['#2a5298']
    )
    fig.update_layout(template="plotly_white", height=400)
    
    return fig