"""Tab: Model Chain Comparison - Cross-Model Chain Analysis"""

import streamlit as st
import pandas as pd
import numpy as np
import re
from config import CONDITIONS_MAP, MODELS
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render(all_data, summary, conditions_map=CONDITIONS_MAP, models=MODELS):
    """Render Model Chain Comparison tab"""
    
    st.markdown("## 🔗 Cross-Model Chain Analysis")
    st.caption("Advanced comparison of chain detection capabilities across DT, LR, NN models")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # ===== COMPUTE CHAIN DATA FOR ALL MODELS =====
    all_chain_data = compute_all_chain_data(all_data, conditions_map, models)
    
    # ===== SECTION 1: GLOBAL OVERVIEW =====
    st.markdown("### 🌐 Global Chain Detection Overview")
    
    render_global_chain_summary(all_chain_data, models)
    
    # ===== SECTION 2: COMMON CHAINS ANALYSIS =====
    st.markdown("---")
    st.markdown("### 🤝 Common Chains Across Models")
    st.caption("Chains detected by all three models with same (Tag, RDP_Event_ID)")
    
    render_common_chains_analysis(all_chain_data, conditions_map, models)
    
    # ===== SECTION 3: UNIQUE CHAIN DETECTION =====
    st.markdown("---")
    st.markdown("### 🎯 Unique Chain Detection")
    st.caption("Chains found exclusively by each model")
    
    render_unique_chains_analysis(all_chain_data, conditions_map, models)
    
    # ===== SECTION 4: PAIRWISE CHAIN OVERLAP =====
    st.markdown("---")
    st.markdown("### 🔄 Pairwise Chain Overlap")
    st.caption("Chains shared between two models but missed by the third")
    
    render_pairwise_chain_overlap(all_chain_data, conditions_map, models)
    
    # ===== SECTION 5: CHAIN COMPLEXITY ANALYSIS =====
    st.markdown("---")
    st.markdown("### 📊 Chain Complexity & Model Performance")
    st.caption("How chain length affects model agreement")
    
    render_chain_complexity_analysis(all_chain_data, models)
    
    # ===== SECTION 6: THESIS-READY SUMMARY =====
    #st.markdown("---")
    #st.markdown("### 📝 Thesis-Ready Interpretation")
    
    #render_thesis_interpretation(all_chain_data, models)


def compute_all_chain_data(all_data, conditions_map, models):
    """
    Extract chain events from all models and conditions.
    Returns dict: {condition: {model: DataFrame_of_chains}}
    """
    all_chain_data = {}
    
    for condition in conditions_map.keys():
        all_chain_data[condition] = {}
        
        for model in models:
            original_df, result_df = all_data[condition][model]
            
            if result_df is None or result_df.empty:
                all_chain_data[condition][model] = pd.DataFrame()
                continue
            
            # Extract chains
            chains = extract_chain_events(result_df, condition, model)
            all_chain_data[condition][model] = chains
    
    return all_chain_data


def extract_chain_events(df: pd.DataFrame, condition: str, model: str) -> pd.DataFrame:
    """
    Extract chain events from a result dataframe.
    Returns dataframe with chain-specific columns.
    """
    df_copy = df.copy()
    
    # Find RDP_Event_ID column
    rdp_id_col = None
    for col in ['RDP_Event_ID', 'rdp_event_id', 'Event_ID']:
        if col in df_copy.columns:
            rdp_id_col = col
            break
    
    if rdp_id_col is None:
        return pd.DataFrame()
    
    # Find Tag column
    tag_col = None
    for col in ['Tag', 'tag', 'Santa_Tag']:
        if col in df_copy.columns:
            tag_col = col
            break
    
    # Find Source_Tab column
    source_col = None
    for col in ['Source_Tab', 'source_tab', 'Step', 'step']:
        if col in df_copy.columns:
            source_col = col
            break
    
    # Identify chains (comma-separated IDs)
    def is_chain(val):
        if pd.isna(val):
            return False
        ids = [x.strip() for x in str(val).split(',') if x.strip()]
        return len(ids) > 1
    
    def get_chain_ids(val):
        if pd.isna(val):
            return []
        return [x.strip() for x in str(val).split(',') if x.strip()]
    
    chain_mask = df_copy[rdp_id_col].apply(is_chain)
    chains_df = df_copy[chain_mask].copy()
    
    if chains_df.empty:
        return pd.DataFrame()
    
    # Add chain-specific columns
    chains_df['chain_ids'] = chains_df[rdp_id_col].apply(get_chain_ids)
    chains_df['chain_length'] = chains_df['chain_ids'].apply(len)
    chains_df['condition'] = condition
    chains_df['model'] = model
    
    # Create unique chain identifier: Tag + sorted RDP IDs
    if tag_col:
        chains_df['tag'] = chains_df[tag_col].astype(str)
    else:
        chains_df['tag'] = 'unknown'
    
    # Create normalized chain signature for comparison
    chains_df['chain_signature'] = chains_df.apply(
        lambda row: f"T{row['tag']}_{'_'.join(sorted(row['chain_ids']))}", 
        axis=1
    )
    
    # Add step info if available
    if source_col:
        chains_df['source_tab'] = chains_df[source_col]
        chains_df['step_num'] = chains_df[source_col].apply(extract_step_number)
    
    return chains_df


def extract_step_number(val):
    """Extract step number from Source_Tab string."""
    if pd.isna(val):
        return 'Unknown'
    val_str = str(val)
    match = re.search(r'Step\s*(\d+)', val_str, re.IGNORECASE)
    if match:
        return f"Step {match.group(1)}"
    match = re.search(r'(\d+)', val_str)
    if match:
        return f"Step {match.group(1)}"
    return 'Unknown'


# ============================================================================
# SECTION 1: GLOBAL OVERVIEW
# ============================================================================

def render_global_chain_summary(all_chain_data, models):
    """Display global chain detection statistics."""
    
    # Aggregate stats
    model_stats = {}
    total_chains_all = 0
    
    for model in models:
        model_chain_count = 0
        model_avg_length = []
        model_conditions = {}
        
        for condition, model_data in all_chain_data.items():
            chains = model_data.get(model, pd.DataFrame())
            count = len(chains)
            model_chain_count += count
            
            if not chains.empty:
                avg_len = chains['chain_length'].mean()
                model_avg_length.extend(chains['chain_length'].tolist())
                model_conditions[condition] = {
                    'count': count,
                    'avg_length': avg_len,
                    'max_length': chains['chain_length'].max()
                }
        
        total_chains_all += model_chain_count
        
        model_stats[model] = {
            'total_chains': model_chain_count,
            'avg_chain_length': np.mean(model_avg_length) if model_avg_length else 0,
            'max_chain_length': max(model_avg_length) if model_avg_length else 0,
            'conditions': model_conditions
        }
    
    # Display summary cards
    col1, col2, col3 = st.columns(3)
    
    model_colors = {'DT': '#4285f4', 'LR': '#34a853', 'NN': '#fbbc04'}
    
    for i, (model, stats) in enumerate(model_stats.items()):
        with [col1, col2, col3][i]:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,{model_colors[model]},rgba(0,0,0,0.3));
                        color:white;padding:1.5rem;border-radius:14px;text-align:center;">
                <div style="font-size:0.8rem;opacity:0.9;text-transform:uppercase;letter-spacing:2px;">
                    {model} Total Chains</div>
                <div style="font-size:2.5rem;font-weight:800;">{stats['total_chains']:,}</div>
                <div style="font-size:0.8rem;opacity:0.8;margin-top:0.5rem;">
                    Avg Length: {stats['avg_chain_length']:.1f}</div>
                <div style="font-size:0.7rem;opacity:0.6;">
                    Max Length: {stats['max_chain_length']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Comparison chart
    st.markdown("---")
    st.markdown("#### Chain Detection Comparison")
    
    comparison_data = []
    for model, stats in model_stats.items():
        for condition, cond_stats in stats['conditions'].items():
            comparison_data.append({
                'Condition': condition,
                'Model': model,
                'Chain Count': cond_stats['count'],
                'Avg Length': cond_stats['avg_length']
            })
    
    comp_df = pd.DataFrame(comparison_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            comp_df, x='Condition', y='Chain Count', color='Model',
            barmode='group',
            title="Chain Count by Condition and Model",
            color_discrete_map=model_colors,
            text='Chain Count'
        )
        fig.update_traces(textposition='outside', textfont=dict(size=11))
        fig.update_layout(template="plotly_white", title_x=0.5, xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig, use_container_width=True, key="global_chain_count")
    
    with col2:
        fig = px.bar(
            comp_df, x='Condition', y='Avg Length', color='Model',
            barmode='group',
            title="Average Chain Length by Condition and Model",
            color_discrete_map=model_colors,
            text=comp_df['Avg Length'].apply(lambda x: f'{x:.1f}')
        )
        fig.update_traces(textposition='outside', textfont=dict(size=11))
        fig.update_layout(template="plotly_white", title_x=0.5, xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig, use_container_width=True, key="global_chain_length")
    
    # Heatmap
    st.markdown("#### Chain Detection Heatmap")
    pivot_count = comp_df.pivot(index='Condition', columns='Model', values='Chain Count')
    
    fig = px.imshow(
        pivot_count, text_auto=True, aspect='auto',
        title="Chain Count Heatmap (darker = more chains)",
        color_continuous_scale='Blues',
        labels={'color': 'Chain Count'}
    )
    fig.update_layout(height=350, title_x=0.5)
    st.plotly_chart(fig, use_container_width=True, key="global_chain_heatmap")


# ============================================================================
# SECTION 2: COMMON CHAINS ANALYSIS
# ============================================================================

def render_common_chains_analysis(all_chain_data, conditions_map, models):
    """Analyze chains detected by all three models."""
    
    st.markdown("#### Chains Detected by ALL Three Models")
    st.caption("Same (Tag, RDP_Event_ID) chain signature found in DT, LR, and NN")
    
    common_summary = []
    all_common_details = []
    
    for condition in conditions_map.keys():
        # Get chain signatures for each model
        dt_signatures = set()
        lr_signatures = set()
        nn_signatures = set()
        
        dt_df = all_chain_data[condition].get('DT', pd.DataFrame())
        lr_df = all_chain_data[condition].get('LR', pd.DataFrame())
        nn_df = all_chain_data[condition].get('NN', pd.DataFrame())
        
        if 'chain_signature' in dt_df.columns:
            dt_signatures = set(dt_df['chain_signature'])
        if 'chain_signature' in lr_df.columns:
            lr_signatures = set(lr_df['chain_signature'])
        if 'chain_signature' in nn_df.columns:
            nn_signatures = set(nn_df['chain_signature'])
        
        # Find common chains
        common_sigs = dt_signatures & lr_signatures & nn_signatures
        
        # Also track partial overlaps
        any_model_count = len(dt_signatures | lr_signatures | nn_signatures)
        
        common_summary.append({
            'Condition': condition,
            'DT Chains': len(dt_signatures),
            'LR Chains': len(lr_signatures),
            'NN Chains': len(nn_signatures),
            'Common (All 3)': len(common_sigs),
            'Union (Any Model)': any_model_count,
            'Agreement Rate': f"{(len(common_sigs)/any_model_count*100):.1f}%" if any_model_count > 0 else "0%"
        })
        
        # Get details for common chains
        if common_sigs and not dt_df.empty:
            common_details = dt_df[dt_df['chain_signature'].isin(common_sigs)][
                ['tag', 'chain_ids', 'chain_length', 'source_tab', 'chain_signature']
            ].copy()
            common_details['condition'] = condition
            all_common_details.append(common_details)
    
    # Display summary table
    common_summary_df = pd.DataFrame(common_summary)
    st.dataframe(common_summary_df, use_container_width=True, height=300)
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            common_summary_df, x='Condition',
            y=['Common (All 3)', 'Union (Any Model)'],
            barmode='group',
            title="Common vs Total Unique Chains",
            labels={'value': 'Count', 'variable': 'Type'},
            color_discrete_map={'Common (All 3)': '#34a853', 'Union (Any Model)': '#4285f4'}
        )
        fig.update_layout(template="plotly_white", title_x=0.5, xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig, use_container_width=True, key="common_chains_bar")
    
    with col2:
        fig = px.line(
            common_summary_df, x='Condition',
            y=['DT Chains', 'LR Chains', 'NN Chains', 'Common (All 3)'],
            title="Chain Detection Trends",
            markers=True,
            color_discrete_map={
                'DT Chains': '#4285f4', 'LR Chains': '#34a853',
                'NN Chains': '#fbbc04', 'Common (All 3)': '#ea4335'
            }
        )
        fig.update_layout(template="plotly_white", title_x=0.5, xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig, use_container_width=True, key="common_chains_trend")
    
    # Detailed common chains table
    if all_common_details:
        st.markdown("---")
        st.markdown("#### Common Chains Detail")
        all_common_df = pd.concat(all_common_details, ignore_index=True)
        st.caption(f"Total common chains across all conditions: {len(all_common_df)}")
        st.dataframe(all_common_df, use_container_width=True, height=400)
        
        # Download
        csv = all_common_df.to_csv(index=False)
        st.download_button(
            "📥 Download Common Chains",
            csv,
            "common_chains_all_models.csv",
            key="dl_common_chains"
        )
    
    # ===== VENN DIAGRAM FOR ALL CONDITIONS COMBINED =====
    st.markdown("---")
    st.markdown("#### Overall Chain Overlap (Venn Diagram)")
    
    try:
        from matplotlib_venn import venn3
        import matplotlib.pyplot as plt
        
        # Combine all signatures
        all_dt_sigs = set()
        all_lr_sigs = set()
        all_nn_sigs = set()
        
        for condition in conditions_map.keys():
            for model, sig_set, df in [
                ('DT', all_dt_sigs, all_chain_data[condition].get('DT', pd.DataFrame())),
                ('LR', all_lr_sigs, all_chain_data[condition].get('LR', pd.DataFrame())),
                ('NN', all_nn_sigs, all_chain_data[condition].get('NN', pd.DataFrame()))
            ]:
                if 'chain_signature' in df.columns:
                    sig_set.update(df['chain_signature'])
        
        # Create figure with unique identifier
        fig, ax = plt.subplots(figsize=(8, 8))
        venn3([all_dt_sigs, all_lr_sigs, all_nn_sigs],
              set_labels=('DT', 'LR', 'NN'),
              set_colors=('#4285f4', '#34a853', '#fbbc04'))
        ax.set_title('Chain Overlap Across All Models (All Conditions)', fontsize=14, fontweight='bold')
        
        # FIX: Remove key parameter from st.pyplot()
        st.pyplot(fig)
        plt.close()
        
        # Also display counts as text
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("DT Only", len(all_dt_sigs - all_lr_sigs - all_nn_sigs))
        with col_b:
            st.metric("LR Only", len(all_lr_sigs - all_dt_sigs - all_nn_sigs))
        with col_c:
            st.metric("NN Only", len(all_nn_sigs - all_dt_sigs - all_lr_sigs))
        
        col_d, col_e = st.columns(2)
        with col_d:
            st.metric("All Three", len(all_dt_sigs & all_lr_sigs & all_nn_sigs))
        with col_e:
            st.metric("Total Unique", len(all_dt_sigs | all_lr_sigs | all_nn_sigs))
            
    except Exception as e:
        # More detailed error message
        st.warning(f"Venn diagram could not be rendered: {str(e)}")
        
        # Fallback: Display overlap counts as a table
        st.markdown("**Chain Overlap Counts (All Conditions Combined):**")
        
        overlap_data = pd.DataFrame({
            'Category': [
                'DT Only', 'LR Only', 'NN Only',
                'DT & LR (not NN)', 'DT & NN (not LR)', 'LR & NN (not DT)',
                'All Three', 'Total Unique'
            ],
            'Count': [
                len(all_dt_sigs - all_lr_sigs - all_nn_sigs),
                len(all_lr_sigs - all_dt_sigs - all_nn_sigs),
                len(all_nn_sigs - all_dt_sigs - all_lr_sigs),
                len((all_dt_sigs & all_lr_sigs) - all_nn_sigs),
                len((all_dt_sigs & all_nn_sigs) - all_lr_sigs),
                len((all_lr_sigs & all_nn_sigs) - all_dt_sigs),
                len(all_dt_sigs & all_lr_sigs & all_nn_sigs),
                len(all_dt_sigs | all_lr_sigs | all_nn_sigs)
            ]
        })
        st.dataframe(overlap_data, use_container_width=True, hide_index=True)

# ============================================================================
# SECTION 3: UNIQUE CHAIN DETECTION
# ============================================================================

def render_unique_chains_analysis(all_chain_data, conditions_map, models):
    """Analyze chains detected exclusively by each model."""
    
    st.markdown("#### Chains Unique to Each Model")
    st.caption("Chains found by only one model")
    
    unique_summary = []
    all_unique_details = []
    
    for condition in conditions_map.keys():
        dt_sigs = set(all_chain_data[condition].get('DT', pd.DataFrame()).get('chain_signature', []))
        lr_sigs = set(all_chain_data[condition].get('LR', pd.DataFrame()).get('chain_signature', []))
        nn_sigs = set(all_chain_data[condition].get('NN', pd.DataFrame()).get('chain_signature', []))
        
        # Unique to each model
        dt_unique = dt_sigs - lr_sigs - nn_sigs
        lr_unique = lr_sigs - dt_sigs - nn_sigs
        nn_unique = nn_sigs - dt_sigs - lr_sigs
        
        unique_summary.append({
            'Condition': condition,
            'DT Unique': len(dt_unique),
            'LR Unique': len(lr_unique),
            'NN Unique': len(nn_unique),
            'DT Uniqueness %': f"{(len(dt_unique)/len(dt_sigs)*100):.1f}%" if dt_sigs else "0%",
            'LR Uniqueness %': f"{(len(lr_unique)/len(lr_sigs)*100):.1f}%" if lr_sigs else "0%",
            'NN Uniqueness %': f"{(len(nn_unique)/len(nn_sigs)*100):.1f}%" if nn_sigs else "0%"
        })
    
    unique_summary_df = pd.DataFrame(unique_summary)
    st.dataframe(unique_summary_df, use_container_width=True, height=300)
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            unique_summary_df, x='Condition',
            y=['DT Unique', 'LR Unique', 'NN Unique'],
            barmode='group',
            title="Unique Chain Detection by Model",
            color_discrete_map={'DT Unique': '#4285f4', 'LR Unique': '#34a853', 'NN Unique': '#fbbc04'},
            text_auto=True
        )
        fig.update_layout(template="plotly_white", title_x=0.5, xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig, use_container_width=True, key="unique_chains_bar")
    
    with col2:
        # Calculate totals
        total_dt_unique = unique_summary_df['DT Unique'].sum()
        total_lr_unique = unique_summary_df['LR Unique'].sum()
        total_nn_unique = unique_summary_df['NN Unique'].sum()
        
        pie_data = pd.DataFrame({
            'Model': ['DT', 'LR', 'NN'],
            'Unique Chains': [total_dt_unique, total_lr_unique, total_nn_unique]
        })
        
        fig = px.pie(
            pie_data, values='Unique Chains', names='Model',
            title="Share of Unique Chain Detection",
            color='Model',
            color_discrete_map={'DT': '#4285f4', 'LR': '#34a853', 'NN': '#fbbc04'},
            hole=0.4
        )
        fig.update_traces(textinfo='value+percent')
        fig.update_layout(template="plotly_white", title_x=0.5, height=400)
        st.plotly_chart(fig, use_container_width=True, key="unique_chains_pie")
    
    # ===== MODEL SENSITIVITY SCORE =====
    st.markdown("---")
    st.markdown("#### 🏆 Model Chain Sensitivity Score")
    st.caption("Higher = better at finding unique chains others miss")
    
    total_dt = sum(len(set(all_chain_data[c].get('DT', pd.DataFrame()).get('chain_signature', []))) 
                   for c in conditions_map.keys())
    total_lr = sum(len(set(all_chain_data[c].get('LR', pd.DataFrame()).get('chain_signature', []))) 
                   for c in conditions_map.keys())
    total_nn = sum(len(set(all_chain_data[c].get('NN', pd.DataFrame()).get('chain_signature', []))) 
                   for c in conditions_map.keys())
    
    sensitivity_data = pd.DataFrame({
        'Model': ['DT', 'LR', 'NN'],
        'Sensitivity Score': [
            (total_dt_unique / total_dt * 100) if total_dt > 0 else 0,
            (total_lr_unique / total_lr * 100) if total_lr > 0 else 0,
            (total_nn_unique / total_nn * 100) if total_nn > 0 else 0
        ],
        'Total Chains': [total_dt, total_lr, total_nn],
        'Unique Chains': [total_dt_unique, total_lr_unique, total_nn_unique]
    })
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(sensitivity_data, use_container_width=True)
    
    with col2:
        fig = px.bar(
            sensitivity_data, x='Model', y='Sensitivity Score',
            color='Model',
            color_discrete_map={'DT': '#4285f4', 'LR': '#34a853', 'NN': '#fbbc04'},
            text=sensitivity_data['Sensitivity Score'].apply(lambda x: f'{x:.1f}%'),
            title="Chain Sensitivity Score (Higher = Better at Unique Detection)"
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(template="plotly_white", title_x=0.5, height=350)
        st.plotly_chart(fig, use_container_width=True, key="sensitivity_score")


# ============================================================================
# SECTION 4: PAIRWISE CHAIN OVERLAP
# ============================================================================

def render_pairwise_chain_overlap(all_chain_data, conditions_map, models):
    """Analyze chains shared between pairs of models."""
    
    st.markdown("#### Pairwise Chain Overlap Analysis")
    st.caption("Chains detected by two models but missed by the third")
    
    pairwise_summary = []
    
    for condition in conditions_map.keys():
        dt_sigs = set(all_chain_data[condition].get('DT', pd.DataFrame()).get('chain_signature', []))
        lr_sigs = set(all_chain_data[condition].get('LR', pd.DataFrame()).get('chain_signature', []))
        nn_sigs = set(all_chain_data[condition].get('NN', pd.DataFrame()).get('chain_signature', []))
        
        # Pairwise overlaps
        dt_lr = dt_sigs & lr_sigs
        dt_nn = dt_sigs & nn_sigs
        lr_nn = lr_sigs & nn_sigs
        
        # Exclude chains found by all three
        all_three = dt_sigs & lr_sigs & nn_sigs
        dt_lr_only = dt_lr - all_three
        dt_nn_only = dt_nn - all_three
        lr_nn_only = lr_nn - all_three
        
        pairwise_summary.append({
            'Condition': condition,
            'DT & LR (not NN)': len(dt_lr_only),
            'DT & NN (not LR)': len(dt_nn_only),
            'LR & NN (not DT)': len(lr_nn_only),
            'All Three': len(all_three)
        })
    
    pairwise_df = pd.DataFrame(pairwise_summary)
    st.dataframe(pairwise_df, use_container_width=True, height=300)
    
    # Visualization
    fig = px.bar(
        pairwise_df, x='Condition',
        y=['DT & LR (not NN)', 'DT & NN (not LR)', 'LR & NN (not DT)', 'All Three'],
        barmode='stack',
        title="Chain Overlap Patterns by Condition",
        color_discrete_map={
            'DT & LR (not NN)': '#9b59b6',
            'DT & NN (not LR)': '#e67e22',
            'LR & NN (not DT)': '#1abc9c',
            'All Three': '#34495e'
        }
    )
    fig.update_layout(template="plotly_white", title_x=0.5, xaxis_tickangle=-45, height=400)
    st.plotly_chart(fig, use_container_width=True, key="pairwise_overlap_bar")
    
    # ===== MODEL AGREEMENT MATRIX =====
    st.markdown("---")
    st.markdown("#### Model Agreement Matrix (All Conditions Combined)")
    
    all_dt = set()
    all_lr = set()
    all_nn = set()
    
    for condition in conditions_map.keys():
        all_dt.update(all_chain_data[condition].get('DT', pd.DataFrame()).get('chain_signature', []))
        all_lr.update(all_chain_data[condition].get('LR', pd.DataFrame()).get('chain_signature', []))
        all_nn.update(all_chain_data[condition].get('NN', pd.DataFrame()).get('chain_signature', []))
    
    agreement_matrix = pd.DataFrame({
        'Model': ['DT', 'LR', 'NN'],
        'DT': [len(all_dt), len(all_dt & all_lr), len(all_dt & all_nn)],
        'LR': [len(all_lr & all_dt), len(all_lr), len(all_lr & all_nn)],
        'NN': [len(all_nn & all_dt), len(all_nn & all_lr), len(all_nn)]
    }).set_index('Model')
    
    fig = px.imshow(
        agreement_matrix, text_auto=True, aspect='auto',
        title="Chain Overlap Matrix (Intersection counts)",
        color_continuous_scale='YlOrRd',
        labels={'color': 'Shared Chains'}
    )
    fig.update_layout(height=350, title_x=0.5)
    st.plotly_chart(fig, use_container_width=True, key="agreement_matrix")


# ============================================================================
# SECTION 5: CHAIN COMPLEXITY ANALYSIS
# ============================================================================

def render_chain_complexity_analysis(all_chain_data, models):
    """Analyze how chain complexity (length) affects model agreement."""
    
    st.markdown("#### Chain Complexity vs Model Agreement")
    st.caption("Does chain length affect how many models detect it?")
    
    # Collect all chains with their detection status
    complexity_data = []
    
    for condition, model_data in all_chain_data.items():
        # Combine all signatures
        all_signatures = set()
        for model in models:
            df = model_data.get(model, pd.DataFrame())
            if 'chain_signature' in df.columns:
                all_signatures.update(df['chain_signature'])
        
        for sig in all_signatures:
            detected_by = []
            chain_length = 0
            tag = ''
            
            for model in models:
                df = model_data.get(model, pd.DataFrame())
                if not df.empty and 'chain_signature' in df.columns:
                    matching = df[df['chain_signature'] == sig]
                    if not matching.empty:
                        detected_by.append(model)
                        if chain_length == 0:
                            chain_length = matching['chain_length'].iloc[0]
                            tag = matching['tag'].iloc[0]
            
            complexity_data.append({
                'condition': condition,
                'chain_signature': sig,
                'tag': tag,
                'chain_length': chain_length,
                'detected_by': ', '.join(sorted(detected_by)),
                'num_models': len(detected_by),
                'detection_status': 'All 3' if len(detected_by) == 3 else 
                                   '2 Models' if len(detected_by) == 2 else '1 Model'
            })
    
    complexity_df = pd.DataFrame(complexity_data)
    
    if not complexity_df.empty:
        # Summary by detection status
        col1, col2 = st.columns(2)
        
        with col1:
            status_stats = complexity_df.groupby('detection_status').agg(
                count=('chain_signature', 'count'),
                avg_length=('chain_length', 'mean'),
                max_length=('chain_length', 'max')
            ).reset_index()
            
            st.write("**Detection Status vs Chain Length**")
            st.dataframe(status_stats, use_container_width=True)
        
        with col2:
            fig = px.box(
                complexity_df, x='detection_status', y='chain_length',
                color='detection_status',
                title="Chain Length Distribution by Detection Status",
                color_discrete_map={'All 3': '#34a853', '2 Models': '#fbbc04', '1 Model': '#ea4335'},
                points='all'
            )
            fig.update_layout(template="plotly_white", title_x=0.5, height=400)
            st.plotly_chart(fig, use_container_width=True, key="complexity_box")
        
        # Scatter: chain length vs number of models
        st.markdown("---")
        
        fig = px.scatter(
            complexity_df, x='chain_length', y='num_models',
            color='detection_status',
            size='chain_length',
            hover_data=['tag', 'condition', 'detected_by'],
            title="Chain Length vs Number of Detecting Models",
            color_discrete_map={'All 3': '#34a853', '2 Models': '#fbbc04', '1 Model': '#ea4335'}
        )
        # Add jitter for better visibility
        fig.update_traces(marker=dict(opacity=0.7))
        fig.update_layout(template="plotly_white", title_x=0.5, height=400,
                         yaxis=dict(tickmode='linear', tick0=1, dtick=1))
        st.plotly_chart(fig, use_container_width=True, key="complexity_scatter")
        
        # Correlation
        if len(complexity_df) > 2:
            correlation = complexity_df['chain_length'].corr(complexity_df['num_models'])
            st.info(f"Correlation (Chain Length vs Model Agreement): **{correlation:.3f}** " +
                   f"({'Positive' if correlation > 0 else 'Negative'}{' (strong)' if abs(correlation) > 0.5 else ''})")


# ============================================================================
# SECTION 6: THESIS INTERPRETATION
# ============================================================================

def render_thesis_interpretation(all_chain_data, models):
    """Generate thesis-ready interpretation of chain analysis."""
    
    st.markdown("#### Key Findings for Thesis Discussion")
    
    # Calculate key metrics
    model_stats = {}
    all_conditions = list(all_chain_data.keys())
    
    for model in models:
        total_chains = 0
        total_unique = 0
        all_sigs = set()
        unique_sigs = set()
        
        for condition in all_conditions:
            df = all_chain_data[condition].get(model, pd.DataFrame())
            if 'chain_signature' in df.columns:
                sigs = set(df['chain_signature'])
                all_sigs.update(sigs)
                total_chains += len(sigs)
        
        # Calculate unique chains
        other_models = [m for m in models if m != model]
        other_sigs = set()
        for other in other_models:
            for condition in all_conditions:
                df = all_chain_data[condition].get(other, pd.DataFrame())
                if 'chain_signature' in df.columns:
                    other_sigs.update(set(df['chain_signature']))
        
        unique_sigs = all_sigs - other_sigs
        
        model_stats[model] = {
            'total': len(all_sigs),
            'unique': len(unique_sigs),
            'uniqueness': f"{(len(unique_sigs)/len(all_sigs)*100):.1f}%" if all_sigs else "0%"
        }
    
    # Find best model for chain detection
    best_chain_model = max(model_stats, key=lambda m: model_stats[m]['total'])
    best_unique_model = max(model_stats, key=lambda m: model_stats[m]['unique'])
    
    # Find common chains percentage
    all_dt = set()
    all_lr = set()
    all_nn = set()
    for condition in all_conditions:
        for model, sig_set in [('DT', all_dt), ('LR', all_lr), ('NN', all_nn)]:
            df = all_chain_data[condition].get(model, pd.DataFrame())
            if 'chain_signature' in df.columns:
                sig_set.update(set(df['chain_signature']))
    
    common_all = all_dt & all_lr & all_nn
    union_all = all_dt | all_lr | all_nn
    agreement_rate = (len(common_all) / len(union_all) * 100) if union_all else 0
    
    # Display interpretation
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0a1628,#1a3a5c);color:white;
                    padding:1.5rem;border-radius:14px;border:1px solid rgba(66,133,244,0.2);">
            <h4 style="color:#5b9cf5;margin-bottom:1rem;">🔍 Chain Detection Capability</h4>
            <ul style="list-style-type:none;padding-left:0;">
                <li style="margin-bottom:0.8rem;">
                    <strong>🏆 Best Chain Detector:</strong> {best_chain_model} 
                    ({model_stats[best_chain_model]['total']} chains)
                </li>
                <li style="margin-bottom:0.8rem;">
                    <strong>🎯 Best Unique Finder:</strong> {best_unique_model}
                    ({model_stats[best_unique_model]['unique']} unique chains, 
                     {model_stats[best_unique_model]['uniqueness']} uniqueness)
                </li>
                <li style="margin-bottom:0.8rem;">
                    <strong>🤝 Model Agreement:</strong> {agreement_rate:.1f}% 
                    ({len(common_all)} chains detected by all three)
                </li>
                <li style="margin-bottom:0.8rem;">
                    <strong>📊 Total Unique Chains:</strong> {len(union_all)} 
                    across all models and conditions
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Summary table
        summary_table = pd.DataFrame({
            'Metric': ['Best Chain Detector', 'Unique Chain Leader', 'Model Agreement Rate', 'Total Chain Pool'],
            'Value': [
                f"{best_chain_model} ({model_stats[best_chain_model]['total']})",
                f"{best_unique_model} ({model_stats[best_unique_model]['unique']})",
                f"{agreement_rate:.1f}%",
                str(len(union_all))
            ]
        })
        st.dataframe(summary_table, use_container_width=True, hide_index=True)
    
    # ===== THESIS TEXT GENERATOR =====
    st.markdown("---")
    st.markdown("#### 📝 Auto-Generated Thesis Paragraph")
    
    # Generate comparison text
    model_ranking = sorted(model_stats.items(), key=lambda x: x[1]['total'], reverse=True)
    
    thesis_text = f"""
    **Chain Detection Analysis in RDP5 AI Models**
    
    Across all {len(all_conditions)} simulation conditions, the three AI models demonstrated varying 
    capabilities in detecting recombination chain events. {model_ranking[0][0]} emerged as the most 
    prolific chain detector, identifying {model_ranking[0][1]['total']} unique chain signatures, 
    followed by {model_ranking[1][0]} ({model_ranking[1][1]['total']} chains) and 
    {model_ranking[2][0]} ({model_ranking[2][1]['total']} chains).
    
    Model complementarity was evident, with only {agreement_rate:.1f}% of chains ({len(common_all)} 
    out of {len(union_all)}) being detected by all three models. This suggests that the models 
    capture different aspects of recombination patterns, with substantial unique contributions 
    from each. {best_unique_model} showed the highest uniqueness rate 
    ({model_stats[best_unique_model]['uniqueness']}), indicating superior sensitivity to rare or 
    complex chain events that other models miss.
    
    These findings support the use of ensemble approaches or model-specific applications depending 
    on whether the goal is maximum chain detection ({best_chain_model}) or identification of unique, 
    potentially novel recombination patterns ({best_unique_model}).
    """
    
    st.markdown(thesis_text)
    
    # ===== EXPORT FULL ANALYSIS =====
    st.markdown("---")
    st.markdown("#### 📥 Export Complete Chain Analysis")
    
    # Create comprehensive export
    export_data = []
    for condition in all_conditions:
        for model in models:
            chains = all_chain_data[condition].get(model, pd.DataFrame())
            if not chains.empty:
                export_cols = ['tag', 'chain_ids', 'chain_length', 'chain_signature']
                if 'source_tab' in chains.columns:
                    export_cols.append('source_tab')
                
                temp_df = chains[export_cols].copy()
                temp_df['condition'] = condition
                temp_df['model'] = model
                export_data.append(temp_df)
    
    if export_data:
        export_df = pd.concat(export_data, ignore_index=True)
        
        col1, col2 = st.columns(2)
        with col1:
            csv = export_df.to_csv(index=False)
            st.download_button(
                "📥 Download All Chain Data",
                csv,
                "chain_analysis_complete.csv",
                key="dl_complete_chain"
            )
        with col2:
            summary_csv = pd.DataFrame([
                {'Model': m, 'Total Chains': s['total'], 'Unique Chains': s['unique'], 
                 'Uniqueness': s['uniqueness']}
                for m, s in model_stats.items()
            ]).to_csv(index=False)
            st.download_button(
                "📥 Download Summary Statistics",
                summary_csv,
                "chain_analysis_summary.csv",
                key="dl_chain_summary"
            )