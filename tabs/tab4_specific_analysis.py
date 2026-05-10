"""Tab: Specific Analysis - Chains, Clusters, and Steps Distribution"""

import streamlit as st
import pandas as pd
import re
from config import CONDITIONS_MAP, MODELS
from collections import Counter


def render(all_data, summary, conditions_map=CONDITIONS_MAP, models=MODELS):
    """Render Specific Analysis tab"""
    
    st.markdown("## ◈ Specific Analysis: Chains, Clusters & Steps")
    st.caption("Detailed analysis of Source_Tab clusters, chain events, and step-by-step distribution")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Condition-level tabs
    condition_tabs = st.tabs(list(conditions_map.keys()))
    
    for i, (condition, params) in enumerate(conditions_map.items()):
        with condition_tabs[i]:
            # Condition header
            st.markdown(f"### Condition {condition}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("μ", f"{params['mutation_rate']:.2e}")
            with col2:
                st.metric("r", f"{params['recomb_rate']:.3f}")
            with col3:
                available = summary['by_condition'].get(condition, {}).get('csv', 0)
                st.metric("Models", f"{available}/3")
            
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            
            # Model tabs within each condition
            model_tabs = st.tabs([f"{model}" for model in models])
            
            for j, model in enumerate(models):
                with model_tabs[j]:
                    original_df, result_df = all_data[condition][model]
                    
                    if result_df is not None and not result_df.empty:
                        render_specific_analysis(result_df, condition, model)
                    else:
                        st.warning(f"No result data for {condition}_{model}.csv")


def render_specific_analysis(result_df: pd.DataFrame, condition: str, model: str):
    """Render the specific analysis for a given condition-model combination."""
    
    # ===== PREPROCESS DATA =====
    processed_df = preprocess_result_df(result_df)
    
    # ===== METRICS OVERVIEW =====
    st.markdown("#### Overview Metrics")
    
    total_events = len(processed_df)
    chain_events = processed_df[processed_df['is_chain'] == True]
    single_events = processed_df[processed_df['is_chain'] == False]
    num_chains = chain_events['chain_group'].nunique() if not chain_events.empty else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Events", total_events)
    with col2:
        st.metric("Chain Events", len(chain_events))
    with col3:
        st.metric("Single Events", len(single_events))
    with col4:
        st.metric("Chain Groups", num_chains)
    
    # ===== TAB 1: CLUSTERS & CHAINS (Source_Tab + Chain Events) =====
    st.markdown("---")
    st.markdown("### 🔗 Clusters & Chains Analysis")
    st.caption("Events with chain RDP_Event_ID grouped by Source_Tab clusters")
    
    render_clusters_and_chains(processed_df, condition, model)
    
    # ===== TAB 2: SINGLE EVENTS (Steps 3, 5, 6, 7 - Non-chain) =====
    st.markdown("---")
    st.markdown("### 📋 Single Events: Steps 3, 5, 6, 7")
    st.caption("Non-chain events with step distribution")
    
    render_single_events_steps(single_events, condition, model)
    
    # ===== TAB 3: STEP 4 SPECIAL CASES =====
    st.markdown("---")
    st.markdown("### ⚠️ Step 4 Special Cases")
    st.caption("Cross-over and special event analysis")
    
    render_step4_special_cases(processed_df, condition, model)


def preprocess_result_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess result dataframe to identify chains and expand events.
    """
    df_copy = df.copy()
    
    # --- Identify chain events (comma-separated RDP_Event_ID) ---
    rdp_id_col = None
    for col in ['RDP_Event_ID', 'rdp_event_id', 'Event_ID']:
        if col in df_copy.columns:
            rdp_id_col = col
            break
    
    if rdp_id_col is None:
        df_copy['is_chain'] = False
        df_copy['chain_ids'] = ''
        df_copy['chain_count'] = 1
        df_copy['chain_group'] = df_copy.index.astype(str)
        return df_copy
    
    def detect_chain(val):
        if pd.isna(val):
            return False, [], 1
        val_str = str(val).strip()
        ids = [x.strip() for x in val_str.split(',') if x.strip()]
        return len(ids) > 1, ids, len(ids)
    
    chain_info = df_copy[rdp_id_col].apply(detect_chain)
    df_copy['is_chain'] = chain_info.apply(lambda x: x[0])
    df_copy['chain_ids'] = chain_info.apply(lambda x: ','.join(x[1]))
    df_copy['chain_count'] = chain_info.apply(lambda x: x[2])
    
    # Create chain group identifier
    df_copy['chain_group'] = df_copy.apply(
        lambda row: f"chain_{row.name}" if row['is_chain'] else f"single_{row.name}", 
        axis=1
    )
    
    # --- Identify Source_Tab (step) ---
    source_col = None
    for col in ['Source_Tab', 'source_tab', 'Step', 'step']:
        if col in df_copy.columns:
            source_col = col
            break
    
    if source_col and df_copy[source_col].dtype == 'object':
        # Extract step number
        df_copy['step_num'] = df_copy[source_col].apply(extract_step_number)
    elif source_col:
        df_copy['step_num'] = df_copy[source_col]
    else:
        df_copy['step_num'] = 'Unknown'
    
    return df_copy


def extract_step_number(val):
    """Extract step number from Source_Tab string."""
    if pd.isna(val):
        return 'Unknown'
    val_str = str(val)
    match = re.search(r'Step\s*(\d+)', val_str, re.IGNORECASE)
    if match:
        return f"Step {match.group(1)}"
    # Try just numbers
    match = re.search(r'(\d+)', val_str)
    if match:
        return f"Step {match.group(1)}"
    return 'Unknown'


def render_clusters_and_chains(df: pd.DataFrame, condition: str, model: str):
    """
    Render cluster and chain analysis.
    """
    
    source_col = None
    for col in ['Source_Tab', 'source_tab', 'Step', 'step']:
        if col in df.columns:
            source_col = col
            break
    
    if not source_col:
        st.warning("No Source_Tab column found")
        return
    
    # ===== CLUSTER SUMMARY =====
    st.markdown("#### Source_Tab Cluster Distribution")
    
    cluster_groups = df.groupby(source_col)
    
    cluster_summary = []
    for cluster_name, cluster_df in cluster_groups:
        total = len(cluster_df)
        chains = cluster_df[cluster_df['is_chain'] == True]
        singles = cluster_df[cluster_df['is_chain'] == False]
        chain_groups = chains['chain_group'].nunique() if not chains.empty else 0
        
        cluster_summary.append({
            'Source_Tab': str(cluster_name),
            'Total Events': total,
            'Chain Events': len(chains),
            'Single Events': len(singles),
            'Chain Groups': chain_groups,
            'Chain %': f"{(len(chains)/total*100):.1f}%" if total > 0 else "0%"
        })
    
    cluster_df_summary = pd.DataFrame(cluster_summary).sort_values('Total Events', ascending=False)
    
    st.dataframe(cluster_df_summary, use_container_width=True, height=300)
    
    # ===== CLUSTER DISTRIBUTION CHART =====
    import plotly.express as px
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            cluster_df_summary,
            x='Source_Tab',
            y=['Single Events', 'Chain Events'],
            title='Event Distribution by Source_Tab Cluster',
            labels={'value': 'Count', 'variable': 'Event Type'},
            barmode='stack',
            color_discrete_map={'Single Events': '#4285f4', 'Chain Events': '#ea4335'}
        )
        fig.update_layout(template="plotly_white", height=350, title_x=0.5)
        # FIX: Add unique key
        st.plotly_chart(
            fig, 
            use_container_width=True, 
            key=f"cluster_dist_{condition}_{model}"
        )
    
    with col2:
        fig = px.pie(
            cluster_df_summary,
            values='Total Events',
            names='Source_Tab',
            title='Cluster Proportion',
            hole=0.4
        )
        fig.update_layout(template="plotly_white", height=350, title_x=0.5)
        # FIX: Add unique key
        st.plotly_chart(
            fig, 
            use_container_width=True, 
            key=f"cluster_pie_{condition}_{model}"
        )
    
    # ===== CROSSOVER EVENTS =====
    # (No plotly charts here, so no key needed)
    st.markdown("---")
    st.markdown("#### 🔄 Cross-Over Events Analysis")
    
    tag_col = None
    for col in ['Tag', 'tag', 'Santa_Tag']:
        if col in df.columns:
            tag_col = col
            break
    
    crossover_events = []
    if tag_col:
        for idx, row in df.iterrows():
            if row['is_chain'] and len(row['chain_ids'].split(',')) > 1:
                chain_ids = [x.strip() for x in row['chain_ids'].split(',')]
                if len(set(chain_ids)) > 1:
                    crossover_events.append({
                        'Index': idx,
                        'Tag': row[tag_col],
                        'RDP_Event_ID': row.get('RDP_Event_ID', row.get('rdp_event_id', '')),
                        'Chain_Count': row['chain_count'],
                        'Chain_IDs': row['chain_ids'],
                        'Source_Tab': row.get(source_col, ''),
                        'Chain_Group': row['chain_group']
                    })
    
    if crossover_events:
        crossover_df = pd.DataFrame(crossover_events)
        st.info(f"Found {len(crossover_df)} cross-over events")
        st.dataframe(crossover_df, use_container_width=True, height=300)
    else:
        st.success("No cross-over events detected")
    
    # ===== CHAIN DETAILS =====
    # (No plotly charts)
    st.markdown("---")
    st.markdown("#### Chain Events Detail")
    
    chain_df = df[df['is_chain'] == True]
    if not chain_df.empty:
        st.metric("Total Chain Events", len(chain_df))
        display_cols = [c for c in [tag_col, 'RDP_Event_ID', source_col, 'chain_count', 'chain_ids'] 
                       if c in chain_df.columns]
        if display_cols:
            st.dataframe(chain_df[display_cols], use_container_width=True, height=300)
    else:
        st.info("No chain events found")
    
    # ===== PER-CLUSTER CHAIN BREAKDOWN =====
    st.markdown("---")
    st.markdown("#### Per-Cluster Chain Breakdown")
    
    for idx, (cluster_name, cluster_df) in enumerate(cluster_groups):
        chains_in_cluster = cluster_df[cluster_df['is_chain'] == True]
        if not chains_in_cluster.empty:
            with st.expander(f"{cluster_name} - {len(chains_in_cluster)} chain events"):
                chain_lengths = chains_in_cluster['chain_count'].value_counts().sort_index()
                
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    st.write("**Chain Length Distribution**")
                    length_df = pd.DataFrame({
                        'Chain Length': chain_lengths.index,
                        'Count': chain_lengths.values
                    })
                    st.dataframe(length_df, use_container_width=True)
                
                with col_b:
                    fig = px.bar(
                        length_df, x='Chain Length', y='Count',
                        title=f'Chain Length Distribution - {cluster_name}',
                        color='Count', color_continuous_scale='Reds'
                    )
                    fig.update_layout(height=250, title_x=0.5)
                    # FIX: Add unique key with cluster name and index
                    st.plotly_chart(
                        fig, 
                        use_container_width=True,
                        key=f"chain_length_{condition}_{model}_{idx}_{cluster_name}"
                    )
def render_single_events_steps(single_df: pd.DataFrame, condition: str, model: str):
    """
    Render analysis for single events (non-chain) with Steps 3, 5, 6, 7.
    """
    
    if single_df.empty:
        st.info("No single events found")
        return
    
    # Filter for Steps 3, 5, 6, 7
    target_steps = ['Step 3', 'Step 5', 'Step 6', 'Step 7']
    step_filtered = single_df[single_df['step_num'].isin(target_steps)]
    
    if step_filtered.empty:
        st.info("No events for Steps 3, 5, 6, 7 in single events")
        return
    
    st.metric("Total Single Events (Steps 3,5,6,7)", len(step_filtered))
    
    # ===== STEP DISTRIBUTION =====
    st.markdown("#### Step Distribution (Single Events)")
    
    step_counts = step_filtered['step_num'].value_counts()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        step_summary = pd.DataFrame({
            'Step': step_counts.index,
            'Count': step_counts.values,
            'Percentage': [f"{(c/len(step_filtered)*100):.1f}%" for c in step_counts.values]
        })
        st.dataframe(step_summary, use_container_width=True)
    
    with col2:
        import plotly.express as px
        
        fig = px.bar(
            step_summary, x='Step', y='Count',
            title='Single Events by Step (3, 5, 6, 7)',
            color='Step',
            color_discrete_map={
                'Step 3': '#4285f4',
                'Step 5': '#34a853',
                'Step 6': '#fbbc04',
                'Step 7': '#ea4335'
            },
            text='Count'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(template="plotly_white", height=350, title_x=0.5)
        st.plotly_chart(fig, use_container_width=True, key=f"single_steps_{condition}_{model}")
    
    # ===== STEP DETAILS =====
    st.markdown("---")
    st.markdown("#### Step-by-Step Detail")
    
    step_tabs = st.tabs(target_steps)
    
    for step_idx, step_name in enumerate(target_steps):
        with step_tabs[step_idx]:
            step_data = step_filtered[step_filtered['step_num'] == step_name]
            
            if not step_data.empty:
                st.metric(f"{step_name} Events", len(step_data))
                
                # Get relevant columns
                tag_col = None
                for col in ['Tag', 'tag', 'Santa_Tag']:
                    if col in step_data.columns:
                        tag_col = col
                        break
                
                display_cols = [c for c in [tag_col, 'step_num', 'RDP_Event_ID', 'rdp_event_id'] 
                               if c in step_data.columns]
                
                if display_cols:
                    st.dataframe(step_data[display_cols], use_container_width=True, height=300)
                    
                    csv = step_data[display_cols].to_csv(index=False)
                    st.download_button(
                        f"Download {step_name} Events",
                        csv,
                        f"{step_name}_{condition}_{model}.csv",
                        key=f"dl_{step_name}_{condition}_{model}"
                    )
            else:
                st.info(f"No events for {step_name}")
    
    # ===== STEP COMPARISON =====
    st.markdown("---")
    st.markdown("#### Step Comparison Metrics")
    
    # Compare step frequencies
    if not step_filtered.empty:
        tag_col = None
        for col in ['Tag', 'tag', 'Santa_Tag']:
            if col in step_filtered.columns:
                tag_col = col
                break
        
        if tag_col:
            # Tags per step
            step_tag_counts = step_filtered.groupby('step_num')[tag_col].nunique()
            step_tag_df = pd.DataFrame({
                'Step': step_tag_counts.index,
                'Unique Tags': step_tag_counts.values
            })
            
            fig = px.bar(
                step_tag_df, x='Step', y='Unique Tags',
                title='Unique Tags per Step',
                color='Step',
                text='Unique Tags'
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(template="plotly_white", height=300, title_x=0.5)
            st.plotly_chart(fig, use_container_width=True, key=f"step_tags_{condition}_{model}")


def render_step4_special_cases(df: pd.DataFrame, condition: str, model: str):
    """
    Render analysis for Step 4 special cases (cross-over events).
    Step 4 typically involves cross-over or recombinant validation.
    """
    
    # Filter for Step 4
    step4_events = df[df['step_num'] == 'Step 4']
    
    if step4_events.empty:
        st.info("No Step 4 events found")
        return
    
    st.metric("Total Step 4 Events", len(step4_events))
    
    # ===== STEP 4: CHAIN VS SINGLE =====
    col1, col2, col3 = st.columns(3)
    
    step4_chains = step4_events[step4_events['is_chain'] == True]
    step4_singles = step4_events[step4_events['is_chain'] == False]
    
    with col1:
        st.metric("Chain Events", len(step4_chains))
    with col2:
        st.metric("Single Events", len(step4_singles))
    with col3:
        st.metric("Chain Groups", step4_chains['chain_group'].nunique() if not step4_chains.empty else 0)
    
    # ===== STEP 4 SPECIAL CHARACTERISTICS =====
    st.markdown("---")
    st.markdown("#### Step 4: Event Characteristics")
    
    # Tag distribution
    tag_col = None
    for col in ['Tag', 'tag', 'Santa_Tag']:
        if col in step4_events.columns:
            tag_col = col
            break
    
    if tag_col:
        tag_dist = step4_events[tag_col].value_counts().head(20)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("**Top Tags in Step 4**")
            tag_summary = pd.DataFrame({
                'Tag': tag_dist.index,
                'Count': tag_dist.values
            })
            st.dataframe(tag_summary, use_container_width=True)
        
        with col2:
            import plotly.express as px
            fig = px.bar(
                tag_summary.head(15), x='Tag', y='Count',
                title=f'Top Tags in Step 4 - {condition} {model}',
                color='Count', color_continuous_scale='Viridis'
            )
            fig.update_layout(height=350, title_x=0.5)
            st.plotly_chart(fig, use_container_width=True, key=f"step4_tags_{condition}_{model}")
    
    # ===== STEP 4 CHAIN ANALYSIS =====
    if not step4_chains.empty:
        st.markdown("---")
        st.markdown("#### Step 4: Chain Event Analysis")
        
        chain_lengths = step4_chains['chain_count'].value_counts().sort_index()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Chain Length Distribution**")
            chain_len_df = pd.DataFrame({
                'Chain Length': chain_lengths.index,
                'Count': chain_lengths.values
            })
            st.dataframe(chain_len_df, use_container_width=True)
        
        with col2:
            fig = px.pie(
                chain_len_df, values='Count', names='Chain Length',
                title='Chain Length Distribution (Step 4)',
                hole=0.3
            )
            fig.update_layout(height=300, title_x=0.5)
            st.plotly_chart(fig, use_container_width=True, key=f"step4_chain_dist_{condition}_{model}")
        
        # Display chain events table
        st.markdown("**Step 4 Chain Events Detail**")
        display_cols = [c for c in [tag_col, 'RDP_Event_ID', 'chain_count', 'chain_ids', 'step_num'] 
                       if c in step4_chains.columns]
        if display_cols:
            st.dataframe(step4_chains[display_cols], use_container_width=True, height=300)
    
    # ===== STEP 4 CROSSOVER DETECTION =====
    st.markdown("---")
    st.markdown("#### Step 4: Cross-Over Detection")
    
    # Detect cross-references in Step 4
    crossover_indicators = []
    for idx, row in step4_events.iterrows():
        rdp_id = str(row.get('RDP_Event_ID', row.get('rdp_event_id', '')))
        if ',' in rdp_id and len(rdp_id.split(',')) > 1:
            ids = [x.strip() for x in rdp_id.split(',')]
            crossover_indicators.append({
                'Index': idx,
                'Tag': row.get(tag_col, '') if tag_col else '',
                'RDP_Event_ID': rdp_id,
                'ID_Count': len(ids),
                'IDs': rdp_id
            })
    
    if crossover_indicators:
        crossover_df = pd.DataFrame(crossover_indicators)
        st.warning(f"⚠️ {len(crossover_df)} potential cross-over events detected in Step 4")
        st.dataframe(crossover_df, use_container_width=True, height=300)
    
    # ===== STEP 4 COMPARISON WITH OTHER STEPS =====
    st.markdown("---")
    st.markdown("#### Step 4 vs Other Steps")
    
    other_steps = df[df['step_num'] != 'Step 4']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Step 4 Total", len(step4_events))
    with col2:
        st.metric("Other Steps Total", len(other_steps))
    with col3:
        step4_chain_pct = (len(step4_chains) / len(step4_events) * 100) if len(step4_events) > 0 else 0
        st.metric("Step 4 Chain %", f"{step4_chain_pct:.1f}%")
    with col4:
        other_chains = other_steps[other_steps['is_chain'] == True]
        other_chain_pct = (len(other_chains) / len(other_steps) * 100) if len(other_steps) > 0 else 0
        st.metric("Other Steps Chain %", f"{other_chain_pct:.1f}%")
    
    # Comparison chart
    comparison_data = pd.DataFrame({
        'Category': ['Step 4', 'Other Steps'],
        'Chain Events': [len(step4_chains), len(other_chains)],
        'Single Events': [len(step4_singles), len(other_steps[other_steps['is_chain'] == False])]
    })
    
    import plotly.express as px
    fig = px.bar(
        comparison_data, x='Category', y=['Chain Events', 'Single Events'],
        title='Step 4 vs Other Steps: Chain vs Single Events',
        barmode='group',
        color_discrete_map={'Chain Events': '#ea4335', 'Single Events': '#4285f4'}
    )
    fig.update_layout(template="plotly_white", height=350, title_x=0.5)
    st.plotly_chart(fig, use_container_width=True, key=f"step4_comparison_{condition}_{model}")
    
    # ===== EXPORT =====
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        csv_all = step4_events.to_csv(index=False)
        st.download_button(
            "Download All Step 4 Events",
            csv_all,
            f"step4_all_{condition}_{model}.csv",
            key=f"dl_step4_all_{condition}_{model}"
        )
    with col2:
        if not step4_chains.empty:
            csv_chains = step4_chains.to_csv(index=False)
            st.download_button(
                "Download Step 4 Chains",
                csv_chains,
                f"step4_chains_{condition}_{model}.csv",
                key=f"dl_step4_chains_{condition}_{model}"
            )
    with col3:
        csv_singles = step4_singles.to_csv(index=False)
        st.download_button(
            "Download Step 4 Singles",
            csv_singles,
            f"step4_singles_{condition}_{model}.csv",
            key=f"dl_step4_singles_{condition}_{model}"
        )