"""UI display components for Streamlit."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re
from typing import Dict, List, Optional, Tuple
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


import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from matplotlib_venn import venn3
import matplotlib.pyplot as plt

def display_model_agreement_analysis(all_data, condition, MODELS):
    """
    For a given condition, compare matched events across DT, LR, NN.
    Computes common, pairwise, and unique detections.
    Handles chain cases where RDP_Event_ID contains multiple IDs.
    Displays detailed tables of common and unique events with all columns.
    """
    import re
    
    # --- Step 1: Extract matched event sets and full dataframes for each model ---
    model_sets = {}
    model_correct_sets = {}
    model_dfs = {}  # Store processed dataframes with event_id column
    model_full_dfs = {}  # Store original result dataframes
    
    for model in MODELS:
        original_df, result_df = all_data[condition][model]
        
        if result_df is None or result_df.empty:
            st.warning(f"No results for {model} in {condition}")
            return
        
        result_copy = result_df.copy()
        model_full_dfs[model] = result_df.copy()  # Keep original
        
        # Handle different possible column names for Tag
        tag_col = None
        for col in ['Tag', 'tag', 'Santa_Tag']:
            if col in result_copy.columns:
                tag_col = col
                break
        
        if tag_col is None:
            st.warning(f"Cannot find Tag column for {model} in {condition}")
            return
        
        # Extract tag number
        def extract_tag_number(val):
            if pd.isna(val):
                return None
            numbers = re.findall(r'\d+', str(val))
            return int(numbers[0]) if numbers else None
        
        result_copy['tag_num'] = result_copy[tag_col].apply(extract_tag_number)
        
        # Get RDP Event ID
        rdp_id_col = None
        for col in ['RDP_Event_ID', 'rdp_event_id', 'Event_ID']:
            if col in result_copy.columns:
                rdp_id_col = col
                break
        
        if rdp_id_col is None:
            st.warning(f"Cannot find RDP Event ID column for {model} in {condition}")
            return
        
        # Handle CHAIN CASES: split comma-separated RDP Event IDs
        # Each ID in a chain is treated as a separate event
        all_events = []
        for idx, row in result_copy.iterrows():
            rdp_val = str(row[rdp_id_col])
            # Split by comma for chain cases (e.g., "6, 8")
            event_ids = [eid.strip() for eid in rdp_val.split(',') if eid.strip()]
            
            for eid in event_ids:
                event_row = row.copy()
                event_row['event_id'] = f"{row['tag_num']}_{eid}"
                event_row['single_rdp_id'] = eid  # Store individual ID
                all_events.append(event_row)
        
        expanded_df = pd.DataFrame(all_events)
        result_copy = expanded_df
        
        # All events are considered matched
        model_sets[model] = set(result_copy['event_id'].dropna())
        
        # Store the expanded dataframe
        model_dfs[model] = result_copy.set_index('event_id')
        
        # For "correct" detections
        if 'Step7' in result_copy.columns:
            correct_mask = result_copy['Step7'] == False
        elif 'step7' in result_copy.columns:
            correct_mask = result_copy['step7'] == False
        elif 'Incorrect_Parental' in result_copy.columns:
            correct_mask = result_copy['Incorrect_Parental'] == False
        else:
            correct_mask = pd.Series(True, index=result_copy.index)
        
        model_correct_sets[model] = set(result_copy.loc[correct_mask, 'event_id'].dropna())
    
    # Get sets for each model
    dt_set = model_sets.get('DT', set())
    lr_set = model_sets.get('LR', set())
    nn_set = model_sets.get('NN', set())
    
    dt_correct = model_correct_sets.get('DT', set())
    lr_correct = model_correct_sets.get('LR', set())
    nn_correct = model_correct_sets.get('NN', set())
    
    if not dt_set or not lr_set or not nn_set:
        st.warning("One or more models have no matched events for this condition.")
    
    # --- Step 2: Compute overlaps ---
    common_all = dt_set & lr_set & nn_set
    dt_lr_overlap = (dt_set & lr_set) - nn_set
    dt_nn_overlap = (dt_set & nn_set) - lr_set
    lr_nn_overlap = (lr_set & nn_set) - dt_set
    dt_only = dt_set - lr_set - nn_set
    lr_only = lr_set - dt_set - nn_set
    nn_only = nn_set - dt_set - lr_set
    
    common_correct = dt_correct & lr_correct & nn_correct
    dt_unique_correct = dt_correct - lr_correct - nn_correct
    lr_unique_correct = lr_correct - dt_correct - nn_correct
    nn_unique_correct = nn_correct - dt_correct - lr_correct
    
    dt_unique_incorrect = (dt_set - lr_set - nn_set) - dt_correct
    lr_unique_incorrect = (lr_set - dt_set - nn_set) - lr_correct
    nn_unique_incorrect = (nn_set - dt_set - lr_set) - nn_correct
    
    # --- Step 3: Display Summary Results ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### A. Overlap Matrix (All Matched Events)")
        overlap_data = {
            'Model': ['DT', 'LR', 'NN'],
            'DT': [len(dt_set), len(dt_set & lr_set), len(dt_set & nn_set)],
            'LR': [len(lr_set & dt_set), len(lr_set), len(lr_set & nn_set)],
            'NN': [len(nn_set & dt_set), len(nn_set & lr_set), len(nn_set)]
        }
        overlap_df = pd.DataFrame(overlap_data).set_index('Model')
        st.dataframe(overlap_df, use_container_width=True)
        
        st.markdown("#### Venn Diagram")
        try:
            fig, ax = plt.subplots(figsize=(6, 6))
            venn3([dt_set, lr_set, nn_set], 
                  set_labels=('DT', 'LR', 'NN'))
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.info(f"Venn diagram not available: {e}")
    
    with col2:
        st.markdown("#### B. Unique Detection Analysis")
        unique_data = {
            'Category': ['Common (All 3)', 'DT Only', 'LR Only', 'NN Only',
                        'DT-LR (no NN)', 'DT-NN (no LR)', 'LR-NN (no DT)'],
            'Total': [len(common_all), len(dt_only), len(lr_only), len(nn_only),
                      len(dt_lr_overlap), len(dt_nn_overlap), len(lr_nn_overlap)]
        }
        unique_df = pd.DataFrame(unique_data)
        st.dataframe(unique_df, use_container_width=True)
        
        st.markdown("#### C. Unique Detections Quality")
        quality_data = {
            'Model': ['DT', 'LR', 'NN'],
            'Unique Correct': [len(dt_unique_correct), len(lr_unique_correct), len(nn_unique_correct)],
            'Unique Incorrect': [len(dt_unique_incorrect), len(lr_unique_incorrect), len(nn_unique_incorrect)],
            'Total Unique': [len(dt_only), len(lr_only), len(nn_only)]
        }
        quality_df = pd.DataFrame(quality_data)
        quality_df['Sensitivity'] = (quality_df['Unique Correct'] / quality_df['Total Unique'].replace(0, 1) * 100).round(1)
        st.dataframe(quality_df, use_container_width=True)
    
    # --- Step 4: DETAILED EVENT TABLES (Common & Missing) ---
    st.markdown("---")
    st.markdown("## 📋 Detailed Event Analysis")
    
    # Helper function to get rows by event IDs from a model's dataframe
    def get_event_rows(event_ids, model_df):
        """Get rows for given event_ids from model dataframe"""
        available_ids = [eid for eid in event_ids if eid in model_df.index]
        if available_ids:
            return model_df.loc[available_ids].reset_index(drop=True)
        return pd.DataFrame()
    
    # --- 4A: Events Common to All 3 Models ---
    st.markdown("### 🔵 Events Detected by ALL Models (Common)")
    st.caption(f"Total: {len(common_all)} events detected by DT, LR, and NN")
    
    if common_all:
        # Get common events from DT's dataframe (they exist in all three)
        common_dt = get_event_rows(common_all, model_dfs['DT'])
        common_lr = get_event_rows(common_all, model_dfs['LR'])
        common_nn = get_event_rows(common_all, model_dfs['NN'])
        
        if not common_dt.empty:
            # Add model-specific columns to distinguish
            common_dt['Detected_By'] = 'DT+LR+NN'
            
            # Display with tabs for each model's view
            common_tabs = st.tabs(["DT View", "LR View", "NN View"])
            with common_tabs[0]:
                st.dataframe(common_dt, use_container_width=True, height=300)
                st.download_button(
                    f"Download Common Events (DT View)", 
                    common_dt.to_csv(index=False),
                    f"common_events_DT_{condition}.csv",
                    key=f"dl_common_dt_{condition}"
                )
            with common_tabs[1]:
                st.dataframe(common_lr, use_container_width=True, height=300)
                st.download_button(
                    f"Download Common Events (LR View)", 
                    common_lr.to_csv(index=False),
                    f"common_events_LR_{condition}.csv",
                    key=f"dl_common_lr_{condition}"
                )
            with common_tabs[2]:
                st.dataframe(common_nn, use_container_width=True, height=300)
                st.download_button(
                    f"Download Common Events (NN View)", 
                    common_nn.to_csv(index=False),
                    f"common_events_NN_{condition}.csv",
                    key=f"dl_common_nn_{condition}"
                )
    else:
        st.info("No events detected by all three models.")
    
    # --- 4B: Events Unique to Each Model ---
    st.markdown("---")
    st.markdown("### 🟡 Events Unique to Each Model")
    
    unique_tabs = st.tabs(["DT Only", "LR Only", "NN Only"])
    
    for tab_idx, (model_name, unique_set) in enumerate([
        ('DT', dt_only), ('LR', lr_only), ('NN', nn_only)
    ]):
        with unique_tabs[tab_idx]:
            st.markdown(f"**{model_name} Unique Detections:** {len(unique_set)} events")
            
            if unique_set:
                unique_rows = get_event_rows(unique_set, model_dfs[model_name])
                if not unique_rows.empty:
                    unique_rows['Status'] = 'Unique to ' + model_name
                    
                    # Highlight incorrect detections
                    incorrect_set = model_name == 'DT' and dt_unique_incorrect or \
                                   model_name == 'LR' and lr_unique_incorrect or \
                                   nn_unique_incorrect
                    
                    # Show correct vs incorrect breakdown
                    correct_in_set = model_name == 'DT' and dt_unique_correct or \
                                    model_name == 'LR' and lr_unique_correct or \
                                    nn_unique_correct
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Correct", len(correct_in_set))
                    with col_b:
                        st.metric("Incorrect", len(incorrect_set))
                    
                    st.dataframe(unique_rows, use_container_width=True, height=300)
                    st.download_button(
                        f"Download {model_name} Unique Events",
                        unique_rows.to_csv(index=False),
                        f"{model_name}_unique_{condition}.csv",
                        key=f"dl_unique_{model_name}_{condition}"
                    )
            else:
                st.info(f"No events unique to {model_name}.")
    
    # --- 4C: Pairwise Overlaps (Missing from one model) ---
    st.markdown("---")
    st.markdown("### 🟠 Pairwise Overlaps (Detected by 2, Missing from 1)")
    
    pairwise_tabs = st.tabs(["DT+LR (Missing NN)", "DT+NN (Missing LR)", "LR+NN (Missing DT)"])
    
    for tab_idx, (label, overlap_set, missing_model) in enumerate([
        ('DT & LR (NN missing)', dt_lr_overlap, 'NN'),
        ('DT & NN (LR missing)', dt_nn_overlap, 'LR'),
        ('LR & NN (DT missing)', lr_nn_overlap, 'DT')
    ]):
        with pairwise_tabs[tab_idx]:
            st.markdown(f"**{label}:** {len(overlap_set)} events")
            
            if overlap_set:
                # Get from one of the available models
                available_models = [m for m in MODELS if m != missing_model]
                display_model = available_models[0]
                overlap_rows = get_event_rows(overlap_set, model_dfs[display_model])
                
                if not overlap_rows.empty:
                    overlap_rows['Missing_From'] = missing_model
                    overlap_rows['Detected_By'] = ' + '.join(available_models)
                    st.dataframe(overlap_rows, use_container_width=True, height=300)
                    st.download_button(
                        f"Download {label} Events",
                        overlap_rows.to_csv(index=False),
                        f"overlap_{missing_model}_missing_{condition}.csv",
                        key=f"dl_overlap_{missing_model}_{condition}"
                    )
            else:
                st.info(f"No events in this category.")
    
    # --- Step 5: Model Sensitivity ---
    st.markdown("---")
    st.markdown("#### D. Model Sensitivity to Unique Events")
    st.caption("Higher = better at finding patterns others miss")
    
    power_data = {
        'Model': ['DT', 'LR', 'NN'],
        'Sensitivity (%)': [
            round(len(dt_unique_correct) / max(len(dt_correct), 1) * 100, 1),
            round(len(lr_unique_correct) / max(len(lr_correct), 1) * 100, 1),
            round(len(nn_unique_correct) / max(len(nn_correct), 1) * 100, 1)
        ]
    }
    power_df = pd.DataFrame(power_data)
    
    fig = px.bar(power_df, x='Model', y='Sensitivity (%)', color='Model',
                 color_discrete_map={'DT': '#1f77b4', 'LR': '#2ca02c', 'NN': '#ff7f0e'},
                 text='Sensitivity (%)')
    fig.update_traces(texttemplate='%{text}%', textposition='outside')
    st.plotly_chart(fig, use_container_width=True, key=f"model_sensitivity_{condition}")
    
    # --- Step 6: Intersection Quality ---
    st.markdown("#### E. Intersection Quality")
    st.caption("Accuracy of shared vs. unique detections")
    
    common_accuracy = (len(common_correct) / len(common_all) * 100) if len(common_all) > 0 else 0
    
    metrics = []
    for model, unique_total, unique_correct in [
        ('DT', len(dt_only), len(dt_unique_correct)),
        ('LR', len(lr_only), len(lr_unique_correct)),
        ('NN', len(nn_only), len(nn_unique_correct))
    ]:
        acc = (unique_correct / unique_total * 100) if unique_total > 0 else 0
        metrics.append({'Category': f'{model} Unique', 'Accuracy (%)': round(acc, 1)})
    
    metrics.append({'Category': 'Common (All 3)', 'Accuracy (%)': round(common_accuracy, 1)})
    
    quality_intersection_df = pd.DataFrame(metrics)
    
    fig2 = px.bar(quality_intersection_df, x='Category', y='Accuracy (%)', color='Category',
                  text='Accuracy (%)')
    fig2.update_traces(texttemplate='%{text}%', textposition='outside')
    st.plotly_chart(fig2, use_container_width=True, key=f"intersection_quality_{condition}")
    
    # --- Step 7: Thesis-Ready Interpretation ---
    st.markdown("---")
    st.markdown("### 📊 Interpretation")
    
    if not power_df.empty and len(power_df) > 0:
        best_sensitivity_model = power_df.loc[power_df['Sensitivity (%)'].idxmax(), 'Model']
        best_sensitivity_value = power_df.loc[power_df['Model'] == best_sensitivity_model, 'Sensitivity (%)'].values[0]
    else:
        best_sensitivity_model = "N/A"
        best_sensitivity_value = 0
    
    # Calculate average accuracy of unique detections
    unique_accuracies = [m['Accuracy (%)'] for m in metrics if 'Unique' in m['Category']]
    avg_unique_acc = sum(unique_accuracies) / len(unique_accuracies) if unique_accuracies else 0
    
    st.markdown(f"""
    **Key Findings for Condition {condition}:**
    
    1. **Complementarity**: The models show {'high' if len(common_all) < max(len(dt_only), len(lr_only), len(nn_only)) else 'low'} complementarity, 
       with **{len(common_all)}** events detected by all three models.
    
    2. **Unique Strength**: **{best_sensitivity_model}** demonstrates the highest sensitivity to unique events 
       (**{best_sensitivity_value}%**), 
       indicating it captures patterns that other models miss.
    
    3. **Detection Quality**: Common detections achieve **{common_accuracy:.1f}%** accuracy, 
       {'higher' if common_accuracy > avg_unique_acc else 'lower'} 
       than unique detections on average (**{avg_unique_acc:.1f}%**).
    
    4. **Redundancy Check**: {
       'DT and LR show high redundancy (many shared detections)' if len(dt_set & lr_set) > len(dt_only) + len(lr_only) else 
       'Models provide complementary information with distinct detection patterns'
    }.
    
    5. **Chain Events**: {
       'Chain events (comma-separated RDP IDs) were expanded for analysis' 
       if any(',' in str(row.get('RDP_Event_ID', '')) for model in MODELS for _, row in model_full_dfs[model].iterrows()) 
       else 'No chain events detected'
    }.
    """)


def display_study_metrics_section(result_df: pd.DataFrame, original_df: pd.DataFrame,
                                   condition: str, model: str):
    """
    Display the special study section with three key metrics.
    """
    from analysis import compute_study_metrics, analyze_original_by_tool, analyze_breakpoint_distances
    
    st.markdown("---")
    st.markdown("## ◈ Performance Metrics")
    st.caption(f"Condition {condition} · Model {model}")
    
    # Compute metrics
    orig_stats = analyze_original_by_tool(original_df) if original_df is not None else {}
    metrics = compute_study_metrics(result_df, orig_stats)
    
    # ===== (1) FALSE POSITIVE RATE =====
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        fpr_val = metrics['false_positive_rate']
        fpr_color = "#34a853" if fpr_val < 0.12 else "#fbbc04" if fpr_val < 0.18 else "#ea4335"
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{fpr_color},rgba(0,0,0,0.3));color:white;
                    padding:1.5rem;border-radius:14px;text-align:center;height:100%;">
            <div style="font-size:0.7rem;opacity:0.9;text-transform:uppercase;letter-spacing:2px;margin-bottom:0.5rem;">
                False Positive Rate</div>
            <div style="font-size:3rem;font-weight:800;">{fpr_val:.1%}</div>
            <div style="font-size:0.8rem;opacity:0.8;margin-top:0.5rem;">
                {metrics['unmatched_rdp']:,} / {metrics['total_original_rdp']:,}</div>
            <div style="font-size:0.7rem;opacity:0.6;">Unmatched / Original RDP</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### (1) False Positive Rate Analysis")
        st.caption("Proportion of RDP-inferred events unmatched to SANTA")
        st.latex(r"FPR = \frac{\text{Unmatched RDP}}{\text{Total Original RDP}}")
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Original RDP", f"{metrics['total_original_rdp']:,}")
        with m2:
            st.metric("Matched RDP", f"{metrics['matched_rdp']:,}")
        with m3:
            st.metric("Unmatched RDP", f"{metrics['unmatched_rdp']:,}")
        
        # Progress bar showing matched vs unmatched
        if metrics['total_original_rdp'] > 0:
            matched_pct = metrics['matched_rdp'] / metrics['total_original_rdp']
            st.progress(matched_pct, text=f"RDP Precision: {matched_pct:.1%} matched")
    
    # ===== (2) RECOMBINANT ACCURACY =====
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        acc_val = metrics['recombinant_accuracy']
        acc_color = "#34a853" if acc_val > 0.95 else "#fbbc04" if acc_val > 0.90 else "#ea4335"
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{acc_color},rgba(0,0,0,0.3));color:white;
                    padding:1.5rem;border-radius:14px;text-align:center;height:100%;">
            <div style="font-size:0.7rem;opacity:0.9;text-transform:uppercase;letter-spacing:2px;margin-bottom:0.5rem;">
                Recombinant Accuracy</div>
            <div style="font-size:3rem;font-weight:800;">{acc_val:.1%}</div>
            <div style="font-size:0.8rem;opacity:0.8;margin-top:0.5rem;">
                {metrics['correct_recombinant']:,} / {metrics['total_matched_rdp']:,}</div>
            <div style="font-size:0.7rem;opacity:0.6;">Correct / Total Matched RDP</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### (2) Recombinant Identification Accuracy")
        st.caption("Step 7 = parental misidentification · All others = correct")
        st.latex(r"\text{Accuracy} = 1 - \frac{\text{Step 7}}{\text{Total Matched RDP}}")
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Matched RDP", f"{metrics['total_matched_rdp']:,}")
        with m2:
            st.metric("Correct", f"{metrics['correct_recombinant']:,}")
        with m3:
            st.metric("Step 7", f"{metrics['step7_count']:,}", delta=f"-{metrics['step7_count']}" if metrics['step7_count'] > 0 else "0")
        
        if metrics['total_matched_rdp'] > 0:
            correct_pct = metrics['correct_recombinant'] / metrics['total_matched_rdp']
            st.progress(correct_pct, text=f"Correct: {correct_pct:.1%}")
    
    # Pie chart
    if metrics['total_matched_rdp'] > 0:
        fig = go.Figure(data=[go.Pie(
            labels=['Correct Recombinant', 'Step 7 (Incorrect)'],
            values=[metrics['correct_recombinant'], metrics['step7_count']],
            marker_colors=['#34a853', '#ea4335'],
            hole=0.5,
            textinfo='label+percent',
            textfont=dict(size=13)
        )])
        fig.update_layout(
            title=f"Recombinant Identification · {metrics['total_matched_rdp']:,} matched RDP events",
            height=320, margin=dict(t=40, b=10), showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True, key=f"recombinant_donut_{condition}_{model}")
    
    # ===== (3) BREAKPOINT DISTANCE =====
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        bp_val = metrics['mean_breakpoint_distance']
        bp_color = "#34a853" if bp_val < 100 else "#fbbc04" if bp_val < 200 else "#ea4335"
        bp_display = f"{bp_val:.1f} bp" if bp_val > 0 else "N/A"
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{bp_color},rgba(0,0,0,0.3));color:white;
                    padding:1.5rem;border-radius:14px;text-align:center;height:100%;">
            <div style="font-size:0.7rem;opacity:0.9;text-transform:uppercase;letter-spacing:2px;margin-bottom:0.5rem;">
                Mean BP Distance</div>
            <div style="font-size:2.5rem;font-weight:800;">{bp_display}</div>
            <div style="font-size:0.8rem;opacity:0.8;margin-top:0.5rem;">
                {metrics['total_breakpoints_analyzed']:,} analyzed</div>
            <div style="font-size:0.7rem;opacity:0.6;">Circular distance (10,000 bp genome)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### (3) Breakpoint Distance Analysis")
        st.caption("Circular distance between inferred and simulated breakpoints")
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Mean Start", f"{metrics['mean_start_distance']:.1f} bp" if metrics['mean_start_distance'] > 0 else 'N/A')
        with m2:
            st.metric("Mean End", f"{metrics['mean_end_distance']:.1f} bp" if metrics['mean_end_distance'] > 0 else 'N/A')
        with m3:
            st.metric("Analyzed", f"{metrics['total_breakpoints_analyzed']:,}")
    
    bp_distances = analyze_breakpoint_distances(result_df)
    
    if bp_distances['total_distances']:
        fig = px.histogram(
            x=bp_distances['total_distances'],
            nbins=30,
            title="Breakpoint Distance Distribution",
            labels={'x': 'Circular Distance (bp)', 'y': 'Frequency'},
            color_discrete_sequence=['#4285f4']
        )
        fig.update_layout(template="plotly_white", title_x=0.5, height=350,
                         bargap=0.05, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True, key=f"bp_hist_{condition}_{model}")
    
    # ===== SUMMARY FOOTER =====
    st.markdown("---")
    
    fpr_color = "#34a853" if metrics['false_positive_rate'] < 0.12 else "#fbbc04" if metrics['false_positive_rate'] < 0.18 else "#ea4335"
    acc_color = "#34a853" if metrics['recombinant_accuracy'] > 0.95 else "#fbbc04" if metrics['recombinant_accuracy'] > 0.90 else "#ea4335"
    
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0a1628,#1a3a5c);color:white;
                padding:1.5rem;border-radius:14px;border:1px solid rgba(66,133,244,0.2);">
        <div style="display:flex;gap:2rem;flex-wrap:wrap;">
            <div style="flex:1;min-width:200px;">
                <div style="font-size:0.7rem;opacity:0.7;text-transform:uppercase;letter-spacing:1px;">Condition · Model</div>
                <div style="font-size:1.1rem;font-weight:600;">{condition} · {model}</div>
            </div>
            <div style="flex:1;min-width:150px;">
                <div style="font-size:0.7rem;opacity:0.7;text-transform:uppercase;letter-spacing:1px;">FPR</div>
                <div style="font-size:1.3rem;font-weight:700;color:{fpr_color};">{metrics['false_positive_rate']:.3%}</div>
            </div>
            <div style="flex:1;min-width:150px;">
                <div style="font-size:0.7rem;opacity:0.7;text-transform:uppercase;letter-spacing:1px;">Accuracy</div>
                <div style="font-size:1.3rem;font-weight:700;color:{acc_color};">{metrics['recombinant_accuracy']:.3%}</div>
            </div>
            <div style="flex:1;min-width:150px;">
                <div style="font-size:0.7rem;opacity:0.7;text-transform:uppercase;letter-spacing:1px;">BP Distance</div>
                <div style="font-size:1.3rem;font-weight:700;">{metrics['mean_breakpoint_distance']:.1f} bp</div>
            </div>
        </div>
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

def display_research_comparison():
    """Research-level cross-condition comparison with model rankings and visualizations."""
    from analysis import compute_study_summary_all_conditions
    
    summary_df = compute_study_summary_all_conditions()
    if not summary_df.empty:
        # Streamlit automatically adds download icon in the dataframe toolbar
        # Just display the dataframe - hover to see the download button
        st.dataframe(summary_df, use_container_width=True, height=500)

    if summary_df.empty:
        st.warning("No data available for comparison.")
        return
    
    # ===== PREPARE DATA =====
    plot_data = []
    for _, row in summary_df.iterrows():
        try:
            fpr = float(str(row['FPR']).strip('%')) / 100 if 'FPR' in row else 0
            accuracy = float(str(row['Accuracy']).strip('%')) / 100 if 'Accuracy' in row else 0
            mean_bp = float(str(row['Mean_BP_Dist'])) if row.get('Mean_BP_Dist') and row['Mean_BP_Dist'] != 'N/A' else 0
        except:
            fpr = 0
            accuracy = 0
            mean_bp = 0
        
        plot_data.append({
            'Condition': row['Condition'],
            'Model': row['Model'],
            'FPR': fpr,
            'Accuracy': accuracy,
            'Mean_BP_Dist': mean_bp,
            'Orig_RDP': row.get('Orig_RDP', 0),
            'Matched_RDP': row.get('Matched_RDP', 0),
            'Label': f"{row['Condition']}-{row['Model']}"
        })
    
    plot_df = pd.DataFrame(plot_data)
    
    # ===== ROW 1: KEY FINDINGS CARDS =====
    st.markdown("##Cross-Condition Analysis")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown("### Key Findings Summary")
    
    # Best and worst models
    best_fpr = plot_df.loc[plot_df['FPR'].idxmin()]
    worst_fpr = plot_df.loc[plot_df['FPR'].idxmax()]
    best_acc = plot_df.loc[plot_df['Accuracy'].idxmax()]
    worst_acc = plot_df.loc[plot_df['Accuracy'].idxmin()]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#34a853,#1b8a3d);color:white;padding:1.2rem;border-radius:12px;text-align:center;">
            <div style="font-size:0.75rem;opacity:0.9;text-transform:uppercase;letter-spacing:1px;">Best FPR</div>
            <div style="font-size:2rem;font-weight:700;">{best_fpr['FPR']:.1%}</div>
            <div style="font-size:0.85rem;">{best_fpr['Label']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#ea4335,#c5221f);color:white;padding:1.2rem;border-radius:12px;text-align:center;">
            <div style="font-size:0.75rem;opacity:0.9;text-transform:uppercase;letter-spacing:1px;">Worst FPR</div>
            <div style="font-size:2rem;font-weight:700;">{worst_fpr['FPR']:.1%}</div>
            <div style="font-size:0.85rem;">{worst_fpr['Label']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#4285f4,#1a73e8);color:white;padding:1.2rem;border-radius:12px;text-align:center;">
            <div style="font-size:0.75rem;opacity:0.9;text-transform:uppercase;letter-spacing:1px;">Best Accuracy</div>
            <div style="font-size:2rem;font-weight:700;">{best_acc['Accuracy']:.1%}</div>
            <div style="font-size:0.85rem;">{best_acc['Label']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#fbbc04,#d4a000);color:white;padding:1.2rem;border-radius:12px;text-align:center;">
            <div style="font-size:0.75rem;opacity:0.9;text-transform:uppercase;letter-spacing:1px;">Worst Accuracy</div>
            <div style="font-size:2rem;font-weight:700;">{worst_acc['Accuracy']:.1%}</div>
            <div style="font-size:0.85rem;">{worst_acc['Label']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ===== ROW 2: MODEL RANKINGS =====
    st.markdown("---")
    st.markdown("### Model Performance Rankings")
    
    # Rank by FPR (lower is better)
    model_fpr = plot_df.groupby('Model')['FPR'].mean().sort_values()
    model_acc = plot_df.groupby('Model')['Accuracy'].mean().sort_values(ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**By False Positive Rate (lower = better)**")
        for rank, (model, fpr) in enumerate(model_fpr.items(), 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:1rem;padding:0.5rem;margin:0.3rem 0;
                        background:rgba(66,133,244,0.05);border-radius:8px;border-left:4px solid #4285f4;">
                <span style="font-size:1.2rem;width:30px;">{medal}</span>
                <span style="font-weight:600;width:40px;">{model}</span>
                <div style="flex:1;height:8px;background:rgba(255,255,255,0.1);border-radius:4px;">
                    <div style="width:{fpr*100}%;height:100%;background:linear-gradient(90deg,#34a853,#ea4335);border-radius:4px;max-width:100%;"></div>
                </div>
                <span style="font-weight:600;min-width:50px;text-align:right;">{fpr:.1%}</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**By Accuracy (higher = better)**")
        for rank, (model, acc) in enumerate(model_acc.items(), 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:1rem;padding:0.5rem;margin:0.3rem 0;
                        background:rgba(52,168,83,0.05);border-radius:8px;border-left:4px solid #34a853;">
                <span style="font-size:1.2rem;width:30px;">{medal}</span>
                <span style="font-weight:600;width:40px;">{model}</span>
                <div style="flex:1;height:8px;background:rgba(255,255,255,0.1);border-radius:4px;">
                    <div style="width:{acc*100}%;height:100%;background:linear-gradient(90deg,#ea4335,#34a853);border-radius:4px;max-width:100%;"></div>
                </div>
                <span style="font-weight:600;min-width:50px;text-align:right;">{acc:.1%}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # ===== ROW 3: VISUALIZATIONS =====
    st.markdown("---")
    st.markdown("### Comparative Visualizations")
    
    # FPR by Condition and Model
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            plot_df, x='Condition', y='FPR', color='Model', barmode='group',
            title="False Positive Rate by Condition",
            labels={'FPR': 'False Positive Rate'},
            color_discrete_map={'DT': '#4285f4', 'LR': '#34a853', 'NN': '#fbbc04'},
            text=plot_df['FPR'].apply(lambda x: f'{x:.1%}')
        )
        fig.update_traces(textposition='outside', textfont=dict(size=11))
        fig.update_layout(template="plotly_white", title_x=0.5, xaxis_tickangle=-45,
                         height=400, yaxis_tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True, key="research_fpr_bar")
    
    with col2:
        fig = px.bar(
            plot_df, x='Condition', y='Accuracy', color='Model', barmode='group',
            title="Recombinant Accuracy by Condition",
            labels={'Accuracy': 'Accuracy'},
            color_discrete_map={'DT': '#4285f4', 'LR': '#34a853', 'NN': '#fbbc04'},
            text=plot_df['Accuracy'].apply(lambda x: f'{x:.1%}')
        )
        fig.update_traces(textposition='outside', textfont=dict(size=11))
        fig.update_layout(template="plotly_white", title_x=0.5, xaxis_tickangle=-45,
                         height=400, yaxis_tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True, key="research_acc_bar")
    
    # ===== ROW 4: FPR vs Accuracy SCATTER =====
    st.markdown("---")
    st.markdown("### FPR vs Accuracy Trade-off")
    
    fig = px.scatter(
        plot_df, x='FPR', y='Accuracy', color='Model', size='Matched_RDP',
        hover_data=['Label', 'Orig_RDP'],
        title="Performance Space: FPR vs Accuracy (size = Matched Events)",
        color_discrete_map={'DT': '#4285f4', 'LR': '#34a853', 'NN': '#fbbc04'},
        text='Label'
    )
    fig.update_traces(textposition='top center', textfont=dict(size=9))
    fig.update_layout(template="plotly_white", title_x=0.5, height=500,
                     xaxis_tickformat='.0%', yaxis_tickformat='.0%')
    # Add ideal point annotation
    fig.add_annotation(x=0, y=1, text="IDEAL", showarrow=True, arrowhead=1,
                      font=dict(color='#34a853', size=12), ax=20, ay=-20)
    st.plotly_chart(fig, use_container_width=True, key="research_scatter")
    
    # ===== ROW 5: CONDITION HEATMAP =====
    st.markdown("---")
    st.markdown("### Performance Heatmap by Condition")
    
    # Create pivot tables
    fpr_pivot = plot_df.pivot(index='Condition', columns='Model', values='FPR')
    acc_pivot = plot_df.pivot(index='Condition', columns='Model', values='Accuracy')
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.imshow(
            fpr_pivot, text_auto='.1%', aspect='auto',
            title="FPR Heatmap (lower = better)",
            color_continuous_scale='RdYlGn_r',
            labels={'color': 'FPR'}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True, key="fpr_heatmap")
    
    with col2:
        fig = px.imshow(
            acc_pivot, text_auto='.1%', aspect='auto',
            title="Accuracy Heatmap (higher = better)",
            color_continuous_scale='RdYlGn',
            labels={'color': 'Accuracy'}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True, key="acc_heatmap")
    
    # ===== ROW 6: DATA TABLE =====
    st.markdown("---")
    with st.expander("View Complete Data Table"):
        st.dataframe(summary_df, use_container_width=True, height=400)
    
    # ===== DOWNLOAD =====
    csv = summary_df.to_csv(index=False)
    st.download_button(
        label="Download Complete Analysis Data",
        data=csv,
        file_name="research_comparison_all_conditions.csv",
        mime="text/csv",
        key="download_research"
    )
# ============================================================================
# EXISTING DISPLAY FUNCTIONS
# ============================================================================

def display_merged_analysis(_original_df: pd.DataFrame, _result_df: pd.DataFrame,
                            condition: str, model: str):
    """
    Merged display: Original Dataset Analysis + Combined Comparison.
    Single professional view with all key metrics.
    """
    original_df = _original_df  # Unpack inside
    result_df = _result_df 
    from analysis import (
        analyze_original_by_tool, extract_tag_level_stats,
        compute_comparison_stats, compute_summary_totals, get_missing_tags_list
    )
    
    st.markdown("---")
    st.markdown("## Dataset Analysis & Comparison")
    
    # ===== ANALYZE ORIGINAL =====
    if original_df is not None:
        orig_stats = analyze_original_by_tool(original_df)
    else:
        orig_stats = {
            'per_tag': {}, 'total_rdp_events': 0, 'total_santa_events': 0,
            'total_events': 0, 'tag_count': 0, 'total_rows': 0,
            'missing_tags': list(range(1, 101)), 'zero_rdp_tags': [], 'zero_santa_tags': [],
            'tags_with_rdp': [], 'tags_with_santa': [], 'tags_with_both': [], 'tags_with_neither': [],
        }
    
    # ===== EXTRACT RESULT STATS =====
    result_tag_df = extract_tag_level_stats(result_df)
    comparison_df = compute_comparison_stats(orig_stats, result_tag_df)
    totals = compute_summary_totals(comparison_df, orig_stats)
    missing_list = get_missing_tags_list(comparison_df)
    
    # ===== ROW 1: ORIGINAL DATASET OVERVIEW =====
    st.markdown("### Original Simulation Dataset")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Total Rows", orig_stats['total_rows'])
    with col2:
        st.metric("Unique Tags", orig_stats['tag_count'])
    with col3:
        st.metric("Total RDP Events", f"{orig_stats['total_rdp_events']:,}")
    with col4:
        st.metric("Total SANTA Events", f"{orig_stats['total_santa_events']:,}")
    with col5:
        st.metric("Tags with RDP", len(orig_stats['tags_with_rdp']))
    with col6:
        st.metric("Tags with SANTA", len(orig_stats['tags_with_santa']))
    
    # Zero event warnings
    warnings = []
    if orig_stats['zero_rdp_tags']:
        warnings.append(f"{len(orig_stats['zero_rdp_tags'])} tags with zero RDP")
    if orig_stats['zero_santa_tags']:
        warnings.append(f"{len(orig_stats['zero_santa_tags'])} tags with zero SANTA")
    if orig_stats.get('missing_tags'):
        warnings.append(f"{len(orig_stats['missing_tags'])} tags missing from original")
    
    if warnings:
        st.warning(" · ".join(warnings))
    
    # Tags summary bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("RDP+ SANTA+", len(orig_stats['tags_with_both']))
    with col2:
        st.metric("RDP+ SANTA−", len(set(orig_stats['tags_with_rdp']) - set(orig_stats['tags_with_santa'])))
    with col3:
        st.metric("RDP− SANTA+", len(set(orig_stats['tags_with_santa']) - set(orig_stats['tags_with_rdp'])))
    with col4:
        st.metric("RDP− SANTA−", len(orig_stats['tags_with_neither']))
    
    # ===== ROW 2: MATCHING RESULTS =====
    # ===== ROW 2: MATCHING RESULTS =====
    st.markdown("---")
    st.markdown("### Matching Results")
    
    total_orig_rdp = totals.get('total_original_rdp', 0)
    total_orig_santa = totals.get('total_original_santa', 0)
    total_matched_rdp = totals.get('total_result_rdp', 0)
    total_matched_santa = totals.get('total_result_santa', 0)
    total_remaining_rdp = totals.get('total_unmatched_rdp', 0)
    total_remaining_santa = totals.get('total_unmatched_santa', 0)
    
    # ===== NEW: RDP match rate for RDP+ SANTA+ tags only =====
    rdp_match_pct = totals.get('rdp_match_rate_eligible', 0)
    total_eligible_rdp = totals.get('total_rdp_in_eligible_tags', total_orig_rdp)
    matched_eligible_rdp = totals.get('total_matched_rdp_in_eligible_tags', total_matched_rdp)
    
    # Main matching metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        orig_total = total_orig_rdp + total_orig_santa
        matched_total = total_matched_rdp + total_matched_santa
        match_rate = (matched_total / orig_total * 100) if orig_total > 0 else 0
        st.metric("Overall Match Rate", f"{match_rate:.1f}%",
                 f"{matched_total:,} / {orig_total:,}")
    with col2:
        # FIXED: Use RDP+ SANTA+ eligible rate
        st.metric("RDP Match Rate", f"{rdp_match_pct:.1f}%",
                 f"{matched_eligible_rdp:,} / {total_eligible_rdp:,}")
    with col3:
        santa_match_rate = (total_matched_santa / total_orig_santa * 100) if total_orig_santa > 0 else 0
        st.metric("SANTA Match Rate", f"{santa_match_rate:.1f}%",
                 f"{total_matched_santa:,} / {total_orig_santa:,}")
    with col4:
        fpr = (total_remaining_rdp / total_matched_rdp * 100) if total_matched_rdp > 0 else 0
        st.metric("False Positive Rate", f"{fpr:.1f}%",
                 f"{total_remaining_rdp:,} unmatched RDP")
    # ===== ROW 3: EVENT FLOW VISUALIZATION =====
    st.markdown("---")
    st.markdown("### Event Flow: Original → Matched → Remaining")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**RDP Events**")
        # Flow: Original → Matched + Remaining
        rdp_data = pd.DataFrame({
            'Stage': ['Original RDP', 'Matched RDP', 'Remaining RDP'],
            'Count': [total_orig_rdp, total_matched_rdp, total_remaining_rdp]
        })
        fig = px.bar(rdp_data, x='Stage', y='Count', text='Count',
                     color='Stage', color_discrete_sequence=['#4285f4', '#34a853', '#ea4335'])
        fig.update_traces(textposition='outside', textfont=dict(size=14, color='white'))
        fig.update_layout(showlegend=False, height=300, margin=dict(t=20), template="plotly_white")
        st.plotly_chart(fig, use_container_width=True, key=f"rdp_flow_{condition}_{model}")
    
    with col2:
        st.markdown("**SANTA Events**")
        santa_data = pd.DataFrame({
            'Stage': ['Original SANTA', 'Matched SANTA', 'Remaining SANTA'],
            'Count': [total_orig_santa, total_matched_santa, total_remaining_santa]
        })
        fig = px.bar(santa_data, x='Stage', y='Count', text='Count',
                     color='Stage', color_discrete_sequence=['#4285f4', '#34a853', '#ea4335'])
        fig.update_traces(textposition='outside', textfont=dict(size=14, color='white'))
        fig.update_layout(showlegend=False, height=300, margin=dict(t=20), template="plotly_white")
        st.plotly_chart(fig, use_container_width=True, key=f"santa_flow_{condition}_{model}")
    
    # ===== ROW 4: TAGS STATUS =====
    st.markdown("---")
    st.markdown("### Tags in Results")
    
    tags_in_results = totals.get('tags_in_results', 0)
    tags_in_original = totals.get('tags_in_original', 0)
    missing_in_results = missing_list.get('missing_in_results', [])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tags in Results", f"{tags_in_results} / {tags_in_original}")
    with col2:
        st.metric("Missing in Results", len(missing_in_results))
    with col3:
        if 'Match_Status' in comparison_df.columns:
            exact = (comparison_df['Match_Status'] == '✅ Exact Match').sum()
            st.metric("Exact Matches", exact)
    
    if missing_in_results:
        missing_rdp = sum(orig_stats['per_tag'].get(t, {}).get('rdp_count', 0) for t in missing_in_results)
        missing_santa = sum(orig_stats['per_tag'].get(t, {}).get('santa_count', 0) for t in missing_in_results)
        st.warning(f"Missing tags: {len(missing_in_results)} tags (RDP: {missing_rdp}, SANTA: {missing_santa})")
    
    # ===== ROW 5: PER-TAG COMPARISON TABLE =====
    with st.expander("Per-Tag Comparison Table"):
        if 'Match_Status' in comparison_df.columns:
            status_counts = comparison_df['Match_Status'].value_counts()
            status_cols = st.columns(len(status_counts))
            for i, (status, count) in enumerate(status_counts.items()):
                with status_cols[i]:
                    st.metric(status, count)
        
        st.dataframe(comparison_df, use_container_width=True, height=400)
    
    # ===== ROW 6: PER-TAG ORIGINAL SUMMARY =====
    if orig_stats['per_tag']:
        with st.expander("Per-Tag Original Event Summary"):
            tag_data = []
            for tag, data in orig_stats['per_tag'].items():
                tag_data.append({
                    'Tag': tag, 'RDP': data['rdp_count'], 'SANTA': data['santa_count'],
                    'Total': data['total'], 'RDP?': '✓' if data['has_rdp'] else '✗',
                    'SANTA?': '✓' if data['has_santa'] else '✗',
                })
            tag_df = pd.DataFrame(tag_data).sort_values('Total', ascending=False)
            st.dataframe(tag_df, use_container_width=True, height=300)
    
    # ===== DOWNLOADS =====
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        csv_comp = comparison_df.to_csv(index=False)
        st.download_button("Download Comparison Table", csv_comp,
                          f"{condition}_{model}_comparison.csv", key=f"dl_comp_{condition}_{model}")
    with col2:
        if not result_tag_df.empty:
            csv_tag = result_tag_df.to_csv(index=False)
            st.download_button("Download Per-Tag Stats", csv_tag,
                              f"{condition}_{model}_pertag.csv", key=f"dl_tag_{condition}_{model}")
    with col3:
        summary = pd.DataFrame({
            'Metric': ['Original RDP', 'Original SANTA', 'Matched RDP', 'Matched SANTA',
                      'Remaining RDP', 'Remaining SANTA', 'Overall Match Rate', 
                      'RDP Match Rate (RDP+ SANTA+)', 'SANTA Match Rate', 'FPR',
                      'Eligible RDP (RDP+ SANTA+ tags)', 'Matched RDP in Eligible Tags'],
            'Value': [total_orig_rdp, total_orig_santa, total_matched_rdp, total_matched_santa,
                     total_remaining_rdp, total_remaining_santa, f"{match_rate:.1f}%", 
                     f"{rdp_match_pct:.1f}%", f"{santa_match_rate:.1f}%", f"{fpr:.1f}%",
                     total_eligible_rdp, matched_eligible_rdp]
        })
        st.download_button("Download Summary", summary.to_csv(index=False),
                          f"{condition}_{model}_summary.csv", key=f"dl_sum_{condition}_{model}")

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






"""
Statistical Display Components for Streamlit
=============================================
Displays hypothesis tests, confidence intervals, and effect sizes.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats


def display_statistical_analysis_section(
    all_per_replicate_data: Dict,
    condition: str,
    models: List[str]
    ):
    """
    Main statistical analysis display section.
    
    Args:
        all_per_replicate_data: Dict[condition][model] = per_replicate DataFrame
        condition: Current condition
        models: List of model names
    """
    from analysis import (
        compute_all_statistics,
        compute_all_pairwise_tests,
        compute_summary_with_ci,
        format_p_value,
    )
    
    st.markdown("---")
    st.markdown("## 📊 Statistical Analysis")
    st.caption("Paired hypothesis tests · 95% Confidence Intervals · Effect Sizes")
    
    # Get data for this condition
    cond_data = all_per_replicate_data.get(condition, {})
    
    if not cond_data:
        st.warning("No per-replicate data available for statistical analysis.")
        st.info("Statistical tests require per-replicate data (not aggregated). " +
                "Please ensure per-replicate metrics are computed.")
        return
    
    # Combine data
    combined_list = []
    for model in models:
        df = cond_data.get(model)
        if df is not None and not df.empty:
            df_copy = df.copy()
            df_copy['Model'] = model
            combined_list.append(df_copy)
    
    if len(combined_list) < 2:
        st.warning("Need at least 2 models for comparison.")
        return
    
    combined_df = pd.concat(combined_list, ignore_index=True)
    
    # ===== TAB 1: SUMMARY STATISTICS =====
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Summary Stats", "🔄 Pairwise Tests", "📈 Visualization", "📝 Report"
    ])
    
    with tab1:
        _display_summary_statistics(combined_df, models)
    
    with tab2:
        _display_pairwise_tests(combined_df, models)
    
    with tab3:
        _display_statistical_visualizations(combined_df, models)
    
    with tab4:
        _display_statistical_report(combined_df, models, condition)


def _display_summary_statistics(combined_df: pd.DataFrame, models: List[str]):
    """Display summary statistics table."""
    from analysis import compute_summary_with_ci
    
    st.markdown("### Descriptive Statistics by Model")
    st.caption("Mean ± SD with 95% Confidence Intervals")
    
    metrics = ['Accuracy', 'FPR', 'BP_Distance']
    available_metrics = [m for m in metrics if m in combined_df.columns]
    
    for metric in available_metrics:
        st.markdown(f"**{metric.replace('_', ' ')}**")
        
        rows = []
        for model in models:
            model_data = combined_df[combined_df['Model'] == model][metric].dropna()
            
            if len(model_data) < 3:
                continue
            
            summary = compute_summary_with_ci(
                pd.DataFrame({metric: model_data}), metric
            )
            
            rows.append({
                'Model': model,
                'N': summary['n'],
                'Mean': f"{summary['mean']:.4f}",
                'SD': f"{summary['std']:.4f}",
                '95% CI': f"[{summary['ci_95_lower']:.4f}, {summary['ci_95_upper']:.4f}]",
                'Median': f"{summary['median']:.4f}",
                'IQR': f"{summary['iqr']:.4f}",
            })
        
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.markdown("")
    
    # Download button
    summary_csv = combined_df.groupby('Model').agg(['mean', 'std', 'count']).to_csv()
    st.download_button(
        "📥 Download Summary Statistics",
        summary_csv,
        "summary_statistics.csv",
        key="dl_summary_stats"
    )


def _display_pairwise_tests(combined_df: pd.DataFrame, models: List[str]):
    """Display pairwise comparison tests."""
    from analysis import compute_all_pairwise_tests, format_p_value
    
    st.markdown("### Pairwise Model Comparisons")
    st.caption("Paired tests with Bonferroni correction for multiple comparisons")
    
    metrics = ['Accuracy', 'FPR', 'BP_Distance']
    available_metrics = [m for m in metrics if m in combined_df.columns]
    
    # Test selection
    test_type = st.radio(
        "Test type:",
        ['auto (Shapiro-Wilk normality check)', 'paired t-test', 'Wilcoxon signed-rank'],
        horizontal=True,
        key="test_type_radio"
    )
    
    if 'auto' in test_type:
        test_method = 'auto'
    elif 't-test' in test_type:
        test_method = 'ttest'
    else:
        test_method = 'wilcoxon'
    
    for metric in available_metrics:
        st.markdown(f"**{metric.replace('_', ' ')}**")
        
        tests_df = compute_all_pairwise_tests(combined_df, metric)
        
        if not tests_df.empty:
            # Format for display
            display_df = tests_df.copy()
            display_df['P_Value'] = display_df['P_Value'].apply(format_p_value)
            display_df['Mean_Diff'] = display_df['Mean_Diff'].round(4)
            display_df["Cohen's d"] = display_df["Cohen's d"].round(3)
            display_df['CI_95'] = display_df.apply(
                lambda r: f"[{r['CI_95_Lower']:.4f}, {r['CI_95_Upper']:.4f}]", axis=1
            )
            
            display_cols = [
                'Comparison', 'N_Pairs', 'Mean_Diff', 'CI_95',
                'Test', 'P_Value', "Cohen's d", 'Significant (Bonferroni)'
            ]
            
            st.dataframe(
                display_df[display_cols],
                use_container_width=True,
                hide_index=True,
            )
            
            # Effect size interpretation
            st.caption(
                "Cohen's d: small = 0.2, medium = 0.5, large = 0.8 | "
                "Significance: * p<.05, ** p<.01, *** p<.001 (Bonferroni corrected)"
            )
        else:
            st.info(f"Insufficient data for {metric} pairwise comparisons.")
        
        st.markdown("")
    
    # Download
    all_tests = []
    for metric in available_metrics:
        tests = compute_all_pairwise_tests(combined_df, metric)
        if not tests.empty:
            tests['Metric'] = metric
            all_tests.append(tests)
    
    if all_tests:
        all_tests_df = pd.concat(all_tests, ignore_index=True)
        st.download_button(
            "📥 Download All Pairwise Tests",
            all_tests_df.to_csv(index=False),
            "pairwise_tests.csv",
            key="dl_pairwise"
        )


def _display_statistical_visualizations(combined_df: pd.DataFrame, models: List[str]):
    """Display statistical visualizations."""
    
    metrics = ['Accuracy', 'FPR', 'BP_Distance']
    available_metrics = [m for m in metrics if m in combined_df.columns]
    
    if not available_metrics:
        return
    
    # Metric selection
    metric = st.selectbox("Select metric:", available_metrics, key="stat_viz_metric")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plot with individual points
        fig = px.box(
            combined_df, x='Model', y=metric, color='Model',
            points='all',  # Show all data points
            title=f"{metric.replace('_', ' ')} Distribution by Model",
            color_discrete_map={'DT': '#4285f4', 'LR': '#34a853', 'NN': '#fbbc04'}
        )
        fig.update_layout(
            template="plotly_white",
            title_x=0.5,
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True, key=f"box_{metric}")
    
    with col2:
        # Violin plot
        fig = px.violin(
            combined_df, x='Model', y=metric, color='Model',
            box=True, points='all',
            title=f"{metric.replace('_', ' ')} Violin Plot",
            color_discrete_map={'DT': '#4285f4', 'LR': '#34a853', 'NN': '#fbbc04'}
        )
        fig.update_layout(
            template="plotly_white",
            title_x=0.5,
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True, key=f"violin_{metric}")
    
    # Paired difference plot
    st.markdown("---")
    st.markdown("### Paired Differences")
    
    # Create paired differences for the best pair
    models_present = sorted(combined_df['Model'].unique())
    
    if len(models_present) >= 2:
        # Pivot to align replicates
        pivot = combined_df.pivot_table(
            index='Replicate',
            columns='Model',
            values=metric
        ).dropna()
        
        if len(pivot) >= 3 and len(models_present) >= 2:
            m1, m2 = models_present[0], models_present[1]
            
            # Check if these columns exist
            if m1 in pivot.columns and m2 in pivot.columns:
                diffs = pivot[m1] - pivot[m2]
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=diffs,
                    nbinsx=20,
                    name=f'{m1} - {m2}',
                    marker_color='#4285f4',
                    opacity=0.7
                ))
                fig.add_vline(
                    x=0, line_dash="dash", line_color="red",
                    annotation_text="No difference"
                )
                fig.add_vline(
                    x=np.mean(diffs), line_dash="solid", line_color="green",
                    annotation_text=f"Mean diff: {np.mean(diffs):.4f}"
                )
                
                fig.update_layout(
                    title=f"Paired Differences: {m1} vs {m2}",
                    xaxis_title=f"Difference in {metric.replace('_', ' ')}",
                    yaxis_title="Frequency",
                    template="plotly_white",
                    title_x=0.5,
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True, key=f"diff_hist_{metric}")
                
                # QQ plot for normality
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.scatter(
                        x=np.sort(stats.norm.ppf((np.arange(1, len(diffs)+1) - 0.5) / len(diffs))),
                        y=np.sort(diffs),
                        title="Q-Q Plot (Normality Check)",
                        labels={'x': 'Theoretical Quantiles', 'y': 'Sample Quantiles'}
                    )
                    # Add reference line
                    min_val = min(np.min(diffs), -3)
                    max_val = max(np.max(diffs), 3)
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val], y=[min_val, max_val],
                        mode='lines', line=dict(dash='dash', color='red'),
                        name='Normal'
                    ))
                    fig.update_layout(template="plotly_white", title_x=0.5, height=350, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True, key=f"qq_{metric}")
                
                with col2:
                    # Shapiro-Wilk test
                    stat, p = stats.shapiro(diffs)
                    st.markdown("**Normality Test (Shapiro-Wilk)**")
                    st.metric("W statistic", f"{stat:.4f}")
                    st.metric("p-value", f"{p:.4f}")
                    if p > 0.05:
                        st.success("✅ Differences are normally distributed → Use paired t-test")
                    else:
                        st.warning("⚠️ Differences are NOT normally distributed → Use Wilcoxon test")


def _display_statistical_report(combined_df: pd.DataFrame, models: List[str], condition: str):
    """Generate a statistical report text."""
    from analysis import compute_all_pairwise_tests, compute_summary_with_ci, format_p_value
    
    st.markdown("### 📝 Statistical Report")
    st.caption("APA-style summary for thesis chapter")
    
    metrics = ['Accuracy', 'FPR', 'BP_Distance']
    available_metrics = [m for m in metrics if m in combined_df.columns]
    
    report_lines = []
    report_lines.append(f"**Condition: {condition}**")
    report_lines.append("")
    
    for metric in available_metrics:
        metric_name = metric.replace('_', ' ')
        
        # Summaries
        summaries = {}
        for model in models:
            model_data = combined_df[combined_df['Model'] == model][metric].dropna()
            if len(model_data) >= 3:
                summaries[model] = compute_summary_with_ci(
                    pd.DataFrame({metric: model_data}), metric
                )
        
        # Pairwise tests
        tests = compute_all_pairwise_tests(combined_df, metric)
        
        report_lines.append(f"**{metric_name}**")
        report_lines.append("")
        
        # Descriptive stats
        for model, s in summaries.items():
            report_lines.append(
                f"- {model}: M = {s['mean']:.4f} (SD = {s['std']:.4f}), "
                f"95% CI [{s['ci_95_lower']:.4f}, {s['ci_95_upper']:.4f}], "
                f"N = {s['n']}"
            )
        
        report_lines.append("")
        
        # Test results
        # Test results
        if not tests.empty:
            for _, row in tests.iterrows():
                sig = "significant" if row['Significant (Bonferroni)'] else "not significant"
                p_str = format_p_value(row['P_Value'])
                cohens_d = row["Cohen's d"]  # Extract first to avoid f-string escaping
                report_lines.append(
                    f"- {row['Comparison']}: {row['Test']}, "
                    f"mean difference = {row['Mean_Diff']:.4f}, "
                    f"p = {p_str}, "
                    f"Cohen's d = {cohens_d:.3f} ({sig})"
                )
        
        report_lines.append("")
    
    report_text = "\n".join(report_lines)
    st.markdown(report_text)
    
    # Download report
    st.download_button(
        "📥 Download Statistical Report",
        report_text,
        f"statistical_report_{condition}.txt",
        key="dl_report"
    )


def display_global_statistical_comparison(all_per_replicate_data: Dict):
    """
    Global comparison across all conditions.
    Uses Friedman test for overall classifier comparison.
    """
    from analysis import (
        compute_all_statistics,
        friedman_test_across_conditions,
        format_p_value,
    )
    
    st.markdown("---")
    st.markdown("## 🌐 Global Statistical Comparison (All Conditions)")
    
    if not all_per_replicate_data:
        st.warning("No per-replicate data available.")
        return
    
    conditions = list(all_per_replicate_data.keys())
    models = set()
    for cond_data in all_per_replicate_data.values():
        models.update(cond_data.keys())
    models = sorted(models)
    
    # Combine all data
    all_data = []
    for condition in conditions:
        for model in models:
            df = all_per_replicate_data.get(condition, {}).get(model)
            if df is not None and not df.empty:
                df_copy = df.copy()
                df_copy['Condition'] = condition
                df_copy['Model'] = model
                all_data.append(df_copy)
    
    if not all_data:
        st.warning("No data to analyze.")
        return
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Friedman test per condition
    st.markdown("### Friedman Test (Non-parametric Repeated Measures ANOVA)")
    st.caption("Tests if classifiers differ significantly, accounting for condition effects")
    
    metrics = ['Accuracy', 'FPR', 'BP_Distance']
    available_metrics = [m for m in metrics if m in combined.columns]
    
    for metric in available_metrics:
        result = friedman_test_across_conditions(combined, metric)
        
        if result.get('test') == 'friedman':
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(f"**{metric}**", "")
            with col2:
                st.metric("χ²", f"{result['statistic']:.2f}")
            with col3:
                st.metric("p-value", format_p_value(result['p_value']))
            with col4:
                st.metric("Kendall's W", f"{result.get('kendall_w', 0):.3f}")
            
            if result['significant']:
                st.success(f"✅ Significant difference among classifiers for {metric}")
            else:
                st.info(f"ℹ️ No significant difference among classifiers for {metric}")
    
    # Global pairwise
    st.markdown("---")
    st.markdown("### Global Pairwise Comparisons (Across All Conditions)")
    
    from analysis import compute_all_pairwise_tests
    
    for metric in available_metrics:
        st.markdown(f"**{metric}**")
        tests = compute_all_pairwise_tests(combined, metric)
        if not tests.empty:
            display_df = tests.copy()
            display_df['P_Value'] = display_df['P_Value'].apply(format_p_value)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Download
    st.download_button(
        "📥 Download Global Statistics",
        combined.to_csv(index=False),
        "global_per_replicate_data.csv",
        key="dl_global_stats"
    )