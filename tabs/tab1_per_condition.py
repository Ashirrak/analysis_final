"""Tab 1: Per-Condition Analysis"""
import streamlit as st
import pandas as pd
from config import CONDITIONS_MAP, MODELS
from ui_components import display_merged_analysis  # Keep shared function


def render(all_data, summary, show_raw, conditions_map=CONDITIONS_MAP, models=MODELS):
    """Render Per-Condition Analysis tab"""
    st.markdown("### Per-Condition Analysis")
    st.caption("Detailed per-condition, per-model evaluation")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    condition_tabs = st.tabs(list(conditions_map.keys()))
    
    for i, (condition, params) in enumerate(conditions_map.items()):
        with condition_tabs[i]:
            # Condition header
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mutation Rate (μ)", f"{params['mutation_rate']:.2e}")
            with col2:
                st.metric("Recombination Rate (r)", f"{params['recomb_rate']:.3f}")
            with col3:
                available = summary['by_condition'].get(condition, {}).get('csv', 0)
                st.metric("Models Available", f"{available}/3")
            
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            
            # Model tabs
            model_tabs = st.tabs([f"{model}" for model in models])
            
            for j, model in enumerate(models):
                with model_tabs[j]:
                    original_df, result_df = all_data[condition][model]
                    
                    if result_df is not None:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption(f"Original: {'loaded' if original_df is not None else 'not found'} · {len(original_df) if original_df is not None else 0} rows")
                        with col2:
                            st.caption(f"Result: loaded · {len(result_df):,} rows")
                        
                        # Single merged analysis call
                        display_merged_analysis(original_df, result_df, condition, model)
                        
                        # ===== DEBUG: Verify RDP counts =====
                        with st.expander("🔍 DEBUG: RDP Count Verification", expanded=False):
                            # Count from RDP_Event_ID (splitting chains)
                            all_rdp_ids = set()
                            rdp_event_id_col = None
                            for col in ['RDP_Event_ID', 'rdp_event_id', 'RDP_Event_ID_Normalized']:
                                if col in result_df.columns:
                                    rdp_event_id_col = col
                                    break
                            
                            if rdp_event_id_col:
                                for _, row in result_df.iterrows():
                                    val = str(row[rdp_event_id_col])
                                    if val and val != 'nan':
                                        for part in val.replace(',', ' ').split():
                                            part_clean = part.strip().rstrip('.0')
                                            if part_clean.isdigit():
                                                all_rdp_ids.add(part_clean)
                                st.write(f"**RDP_Event_ID column used:** `{rdp_event_id_col}`")
                                st.write(f"**Total unique RDP IDs found:** {len(all_rdp_ids)}")
                                st.write(f"**Sample RDP IDs:** {sorted(list(all_rdp_ids), key=int)[:20]}")
                            
                            # Count from RDP_Set if available
                            if 'RDP_Set' in result_df.columns:
                                rdp_set_total = set()
                                for _, row in result_df.iterrows():
                                    rdp_set_val = row.get('RDP_Set', set())
                                    if isinstance(rdp_set_val, (set, frozenset)):
                                        rdp_set_total.update(rdp_set_val)
                                    elif isinstance(rdp_set_val, str) and rdp_set_val and rdp_set_val != 'nan':
                                        for part in rdp_set_val.replace(',', ' ').split():
                                            part_clean = part.strip().rstrip('.0')
                                            if part_clean.isdigit():
                                                rdp_set_total.add(part_clean)
                                st.write(f"**RDP_Set total unique IDs:** {len(rdp_set_total)}")
                            
                            # Count by match type
                            if 'Match_Type' in result_df.columns:
                                match_types = result_df['Match_Type'].value_counts()
                                st.write("**Match Types:**")
                                st.dataframe(match_types)
                            
                            # Count by source tab
                            if 'Source_Tab' in result_df.columns:
                                source_tabs = result_df['Source_Tab'].value_counts()
                                st.write("**Source Tabs:**")
                                st.dataframe(source_tabs)
                            
                            # Row count vs RDP count
                            st.write(f"**Result rows:** {len(result_df)}")
                            st.write(f"**Unique RDP count:** {len(all_rdp_ids)}")
                            st.write(f"**Difference:** {len(all_rdp_ids)} - {len(result_df)} = {len(all_rdp_ids) - len(result_df)} (extra from chains)")
                        if show_raw:
                            with st.expander("Raw Result Data"):
                                st.dataframe(result_df, use_container_width=True, height=400)
                    else:
                        st.warning(f"No result data for {condition}_{model}.csv")

