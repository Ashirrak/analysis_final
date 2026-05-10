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
                        
                        if show_raw:
                            with st.expander("Raw Result Data"):
                                st.dataframe(result_df, use_container_width=True, height=400)
                    else:
                        st.warning(f"No result data for {condition}_{model}.csv")