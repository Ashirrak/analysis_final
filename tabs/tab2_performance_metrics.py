"""Tab 2: Performance Metrics + Model Agreement"""
import streamlit as st
from config import CONDITIONS_MAP, MODELS
from ui_components import display_study_metrics_section, display_model_agreement_analysis


def render(all_data, conditions_map=CONDITIONS_MAP, models=MODELS):
    """Render Performance Metrics tab"""
    st.markdown("## 🔬 Study Metrics — Individual & Comparative Analysis")
    st.caption("False Positive Rate · Recombinant Accuracy · Breakpoint Distance · Model Agreement")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    condition_tabs = st.tabs(list(conditions_map.keys()))
    
    for i, (condition, params) in enumerate(conditions_map.items()):
        with condition_tabs[i]:
            st.markdown(f"### Condition {condition}")
            st.caption(f"μ = {params['mutation_rate']:.2e} | r = {params['recomb_rate']:.3f}")
            
            # 4 tabs: 3 individual models + 1 agreement
            model_tabs = st.tabs(
                [f"🤖 {model}" for model in models] + ["🔍 Model Agreement"]
            )
            
            # --- Individual Model Tabs ---
            for j, model in enumerate(models):
                with model_tabs[j]:
                    original_df, result_df = all_data[condition][model]
                    
                    if result_df is not None:
                        display_study_metrics_section(result_df, original_df, condition, model)
                    else:
                        st.warning(f"No data for {condition}_{model}")
            
            # --- Model Agreement Tab ---
            with model_tabs[3]:
                st.markdown("### 🔍 Model Agreement & Uniqueness Analysis")
                st.caption("Comparing matched events across DT, LR, and NN")
                display_model_agreement_analysis(all_data, condition, models)