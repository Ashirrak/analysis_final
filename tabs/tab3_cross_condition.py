"""Tab 3: Cross-Condition Summary"""
import streamlit as st
from ui_components import display_study_summary_all_conditions, display_research_comparison


def render():
    """Render Cross-Condition Summary tab"""
    st.markdown("## 📈 All Conditions — Summary Comparison")
    st.caption("Cross-condition performance overview")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    display_study_summary_all_conditions()
    
    st.markdown("## Cross-Condition Research Analysis")
    st.caption("Model comparison · Rankings · Performance trade-offs")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    display_research_comparison()