"""Main Streamlit application for Genomic Simulation Analysis."""

import streamlit as st
import pandas as pd
from pathlib import Path

from config import CONDITIONS_MAP, MODELS, RESULTS_DIR, ORIGINAL_DIR
from data_loader import load_both_files, get_available_files_summary
from ui_components import (
    display_css,
    display_original_analysis,
    display_comparison_analysis,
    display_study_summary_all_conditions,
    # NEW MASTER-LEVEL IMPORTS
    display_study_metrics_section,
)


def main():
    """Main Streamlit application."""
    
    st.set_page_config(
        page_title="Genomic Simulation Analysis Suite",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    display_css()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧬 RDP5 AI Models Analysis </h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🧪 Experimental Parameters")
        st.markdown("---")
        
        st.markdown("### 📊 Analysis Controls")
        show_original = st.checkbox("Show Original Dataset Analysis", value=True)
        show_result_tags = st.checkbox("Show Result Per-Tag Analysis", value=True)
        show_comparison = st.checkbox("Show Combined Comparison", value=True)
        show_classification = st.checkbox("Show Classification Metrics", value=True)
        show_stability = st.checkbox("Show Stability Analysis", value=True)
        show_raw = st.checkbox("Show Raw Data Preview", value=False)
        
        st.markdown("---")
        st.markdown("### 📈 File Availability")
        
        summary = get_available_files_summary()
        
        csv_color = "green" if summary['total_csv'] > 0 else "red"
        orig_color = "green" if summary['total_original'] > 0 else "orange"
        
        st.markdown(f"**CSV Results:** :{csv_color}[{summary['total_csv']}/27]")
        st.markdown(f"**Original Files:** :{orig_color}[{summary['total_original']}/27]")
        
        # Debug: Show what files were found
        with st.expander("🔍 Available Files"):
            st.write("**Original Data Directory:**")
            if summary['available_files']['original_data']:
                for f in sorted(summary['available_files']['original_data']):
                    st.write(f"- {f}")
            else:
                st.write("No files found")
            
            st.write("**Results Directory:**")
            if summary['available_files']['results']:
                for f in sorted(summary['available_files']['results']):
                    st.write(f"- {f}")
            else:
                st.write("No files found")
            
            # Show detection status for each condition/model
            st.write("**Detection Status:**")
            from config import CONDITIONS_MAP, MODELS
            for condition in CONDITIONS_MAP.keys():
                for model in MODELS:
                    orig, res = load_both_files(condition, model)
                    if res is not None:
                        orig_icon = "✅" if orig is not None else "❌"
                        st.write(f"{condition}_{model}: CSV ✅ | Original {orig_icon}")
        
        st.markdown("---")
        st.markdown("### 🔬 Conditions")
        for cond, params in CONDITIONS_MAP.items():
            st.markdown(f"""
            <div class="condition-card">
                <strong>{cond}</strong><br>
                μ = {params['mutation_rate']:.2e}<br>
                r = {params['recomb_rate']:.3f}
            </div>
            """, unsafe_allow_html=True)
    
    # Main tabs - Now 7 tabs for comprehensive analysis
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Individual Analysis",
        "📋 Summary Matrix",
        "🔬 Study Metrics",
        "📊 All Conditions Study",

    ])
    
    with tab1:
        st.markdown("### 🔍 Individual Condition Analysis")
        
        condition_tabs = st.tabs(list(CONDITIONS_MAP.keys()))
        
        for i, (condition, params) in enumerate(CONDITIONS_MAP.items()):
            with condition_tabs[i]:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mutation Rate (μ)", f"{params['mutation_rate']:.2e}")
                with col2:
                    st.metric("Recombination Rate (r)", f"{params['recomb_rate']:.3f}")
                with col3:
                    available = summary['by_condition'].get(condition, {}).get('csv', 0)
                    st.metric("Available Models", f"{available}/3")
                
                st.markdown("---")
                
                model_tabs = st.tabs([f"🤖 {model}" for model in MODELS])
                
                for j, model in enumerate(MODELS):
                    with model_tabs[j]:
                        original_df, result_df = load_both_files(condition, model)
                        
                        if result_df is not None:
                            # File status
                            col1, col2 = st.columns(2)
                            with col1:
                                if original_df is not None:
                                    st.success(f"✅ Original loaded")
                                else:
                                    st.warning(f"⚠️ Original not found")
                            with col2:
                                st.success(f"✅ Result CSV loaded")
                            
                            # Basic info
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Result Rows", len(result_df))
                            with col2:
                                st.metric("Original Rows", len(original_df) if original_df is not None else 'N/A')
                            with col3:
                                st.metric("Result Columns", len(result_df.columns))
                            
                            # Original dataset analysis
                            if show_original and original_df is not None:
                                display_original_analysis(original_df, condition, model)
                            
                            
                            # Combined comparison
                            if show_comparison:
                                display_comparison_analysis(original_df, result_df, condition, model)
                            
                            # Raw data
                            if show_raw:
                                st.markdown("---")
                                st.subheader("📄 Raw Data Preview")
                                st.dataframe(result_df, use_container_width=True, height=400)
                        else:
                            st.warning(f"⚠️ No result CSV found for {condition}_{model}.csv")
    
    with tab2:
        st.markdown("### 📋 Summary Matrix - All Conditions")
        
        # Create summary matrix
        matrix_data = []
        for condition in CONDITIONS_MAP.keys():
            row = {'Condition': condition}
            for model in MODELS:
                orig, res = load_both_files(condition, model)
                if res is not None and orig is not None:
                    row[model] = '✅✅'
                elif res is not None:
                    row[model] = '✅'
                else:
                    row[model] = '❌'
            matrix_data.append(row)
        
        if matrix_data:
            matrix_df = pd.DataFrame(matrix_data)
            st.dataframe(matrix_df, use_container_width=True)
            st.caption("✅✅ = Both files | ✅ = CSV only | ❌ = None")
    

    with tab3:
        st.markdown("## 🔬 Study Metrics - Individual Analysis")
        st.markdown("*False Positive Rate, Recombinant Accuracy, Breakpoint Distance*")
        
        condition_tabs = st.tabs(list(CONDITIONS_MAP.keys()))
        
        for i, (condition, params) in enumerate(CONDITIONS_MAP.items()):
            with condition_tabs[i]:
                st.markdown(f"### Condition {condition}")
                st.markdown(f"μ = {params['mutation_rate']:.2e}, r = {params['recomb_rate']:.3f}")
                
                model_tabs = st.tabs([f"🤖 {model}" for model in MODELS])
                
                for j, model in enumerate(MODELS):
                    with model_tabs[j]:
                        original_df, result_df = load_both_files(condition, model)
                        
                        if result_df is not None:
                            display_study_metrics_section(result_df, original_df, condition, model)
                        else:
                            st.warning(f"No data for {condition}_{model}")
    
    with tab4:
        st.markdown("## 🔬 Study Metrics - All Conditions Summary")
        display_study_summary_all_conditions()
    



if __name__ == "__main__":
    # Create directories
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    Path(ORIGINAL_DIR).mkdir(exist_ok=True)
    
    main()
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🧬 Genomic Simulation Analysis Suite | Master-Level Recombination Detection Analysis</p>
    </div>
    """, unsafe_allow_html=True)