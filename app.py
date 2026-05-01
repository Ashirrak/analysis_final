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
    display_unmatched_events_table,
    display_study_metrics_section,
)


# ===== CACHE ALL DATA LOADING =====
@st.cache_data(ttl=3600)
def load_all_data():
    """Load all files once and cache them."""
    all_data = {}
    for condition in CONDITIONS_MAP.keys():
        all_data[condition] = {}
        for model in MODELS:
            all_data[condition][model] = load_both_files(condition, model)
    return all_data


@st.cache_data(ttl=3600)
def get_cached_summary():
    """Cache file availability summary."""
    return get_available_files_summary()


def main():
    """Main Streamlit application."""
    
    st.set_page_config(
        page_title="Genomic Simulation Analysis Suite",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # ===== ADVANCED RESEARCH-LEVEL CSS =====
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        * { font-family: 'Inter', sans-serif; }
        
        /* Main Header */
        .main-header {
            background: linear-gradient(135deg, #0a1628 0%, #1a3a5c 30%, #1e3c72 60%, #2a5298 100%);
            padding: 2.5rem 3rem;
            border-radius: 20px;
            margin-bottom: 2.5rem;
            box-shadow: 0 20px 60px rgba(10, 22, 40, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
            position: relative;
            overflow: hidden;
        }
        .main-header::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -10%;
            width: 300px;
            height: 300px;
            background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 70%);
            border-radius: 50%;
        }
        .main-header h1 {
            color: #ffffff;
            font-size: 2.8rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.5px;
        }
        .main-header p {
            color: #b8c7e0;
            font-size: 1.1rem;
            margin-top: 0.5rem;
            font-weight: 300;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0a1628 0%, #0f1f3a 50%, #0a1628 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        [data-testid="stSidebar"] * {
            color: #e0e8f0 !important;
        }
        
        /* Sidebar Section Headers */
        .sidebar-section {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 12px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        /* Condition Cards in Sidebar */
        .condition-card {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-left: 4px solid #2a5298;
            border-radius: 10px;
            padding: 0.8rem 1rem;
            margin: 0.4rem 0;
            transition: all 0.2s ease;
            cursor: default;
        }
        .condition-card:hover {
            background: rgba(42, 82, 152, 0.15);
            border-left-color: #4a7ad4;
            transform: translateX(3px);
        }
        .condition-card strong {
            color: #4a7ad4;
            font-size: 1.1rem;
        }
        .condition-card .param {
            color: #b8c7e0;
            font-size: 0.85rem;
        }
        
        /* Metric Cards */
        .metric-card {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
            backdrop-filter: blur(10px);
        }
        .metric-value {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .metric-label {
            font-size: 0.85rem;
            color: #8899aa;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-top: 0.3rem;
        }
        
        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 14px;
            padding: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: 500;
            color: #8899aa;
            border: none;
            transition: all 0.2s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(255, 255, 255, 0.05);
            color: #c0c8d8;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        /* Info/Success/Warning Boxes */
        .info-box {
            background: rgba(33, 150, 243, 0.08);
            border: 1px solid rgba(33, 150, 243, 0.2);
            border-left: 4px solid #2196F3;
            border-radius: 10px;
            padding: 1rem 1.5rem;
            margin: 1rem 0;
        }
        .success-box {
            background: rgba(76, 175, 80, 0.08);
            border: 1px solid rgba(76, 175, 80, 0.2);
            border-left: 4px solid #4CAF50;
            border-radius: 10px;
            padding: 1rem 1.5rem;
            margin: 1rem 0;
        }
        .warning-box {
            background: rgba(255, 193, 7, 0.08);
            border: 1px solid rgba(255, 193, 7, 0.2);
            border-left: 4px solid #FFC107;
            border-radius: 10px;
            padding: 1rem 1.5rem;
            margin: 1rem 0;
        }
        
        /* DataFrames */
        [data-testid="stDataFrame"] {
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.06);
            overflow: hidden;
        }
        
        /* Expanders */
        [data-testid="stExpander"] {
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            overflow: hidden;
        }
        
        /* Progress Bar */
        .stProgress > div > div {
            background: linear-gradient(90deg, #667eea, #764ba2);
        }
        
        /* Metrics */
        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 12px;
            padding: 1rem;
        }
        [data-testid="stMetricValue"] {
            font-weight: 700;
            color: #2a5298;
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.2);
        }
        
        /* Section Headers */
        .section-header {
            font-size: 1.6rem;
            font-weight: 600;
            color: #e0e8f0;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid rgba(102, 126, 234, 0.3);
        }
        
        /* Status Indicators */
        .status-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 6px;
        }
        .status-dot.green { background: #4CAF50; box-shadow: 0 0 8px rgba(76, 175, 80, 0.5); }
        .status-dot.orange { background: #FF9800; box-shadow: 0 0 8px rgba(255, 152, 0, 0.5); }
        .status-dot.red { background: #F44336; box-shadow: 0 0 8px rgba(244, 67, 54, 0.5); }
        
        /* Divider */
        .divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
            margin: 1.5rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Load all data ONCE (cached)
    all_data = load_all_data()
    summary = get_cached_summary()
    
    # ===== HEADER =====
    st.markdown("""
    <div class="main-header">
        <h1>🧬 RDP5 AI Models Analysis</h1>
        <p>Recombination Detection Performance Evaluation | RDP-SANTA Comparative Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ===== SIDEBAR =====
    with st.sidebar:
        st.markdown("## 🧪 Experimental Setup")
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # File Availability Section
        st.markdown("### 📊 Data Availability")
        st.markdown(f"""
        <div class="sidebar-section">
            <table style="width:100%; color: #e0e8f0;">
                <tr>
                    <td><span class="status-dot green"></span> CSV Results</td>
                    <td style="text-align:right;"><strong>{summary['total_csv']}</strong>/27</td>
                </tr>
                <tr>
                    <td><span class="status-dot {'green' if summary['total_original'] > 10 else 'orange' if summary['total_original'] > 0 else 'red'}"></span> Original Files</td>
                    <td style="text-align:right;"><strong>{summary['total_original']}</strong>/27</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress bar
        if summary['total_csv'] > 0:
            progress = summary['total_csv'] / 27
            st.progress(progress, text=f"Overall Completion: {progress:.0%}")
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Analysis Controls
        st.markdown("### ⚙️ Display Options")
        show_raw = st.checkbox("Show Raw Data Preview", value=False, key="show_raw")
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Conditions Section
        st.markdown("### 🔬 Simulation Conditions")
        st.caption("Mutation rate (μ) × Recombination rate (r)")
        
        # Define mutation rate groups for styling
        mu_groups = {
            2.5e-5: {"name": "Low μ", "color": "#4CAF50", "icon": "🟢"},
            1e-4: {"name": "Medium μ", "color": "#FF9800", "icon": "🟡"},
            2e-4: {"name": "High μ", "color": "#F44336", "icon": "🔴"}
        }
        
        for cond, params in CONDITIONS_MAP.items():
            mu = params['mutation_rate']
            r = params['recomb_rate']
            
            # Get mutation rate group
            group = mu_groups.get(mu, {"color": "#999", "icon": "⚪"})
            
            # Determine available models count
            available = sum(1 for m in MODELS if all_data[cond][m][1] is not None)
            status_color = "green" if available == 3 else "orange" if available > 0 else "red"
            
            st.markdown(f"""
            <div class="condition-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>{cond}</strong>
                        <span style="color: {status_color}; font-size: 0.8rem; margin-left: 8px;">({available}/3 models)</span>
                    </div>
                    <span style="color: {group['color']}; font-size: 0.8rem;">{group['icon']}</span>
                </div>
                <div class="param">
                    μ = {mu:.2e} &nbsp;|&nbsp; r = {r:.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Legend
        with st.expander("📖 Legend"):
            st.markdown("""
            <div style="font-size: 0.85rem;">
                <p><span class="status-dot green"></span> <strong>C1-C3:</strong> Low mutation (μ=2.5e-5)</p>
                <p><span class="status-dot orange"></span> <strong>C4-C6:</strong> Medium mutation (μ=1e-4)</p>
                <p><span class="status-dot red"></span> <strong>C7-C9:</strong> High mutation (μ=2e-4)</p>
                <hr style="border-color: rgba(255,255,255,0.1);">
                <p>🔴 <strong>RDP:</strong> Recombination Detection Program</p>
                <p>🟢 <strong>SANTA:</strong> Simulated ANcestral Tree Analysis</p>
                <p>🤖 <strong>DT/LR/NN:</strong> Decision Tree / Logistic Regression / Neural Network</p>
                <hr style="border-color: rgba(255,255,255,0.1);">
                <p><strong>FPR:</strong> False Positive Rate</p>
                <p><strong>BP:</strong> Breakpoint Distance (circular, bp)</p>
            </div>
            """, unsafe_allow_html=True)
    
    # ===== MAIN TABS =====
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Individual Analysis",
        "📋 Summary Matrix",
        "🔬 Study Metrics",
        "📈 All Conditions",
        "❌ Unmatched Events",
    ])
    
    with tab1:
        st.markdown("### 🔍 Individual Condition Analysis")
        st.caption("Detailed per-condition, per-model evaluation")
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        condition_tabs = st.tabs(list(CONDITIONS_MAP.keys()))
        
        for i, (condition, params) in enumerate(CONDITIONS_MAP.items()):
            with condition_tabs[i]:
                # Condition header with style
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mutation Rate (μ)", f"{params['mutation_rate']:.2e}")
                with col2:
                    st.metric("Recombination Rate (r)", f"{params['recomb_rate']:.3f}")
                with col3:
                    available = summary['by_condition'].get(condition, {}).get('csv', 0)
                    st.metric("Models Available", f"{available}/3")
                
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                
                model_tabs = st.tabs([f"🤖 {model}" for model in MODELS])
                
                for j, model in enumerate(MODELS):
                    with model_tabs[j]:
                        original_df, result_df = all_data[condition][model]
                        
                        if result_df is not None:
                            # Status indicators
                            col1, col2 = st.columns(2)
                            with col1:
                                if original_df is not None:
                                    st.success(f"✅ Original dataset loaded")
                                else:
                                    st.warning(f"⚠️ Original dataset not found")
                            with col2:
                                st.success(f"✅ Result data loaded ({len(result_df):,} rows)")
                            
                            display_original_analysis(original_df, condition, model)
                            display_comparison_analysis(original_df, result_df, condition, model)
                            
                            if show_raw:
                                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                                st.subheader("📄 Raw Data Preview")
                                st.dataframe(result_df, use_container_width=True, height=400)
                        else:
                            st.warning(f"⚠️ No result data found for {condition}_{model}.csv")
    
    with tab2:
        st.markdown("### 📋 Data Availability Matrix")
        st.caption("Overview of available datasets across all conditions and models")
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        matrix_data = []
        for condition in CONDITIONS_MAP.keys():
            row = {'Condition': condition}
            for model in MODELS:
                orig, res = all_data[condition][model]
                if res is not None and orig is not None:
                    row[model] = '✅✅'
                elif res is not None:
                    row[model] = '✅'
                else:
                    row[model] = '❌'
            matrix_data.append(row)
        
        if matrix_data:
            matrix_df = pd.DataFrame(matrix_data)
            st.dataframe(
                matrix_df, 
                use_container_width=True,
                column_config={
                    "Condition": st.column_config.TextColumn("Condition", width="small"),
                    "DT": st.column_config.TextColumn("DT", width="small"),
                    "LR": st.column_config.TextColumn("LR", width="small"),
                    "NN": st.column_config.TextColumn("NN", width="small"),
                }
            )
            st.caption("✅✅ = Both files | ✅ = CSV only | ❌ = None")
    
    with tab3:
        st.markdown("## 🔬 Study Metrics — Individual Analysis")
        st.caption("False Positive Rate · Recombinant Accuracy · Breakpoint Distance")
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        condition_tabs = st.tabs(list(CONDITIONS_MAP.keys()))
        
        for i, (condition, params) in enumerate(CONDITIONS_MAP.items()):
            with condition_tabs[i]:
                st.markdown(f"### Condition {condition}")
                st.caption(f"μ = {params['mutation_rate']:.2e} | r = {params['recomb_rate']:.3f}")
                
                model_tabs = st.tabs([f"🤖 {model}" for model in MODELS])
                
                for j, model in enumerate(MODELS):
                    with model_tabs[j]:
                        original_df, result_df = all_data[condition][model]
                        
                        if result_df is not None:
                            display_study_metrics_section(result_df, original_df, condition, model)
                        else:
                            st.warning(f"No data for {condition}_{model}")
    
    with tab4:
        st.markdown("## 📈 All Conditions — Summary Comparison")
        st.caption("Cross-condition performance overview")
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        display_study_summary_all_conditions()
    
    with tab5:
        st.markdown("## ❌ Unmatched Events Analysis")
        st.caption("Original events that could not be matched — filtered by event ID per tag")
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        condition_tabs = st.tabs(list(CONDITIONS_MAP.keys()))
        
        for i, (condition, params) in enumerate(CONDITIONS_MAP.items()):
            with condition_tabs[i]:
                st.markdown(f"### Condition {condition}")
                
                model_tabs = st.tabs([f"🤖 {model}" for model in MODELS])
                
                for j, model in enumerate(MODELS):
                    with model_tabs[j]:
                        original_df, result_df = all_data[condition][model]
                        
                        if result_df is not None:
                            display_unmatched_events_table(result_df, original_df, condition, model)
                        else:
                            st.warning(f"No data for {condition}_{model}")


if __name__ == "__main__":
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    Path(ORIGINAL_DIR).mkdir(exist_ok=True)
    main()