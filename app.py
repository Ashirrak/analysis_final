"""Main Streamlit application for Genomic Simulation Analysis."""

import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
from config import CONDITIONS_MAP, MODELS, RESULTS_DIR, ORIGINAL_DIR
from data_loader import load_both_files, get_available_files_summary
from ui_components import (
    display_css,
    display_study_summary_all_conditions,
    display_unmatched_events_table,
    display_study_metrics_section,
    display_merged_analysis,
    display_research_comparison,
    display_research_report,
    display_model_agreement_analysis,
)



def load_all_data():
    """Load all files once and cache them."""
    all_data = {}
    for condition in CONDITIONS_MAP.keys():
        all_data[condition] = {}
        for model in MODELS:
            all_data[condition][model] = load_both_files(condition, model)
    return all_data



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
    
    # ===== PROFESSIONAL BIOINFORMATICS CSS =====
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        * { font-family: 'Inter', sans-serif; }
        
        /* --- TAB STYLING --- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: linear-gradient(180deg, rgba(66, 133, 244, 0.03) 0%, rgba(66, 133, 244, 0.01) 100%);
            border-radius: 16px;
            padding: 8px 10px;
            border: 1px solid rgba(66, 133, 244, 0.08);
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 12px;
            padding: 12px 24px;
            font-weight: 500;
            font-size: 0.92rem;
            color: #7a8ea8;
            border: 1px solid transparent;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            letter-spacing: 0.2px;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(66, 133, 244, 0.12) !important;
            color: #b8d0f8 !important;
            border-color: rgba(66, 133, 244, 0.25);
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(66, 133, 244, 0.1);
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #1a73e8 0%, #4285f4 60%, #5b9cf5 100%) !important;
            color: #ffffff !important;
            font-weight: 600;
            border-color: transparent !important;
            box-shadow: 0 6px 20px rgba(26, 115, 232, 0.35);
            letter-spacing: 0.3px;
        }
        
        /* --- NESTED TABS (Conditions & Models) --- */
        .stTabs + .stTabs [data-baseweb="tab-list"] {
            background: linear-gradient(180deg, rgba(52, 168, 83, 0.03) 0%, rgba(52, 168, 83, 0.01) 100%);
            border: 1px solid rgba(52, 168, 83, 0.08);
            border-radius: 12px;
            padding: 5px 8px;
            gap: 5px;
        }
        .stTabs + .stTabs [data-baseweb="tab"] {
            padding: 8px 16px;
            font-size: 0.85rem;
            border-radius: 8px;
            color: #6a8a6a;
        }
        .stTabs + .stTabs [data-baseweb="tab"]:hover {
            background: rgba(52, 168, 83, 0.1) !important;
            color: #a0d8a0 !important;
            border-color: rgba(52, 168, 83, 0.2);
        }
        .stTabs + .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #1b8a3d 0%, #34a853 60%, #4caf50 100%) !important;
            box-shadow: 0 4px 15px rgba(52, 168, 83, 0.3);
        }
        
        /* --- THIRD-LEVEL TABS (Model DT/LR/NN) --- */
        .stTabs + .stTabs + .stTabs [data-baseweb="tab-list"] {
            background: linear-gradient(180deg, rgba(251, 188, 4, 0.03) 0%, rgba(251, 188, 4, 0.01) 100%);
            border: 1px solid rgba(251, 188, 4, 0.08);
            border-radius: 10px;
            padding: 4px 6px;
            gap: 4px;
        }
        .stTabs + .stTabs + .stTabs [data-baseweb="tab"] {
            padding: 6px 14px;
            font-size: 0.82rem;
            border-radius: 7px;
            color: #8a7a4a;
        }
        .stTabs + .stTabs + .stTabs [data-baseweb="tab"]:hover {
            background: rgba(251, 188, 4, 0.1) !important;
            color: #e8d060 !important;
            border-color: rgba(251, 188, 4, 0.2);
        }
        .stTabs + .stTabs + .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #d4a000 0%, #fbbc04 60%, #fdd835 100%) !important;
            box-shadow: 0 4px 12px rgba(251, 188, 4, 0.3);
            color: #1a1a1a !important;
        }
        
        /* --- MAIN HEADER --- */
        .main-header {
            background: linear-gradient(135deg, #0a1628 0%, #14213d 30%, #1a3a5c 60%, #1e3c72 100%);
            padding: 1rem 2rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            margin-top: -2rem;
            box-shadow: 0 8px 30px rgba(10, 22, 40, 0.25);
            border: 1px solid rgba(66, 133, 244, 0.12);
            position: relative;
            overflow: hidden;
        }
        .main-header::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -5%;
            width: 150px;
            height: 150px;
            background: radial-gradient(circle, rgba(66, 133, 244, 0.06) 0%, transparent 70%);
            border-radius: 50%;
        }
        .main-header h1 {
            color: #ffffff !important;
            font-size: 1.4rem !important;
            font-weight: 700 !important;
            margin: 0 !important;
            letter-spacing: -0.3px !important;
            padding: 0 !important;
        }
        .main-header p {
            color: #93b4e8 !important;
            font-size: 0.8rem !important;
            margin-top: 2px !important;
            font-weight: 300 !important;
        }
        
        /* --- SIDEBAR --- */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0a1628 0%, #0f1d35 50%, #0a1628 100%);
            border-right: 1px solid rgba(66, 133, 244, 0.1);
        }
        [data-testid="stSidebar"] * { color: #d0ddf0 !important; }
        
        .sidebar-section {
            background: rgba(66, 133, 244, 0.04);
            border: 1px solid rgba(66, 133, 244, 0.1);
            border-radius: 12px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .condition-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(66, 133, 244, 0.12);
            border-left: 4px solid #4285f4;
            border-radius: 10px;
            padding: 0.8rem 1rem;
            margin: 0.4rem 0;
            transition: all 0.25s ease;
            cursor: default;
        }
        .condition-card:hover {
            background: rgba(66, 133, 244, 0.12);
            border-left-color: #5b9cf5;
            transform: translateX(4px);
            box-shadow: 0 4px 15px rgba(66, 133, 244, 0.15);
        }
        .condition-card strong { color: #5b9cf5; font-size: 1.05rem; }
        .condition-card .param { color: #a0bbdf; font-size: 0.82rem; }
        
        /* --- BOXES --- */
        .info-box { background: rgba(66, 133, 244, 0.06); border: 1px solid rgba(66, 133, 244, 0.15); border-left: 4px solid #4285f4; border-radius: 10px; padding: 1rem 1.5rem; margin: 1rem 0; }
        .success-box { background: rgba(52, 168, 83, 0.06); border: 1px solid rgba(52, 168, 83, 0.15); border-left: 4px solid #34a853; border-radius: 10px; padding: 1rem 1.5rem; margin: 1rem 0; }
        .warning-box { background: rgba(251, 188, 4, 0.06); border: 1px solid rgba(251, 188, 4, 0.15); border-left: 4px solid #fbbc04; border-radius: 10px; padding: 1rem 1.5rem; margin: 1rem 0; }
        
        /* --- DATAFRAMES & EXPANDERS --- */
        [data-testid="stDataFrame"] { border-radius: 12px; border: 1px solid rgba(66, 133, 244, 0.1); overflow: hidden; }
        [data-testid="stExpander"] { border: 1px solid rgba(66, 133, 244, 0.12); border-radius: 12px; overflow: hidden; }
        
        /* --- PROGRESS BAR --- */
        .stProgress > div > div { background: linear-gradient(90deg, #1a73e8, #4285f4, #5b9cf5); }
        
        /* --- METRICS --- */
        [data-testid="stMetric"] { background: rgba(66, 133, 244, 0.03); border: 1px solid rgba(66, 133, 244, 0.08); border-radius: 12px; padding: 1rem; }
        [data-testid="stMetricValue"] { font-weight: 700; color: #4285f4; }
        
        /* --- SCROLLBAR --- */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: rgba(66, 133, 244, 0.03); border-radius: 3px; }
        ::-webkit-scrollbar-thumb { background: rgba(66, 133, 244, 0.2); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(66, 133, 244, 0.35); }
        
        /* --- STATUS DOTS --- */
        .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
        .status-dot.green { background: #34a853; box-shadow: 0 0 8px rgba(52, 168, 83, 0.5); }
        .status-dot.orange { background: #fbbc04; box-shadow: 0 0 8px rgba(251, 188, 4, 0.5); }
        .status-dot.red { background: #ea4335; box-shadow: 0 0 8px rgba(234, 67, 53, 0.5); }
        
        /* --- DIVIDER --- */
        .divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(66, 133, 244, 0.15), transparent); margin: 1.5rem 0; }
        
        /* --- SIDEBAR HEADERS --- */
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #5b9cf5 !important; font-weight: 600; }
        [data-testid="stCheckbox"] label { color: #d0ddf0 !important; }
    </style>
    """, unsafe_allow_html=True)
    
    all_data = load_all_data()
    summary = get_cached_summary()
    
    # ===== HEADER =====
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0a1628 0%, #14213d 30%, #1a3a5c 60%, #1e3c72 100%);
        padding: 1rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        margin-top: -3rem;
        box-shadow: 0 8px 30px rgba(10, 22, 40, 0.25);
        border: 1px solid rgba(66, 133, 244, 0.12);
        display: flex;
        align-items: center;
        gap: 1rem;
    ">
        <span style="font-size: 1.8rem;">🧬</span>
        <div>
            <h1 style="color:#fff;font-size:1.4rem;font-weight:700;margin:0;">RDP5 AI Models Analysis</h1>
            <p style="color:#93b4e8;font-size:0.8rem;margin:2px 0 0 0;font-weight:300;">Recombination Detection Performance · RDP–SANTA Comparative Analysis</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ===== SIDEBAR =====
    with st.sidebar:
        st.markdown("## Experimental Configuration")
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        st.markdown("### Data Status")
        st.markdown(f"""
        <div class="sidebar-section">
            <table style="width:100%; color: #e0e8f0;">
                <tr><td><span class="status-dot green"></span> Result Files</td><td style="text-align:right;"><strong>{summary['total_csv']}</strong>/27</td></tr>
                <tr><td><span class="status-dot {'green' if summary['total_original'] > 10 else 'orange' if summary['total_original'] > 0 else 'red'}"></span> Original Files</td><td style="text-align:right;"><strong>{summary['total_original']}</strong>/27</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        st.markdown("### Display Controls")
        show_raw = st.checkbox("Show raw data tables", value=True, key="show_raw")
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        st.markdown("### Simulation Parameters")
        st.caption("Mutation rate μ · Recombination rate r")
        
        mu_groups = {
            2.5e-5: {"color": "#34a853", "label": "Low"},
            1e-4: {"color": "#fbbc04", "label": "Medium"},
            2e-4: {"color": "#ea4335", "label": "High"}
        }
        
        for cond, params in CONDITIONS_MAP.items():
            mu = params['mutation_rate']
            r = params['recomb_rate']
            group = mu_groups.get(mu, {"color": "#999", "label": "?"})
            available = sum(1 for m in MODELS if all_data[cond][m][1] is not None)
            status_color = "#34a853" if available == 3 else "#fbbc04" if available > 0 else "#ea4335"
            
            st.markdown(f"""
            <div class="condition-card">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <strong>{cond}</strong>
                    <span style="color:{status_color};font-size:0.75rem;">{available}/3</span>
                </div>
                <div class="param">μ = {mu:.2e} · r = {r:.3f}</div>
                <div style="margin-top:4px;">
                    <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{group['color']};margin-right:4px;"></span>
                    <span style="font-size:0.7rem;color:#8899aa;">{group['label']} mutation</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        with st.expander("Legend"):
            st.markdown("""
            <div style="font-size:0.82rem;">
                <p><span class="status-dot green"></span> <strong>C1–C3</strong> — Low μ (2.5×10⁻⁵)</p>
                <p><span class="status-dot orange"></span> <strong>C4–C6</strong> — Medium μ (1×10⁻⁴)</p>
                <p><span class="status-dot red"></span> <strong>C7–C9</strong> — High μ (2×10⁻⁴)</p>
                <hr style="border-color:rgba(66,133,244,0.1);">
                <p><strong>RDP</strong> — Recombination Detection Program</p>
                <p><strong>SANTA</strong> — Simulated Ancestral Tree Analysis</p>
                <p><strong>DT</strong> Decision Tree · <strong>LR</strong> Logistic Regression · <strong>NN</strong> Neural Network</p>
                <hr style="border-color:rgba(66,133,244,0.1);">
                <p><strong>FPR</strong> — False Positive Rate</p>
                <p><strong>BP Dist</strong> — Breakpoint Distance (circular, bp)</p>
            </div>
            """, unsafe_allow_html=True)
    
    # ===== MAIN TABS =====
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "◈ Per-Condition Analysis",
        "◈ Performance Metrics",
        "◈ Cross-Condition Summary",
        "◈ Unmatched Events",
        "◈ Report",
    ])
    

    with tab1:
        st.markdown("### Per-Condition Analysis")
        st.caption("Detailed per-condition, per-model evaluation")
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
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
                    st.metric("Models Available", f"{available}/3")
                
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                
                model_tabs = st.tabs([f"{model}" for model in MODELS])
                
                for j, model in enumerate(MODELS):
                    with model_tabs[j]:
                        original_df, result_df = all_data[condition][model]
                        
                        if result_df is not None:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.caption(f"Original: {'loaded' if original_df is not None else 'not found'} · {len(original_df) if original_df is not None else 0} rows")
                            with col2:
                                st.caption(f"Result: loaded · {len(result_df):,} rows")
                            
                            # SINGLE MERGED FUNCTION CALL
                            display_merged_analysis(original_df, result_df, condition, model)
                            
                            if show_raw:
                                with st.expander("Raw Result Data"):
                                    st.dataframe(result_df, use_container_width=True, height=400)
                        else:
                            st.warning(f"No result data for {condition}_{model}.csv")
    
    with tab2:
        st.markdown("## 🔬 Study Metrics — Individual & Comparative Analysis")
        st.caption("False Positive Rate · Recombinant Accuracy · Breakpoint Distance · Model Agreement")
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        condition_tabs = st.tabs(list(CONDITIONS_MAP.keys()))
        
        for i, (condition, params) in enumerate(CONDITIONS_MAP.items()):
            with condition_tabs[i]:
                st.markdown(f"### Condition {condition}")
                st.caption(f"μ = {params['mutation_rate']:.2e} | r = {params['recomb_rate']:.3f}")
                
                # Now 4 tabs: 3 individual models + 1 comparison
                model_tabs = st.tabs(
                    [f"🤖 {model}" for model in MODELS] + ["🔍 Model Agreement"]
                )
                
                # --- Individual Model Tabs (existing code) ---
                for j, model in enumerate(MODELS):
                    with model_tabs[j]:
                        original_df, result_df = all_data[condition][model]
                        
                        if result_df is not None:
                            display_study_metrics_section(result_df, original_df, condition, model)
                        else:
                            st.warning(f"No data for {condition}_{model}")
                
                # --- NEW: Model Agreement Tab ---
                with model_tabs[3]:
                    st.markdown("### 🔍 Model Agreement & Uniqueness Analysis")
                    st.caption("Comparing matched events across DT, LR, and NN")
                    
                    display_model_agreement_analysis(all_data, condition,MODELS)
    with tab3:
        st.markdown("## 📈 All Conditions — Summary Comparison")
        st.caption("Cross-condition performance overview")
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        display_study_summary_all_conditions()
        st.markdown("## Cross-Condition Research Analysis")
        st.caption("Model comparison · Rankings · Performance trade-offs")
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        display_research_comparison()
    
    with tab4:
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

    with tab5:
        display_research_report()

if __name__ == "__main__":
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    Path(ORIGINAL_DIR).mkdir(exist_ok=True)
    main()