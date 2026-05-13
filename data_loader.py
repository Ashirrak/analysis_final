"""Data loading functions for original and result files."""

import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import streamlit as st
from config import RESULTS_DIR, ORIGINAL_DIR


def load_original_file(condition: str, model: str) -> Optional[pd.DataFrame]:
    """
    Load original Excel/CSV file for a specific condition and model.
    """
    patterns = [
        f"original_{condition}_{model}",      # original_C1_DT
        f"Original_{condition}_{model}",      # Original_C1_DT
        f"original_{condition}_{model}.xlsx", # original_C1_DT.xlsx
        f"original_{condition}_{model}.xls",  # original_C1_DT.xls
        f"{condition}_{model}",               # C1_DT
    ]
    
    extensions = ['.xlsx', '.xls', '.csv']
    
    # First check with exact filename
    exact_filename = f"original_{condition}_{model}.xlsx"
    filepath = Path(ORIGINAL_DIR) / exact_filename
    if filepath.exists():
        try:
            df = pd.read_excel(filepath, engine='openpyxl')
            return df
        except Exception as e:
            # Try with xlrd if openpyxl fails
            try:
                df = pd.read_excel(filepath, engine='xlrd')
                return df
            except:
                pass
    
    # Check for .xls extension
    exact_filename_xls = f"original_{condition}_{model}.xls"
    filepath = Path(ORIGINAL_DIR) / exact_filename_xls
    if filepath.exists():
        try:
            df = pd.read_excel(filepath)
            return df
        except Exception as e:
            pass
    
    # Try CSV
    exact_filename_csv = f"original_{condition}_{model}.csv"
    filepath = Path(ORIGINAL_DIR) / exact_filename_csv
    if filepath.exists():
        try:
            df = pd.read_csv(filepath)
            return df
        except Exception as e:
            pass
    
    # Try pattern matching
    for pattern in patterns:
        for ext in extensions:
            filepath = Path(ORIGINAL_DIR) / f"{pattern}{ext}"
            if filepath.exists():
                try:
                    if ext == '.csv':
                        df = pd.read_csv(filepath)
                    else:
                        df = pd.read_excel(filepath, engine='openpyxl')
                    return df
                except Exception:
                    continue
    
    return None


def load_result_csv(condition: str, model: str) -> Optional[pd.DataFrame]:
    """Load CSV result file for a specific condition and model."""
    # Exact filename
    exact_filename = f"{condition}_{model}.csv"
    filepath = Path(RESULTS_DIR) / exact_filename
    
    if filepath.exists():
        try:
            df = pd.read_csv(filepath)
            return df
        except Exception as e:
            st.error(f"Error loading {exact_filename}: {str(e)}")
            return None
    
    # Try other patterns
    patterns = [
        f"{condition}_{model}",
        f"{condition}_{model}_results",
    ]
    
    for pattern in patterns:
        filepath = Path(RESULTS_DIR) / f"{pattern}.csv"
        if filepath.exists():
            try:
                df = pd.read_csv(filepath)
                return df
            except Exception:
                continue
    
    return None


def load_both_files(condition: str, model: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load both original and result files."""
    original_df = load_original_file(condition, model)
    result_df = load_result_csv(condition, model)
    return original_df, result_df


def scan_available_files() -> dict:
    """
    Scan directories and return all available files.
    """
    available = {
        'original_data': [],
        'results': [],
    }
    
    # Scan original_data directory
    if Path(ORIGINAL_DIR).exists():
        for file in Path(ORIGINAL_DIR).iterdir():
            if file.is_file():
                available['original_data'].append(file.name)
    
    # Scan results directory
    if Path(RESULTS_DIR).exists():
        for file in Path(RESULTS_DIR).iterdir():
            if file.is_file():
                available['results'].append(file.name)
    
    return available


def get_available_files_summary() -> dict:
    """Get summary of available files across all conditions and models."""
    from config import CONDITIONS_MAP, MODELS
    
    summary = {
        'total_csv': 0,
        'total_original': 0,
        'by_condition': {},
        'by_model': {m: {'csv': 0, 'original': 0} for m in MODELS},
        'available_files': scan_available_files(),
    }
    
    for condition in CONDITIONS_MAP.keys():
        summary['by_condition'][condition] = {'csv': 0, 'original': 0}
        for model in MODELS:
            result = load_result_csv(condition, model)
            original = load_original_file(condition, model)
            
            if result is not None:
                summary['total_csv'] += 1
                summary['by_condition'][condition]['csv'] += 1
                summary['by_model'][model]['csv'] += 1
            
            if original is not None:
                summary['total_original'] += 1
                summary['by_condition'][condition]['original'] += 1
                summary['by_model'][model]['original'] += 1
    
    return summary


# Debug function to print found files
def debug_file_detection():
    """Print debug info about file detection."""
    from config import CONDITIONS_MAP, MODELS
    
    print("=== FILE DETECTION DEBUG ===")
    print(f"Original dir: {Path(ORIGINAL_DIR).absolute()}")
    print(f"Results dir: {Path(RESULTS_DIR).absolute()}")
    
    print("\n--- Checking all conditions/models ---")
    for condition in CONDITIONS_MAP.keys():
        for model in MODELS:
            orig = load_original_file(condition, model)
            res = load_result_csv(condition, model)
            
            orig_status = "✅" if orig is not None else "❌"
            res_status = "✅" if res is not None else "❌"
            
            print(f"{condition}_{model}: Original {orig_status} | Result {res_status}")