"""Analysis functions for original and result datasets."""
import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Set


def analyze_original_by_tool(original_df: pd.DataFrame, expected_tag_count: int = 100) -> Dict:
    """
    Analyze original dataset by splitting according to 'tool' column (RDP vs Santa).
    Returns per-tag and total statistics.
    """
    stats = {
        'total_rows': 0,
        'unique_tags': [],
        'tag_count': 0,
        'expected_tag_count': expected_tag_count,
        'missing_tags': [],
        'per_tag': {},
        'total_rdp_events': 0,
        'total_santa_events': 0,
        'total_events': 0,
        'tags_with_rdp': [],
        'tags_with_santa': [],
        'tags_with_both': [],
        'tags_with_neither': [],
        'zero_rdp_tags': [],
        'zero_santa_tags': [],
    }
    
    if original_df is None or len(original_df) == 0:
        stats['missing_tags'] = [str(i) for i in range(1, expected_tag_count + 1)]
        return stats
    
    stats['total_rows'] = len(original_df)
    
    tag_col = original_df.columns[0]
    
    tool_col = None
    for col in original_df.columns:
        if col.lower() == 'tool':
            tool_col = col
            break
    
    if tool_col is None:
        return stats
    
    def extract_tag_number(value):
        if pd.isna(value):
            return None
        val_str = str(value)
        match = re.match(r'^(\d+)_', val_str)
        if match:
            return int(match.group(1))
        numbers = re.findall(r'\d+', val_str)
        if numbers:
            return int(numbers[0])
        return None
    
    original_df = original_df.copy()
    original_df['tag_number'] = original_df[tag_col].apply(extract_tag_number)
    
    tag_numbers = original_df['tag_number'].dropna().unique().tolist()
    tag_numbers = sorted([int(t) for t in tag_numbers])
    stats['unique_tags'] = tag_numbers
    stats['tag_count'] = len(tag_numbers)
    
    all_expected = set(range(1, expected_tag_count + 1))
    found_tags = set(tag_numbers)
    stats['missing_tags'] = sorted(list(all_expected - found_tags))
    
    for tag_num in tag_numbers:
        tag_data = original_df[original_df['tag_number'] == tag_num]
        
        rdp_data = tag_data[tag_data[tool_col].astype(str).str.upper() == 'RDP']
        santa_data = tag_data[tag_data[tool_col].astype(str).str.upper() == 'SANTA']
        
        rdp_count = len(rdp_data)
        santa_count = len(santa_data)
        
        stats['total_rdp_events'] += rdp_count
        stats['total_santa_events'] += santa_count
        
        has_rdp = rdp_count > 0
        has_santa = santa_count > 0
        
        stats['per_tag'][tag_num] = {
            'tag_number': tag_num,
            'rdp_count': rdp_count,
            'santa_count': santa_count,
            'total': rdp_count + santa_count,
            'has_rdp': has_rdp,
            'has_santa': has_santa,
        }
        
        if has_rdp:
            stats['tags_with_rdp'].append(tag_num)
        else:
            stats['zero_rdp_tags'].append(tag_num)
            
        if has_santa:
            stats['tags_with_santa'].append(tag_num)
        else:
            stats['zero_santa_tags'].append(tag_num)
            
        if has_rdp and has_santa:
            stats['tags_with_both'].append(tag_num)
        if not has_rdp and not has_santa:
            stats['tags_with_neither'].append(tag_num)
    
    stats['total_events'] = stats['total_rdp_events'] + stats['total_santa_events']
    
    return stats


def safe_parse_numeric(value) -> float:
    """Safely parse a value to float, handling strings with symbols."""
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r'[^\d.-]', '', value)
        try:
            return float(cleaned) if cleaned else 0.0
        except ValueError:
            return 0.0
    return 0.0


def parse_rdp_santa_equal(value) -> Dict:
    """Parse RDP_Santa_Equal column value."""
    if pd.isna(value):
        return {'is_equal': False, 'is_diff': False, 'rdp_val': None, 'santa_val': None}
    
    value_str = str(value).strip()
    
    if '✅' in value_str or 'Equal' in value_str:
        return {'is_equal': True, 'is_diff': False, 'rdp_val': None, 'santa_val': None}
    elif '❌' in value_str or 'Diff' in value_str:
        rdp_match = re.search(r'RDP=(\d+)', value_str)
        santa_match = re.search(r'Santa=(\d+)', value_str)
        return {
            'is_equal': False,
            'is_diff': True,
            'rdp_val': int(rdp_match.group(1)) if rdp_match else None,
            'santa_val': int(santa_match.group(1)) if santa_match else None,
        }
    
    return {'is_equal': False, 'is_diff': False, 'rdp_val': None, 'santa_val': None}


def extract_tag_number_from_row(row: pd.Series) -> Optional[int]:
    """Extract tag number from a row in result CSV."""
    tag_columns = ['Tag', 'tag', 'Santa_Tag', 'santa_tag', 'RDP_Tags', 'rdp_tags']
    
    for col in tag_columns:
        if col in row.index:
            val = row[col]
            if not pd.isna(val):
                val_str = str(val)
                numbers = re.findall(r'\d+', val_str)
                if numbers:
                    return int(numbers[0])
    
    if 'SantaID' in row.index:
        val = row['SantaID']
        if not pd.isna(val):
            val_str = str(val)
            numbers = re.findall(r'\d+', val_str)
            if numbers:
                return int(numbers[0])
    
    return None


def get_all_tags_from_result(result_df: pd.DataFrame) -> List[int]:
    """Get all unique tag numbers from result DataFrame."""
    tags = set()
    for _, row in result_df.iterrows():
        tag_num = extract_tag_number_from_row(row)
        if tag_num is not None:
            tags.add(tag_num)
    return sorted(list(tags))


def extract_tag_level_stats(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract per-tag statistics from result dataset.
    Takes UNIQUE values per tag (first row since all rows for same tag have same aggregates).
    """
    if result_df is None or len(result_df) == 0:
        return pd.DataFrame()
    
    all_tags = get_all_tags_from_result(result_df)
    
    if not all_tags:
        return pd.DataFrame()
    
    # Define the CORRECT columns to use
    tag_level_columns = [
        'Tag_RDP_Count',           # Total matched RDP events
        'Tag_Santa_Count',         # Total matched Santa events
        'Tag_Original_RDP_Count',  # Total original RDP events
        'Tag_Original_Santa_Count', # Total original Santa events
        'Tag_Remaining_RDP',
        'Tag_Remaining_Santa',
        'Tag_RDP_Match_Rate',
        'Tag_Santa_Match_Rate',
        'RDP_Santa_Equal',
    ]
    
    # Check which columns exist
    available_cols = [col for col in tag_level_columns if col in result_df.columns]
    
    per_tag_data = []
    
    for tag_num in all_tags:
        # Find all rows for this tag
        tag_rows = []
        for idx, row in result_df.iterrows():
            row_tag = extract_tag_number_from_row(row)
            if row_tag == tag_num:
                tag_rows.append(row)
        
        if not tag_rows:
            continue
        
        # Take first row (all rows for same tag have identical aggregate values)
        first_row = tag_rows[0]
        
        tag_stats = {
            'Tag_Number': tag_num,
            'Row_Count': len(tag_rows)
        }
        
        # Extract the CORRECT values
        if 'Tag_Original_RDP_Count' in result_df.columns:
            tag_stats['Original_RDP'] = safe_parse_numeric(first_row.get('Tag_Original_RDP_Count', 0))
        
        if 'Tag_Original_Santa_Count' in result_df.columns:
            tag_stats['Original_Santa'] = safe_parse_numeric(first_row.get('Tag_Original_Santa_Count', 0))
        
        if 'Tag_RDP_Count' in result_df.columns:
            tag_stats['Result_RDP'] = safe_parse_numeric(first_row.get('Tag_RDP_Count', 0))
        
        if 'Tag_Santa_Count' in result_df.columns:
            tag_stats['Result_Santa'] = safe_parse_numeric(first_row.get('Tag_Santa_Count', 0))
        
        if 'Tag_Remaining_RDP' in result_df.columns:
            tag_stats['Remaining_RDP'] = safe_parse_numeric(first_row.get('Tag_Remaining_RDP', 0))
        
        if 'Tag_Remaining_Santa' in result_df.columns:
            tag_stats['Remaining_Santa'] = safe_parse_numeric(first_row.get('Tag_Remaining_Santa', 0))
        
        # Match rates
        if 'Tag_RDP_Match_Rate' in result_df.columns:
            val = first_row.get('Tag_RDP_Match_Rate')
            if not pd.isna(val):
                val_str = str(val)
                if '%' in val_str:
                    tag_stats['RDP_Match_Rate'] = float(val_str.replace('%', '')) / 100
                else:
                    tag_stats['RDP_Match_Rate'] = safe_parse_numeric(val)
            else:
                tag_stats['RDP_Match_Rate'] = 0.0
        
        if 'Tag_Santa_Match_Rate' in result_df.columns:
            val = first_row.get('Tag_Santa_Match_Rate')
            if not pd.isna(val):
                val_str = str(val)
                if '%' in val_str:
                    tag_stats['Santa_Match_Rate'] = float(val_str.replace('%', '')) / 100
                else:
                    tag_stats['Santa_Match_Rate'] = safe_parse_numeric(val)
            else:
                tag_stats['Santa_Match_Rate'] = 0.0
        
        # RDP_Santa_Equal
        if 'RDP_Santa_Equal' in result_df.columns:
            equal_count = 0
            diff_count = 0
            for row in tag_rows:
                val = row.get('RDP_Santa_Equal')
                parsed = parse_rdp_santa_equal(val)
                if parsed['is_equal']:
                    equal_count += 1
                elif parsed['is_diff']:
                    diff_count += 1
            tag_stats['Equal_Count'] = equal_count
            tag_stats['Diff_Count'] = diff_count
            tag_stats['Is_Equal'] = (equal_count > 0)
        
        # Calculate totals
        tag_stats['Original_Total'] = tag_stats.get('Original_RDP', 0) + tag_stats.get('Original_Santa', 0)
        tag_stats['Result_Total'] = tag_stats.get('Result_RDP', 0) + tag_stats.get('Result_Santa', 0)
        
        per_tag_data.append(tag_stats)
    
    return pd.DataFrame(per_tag_data)


def compute_comparison_stats(original_stats: Dict, result_tag_df: pd.DataFrame, 
                             expected_tag_count: int = 100) -> pd.DataFrame:
    """
    Create a comparison DataFrame merging original per-tag counts with result per-tag stats.
    """
    comparison_data = []
    
    # Create mapping of result tags
    result_tag_map = {}
    if not result_tag_df.empty and 'Tag_Number' in result_tag_df.columns:
        for _, row in result_tag_df.iterrows():
            tag_num = int(row['Tag_Number'])
            result_tag_map[tag_num] = row.to_dict()
    
    orig_per_tag = original_stats.get('per_tag', {})
    
    for tag_num in range(1, expected_tag_count + 1):
        orig_data = orig_per_tag.get(tag_num, {})
        result_data = result_tag_map.get(tag_num)
        
        row = {
            'Tag_Number': tag_num,
            'In_Original': tag_num in orig_per_tag,
            'In_Results': result_data is not None,
        }
        
        # Original data (from original dataset split by tool)
        row['Original_RDP_True'] = orig_data.get('rdp_count', 0)
        row['Original_Santa_True'] = orig_data.get('santa_count', 0)
        row['Original_Total_True'] = row['Original_RDP_True'] + row['Original_Santa_True']
        
        # Result data (from CSV columns)
        if result_data:
            row['Original_RDP_CSV'] = result_data.get('Original_RDP', 0)
            row['Original_Santa_CSV'] = result_data.get('Original_Santa', 0)
            row['Original_Total_CSV'] = result_data.get('Original_Total', 0)
            
            row['Result_RDP'] = result_data.get('Result_RDP', 0)
            row['Result_Santa'] = result_data.get('Result_Santa', 0)
            row['Result_Total'] = result_data.get('Result_Total', 0)
            
            row['Remaining_RDP'] = result_data.get('Remaining_RDP', 0)
            row['Remaining_Santa'] = result_data.get('Remaining_Santa', 0)
            
            row['RDP_Match_Rate'] = result_data.get('RDP_Match_Rate', 0)
            row['Santa_Match_Rate'] = result_data.get('Santa_Match_Rate', 0)
            
            row['Is_Equal'] = result_data.get('Is_Equal', False)
            row['Equal_Count'] = result_data.get('Equal_Count', 0)
            row['Diff_Count'] = result_data.get('Diff_Count', 0)
            row['Row_Count'] = result_data.get('Row_Count', 0)
        else:
            row['Original_RDP_CSV'] = 0
            row['Original_Santa_CSV'] = 0
            row['Original_Total_CSV'] = 0
            row['Result_RDP'] = 0
            row['Result_Santa'] = 0
            row['Result_Total'] = 0
            row['Remaining_RDP'] = 0
            row['Remaining_Santa'] = 0
            row['RDP_Match_Rate'] = 0
            row['Santa_Match_Rate'] = 0
            row['Is_Equal'] = False
            row['Equal_Count'] = 0
            row['Diff_Count'] = 0
            row['Row_Count'] = 0
        
        # Match status
        if not row['In_Original'] and not row['In_Results']:
            row['Match_Status'] = '❌ Missing (Both)'
        elif not row['In_Original']:
            row['Match_Status'] = '⚠️ Missing in Original'
        elif not row['In_Results']:
            row['Match_Status'] = '❌ Missing in Results'
        elif row['Original_RDP_True'] == row['Result_RDP'] and row['Original_Santa_True'] == row['Result_Santa']:
            row['Match_Status'] = '✅ Exact Match'
        else:
            row['Match_Status'] = '❌ Mismatch'
        
        # Zero events
        row['Zero_RDP'] = (row['Original_RDP_True'] == 0) if row['In_Original'] else None
        row['Zero_Santa'] = (row['Original_Santa_True'] == 0) if row['In_Original'] else None
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)


def compute_summary_totals(comparison_df: pd.DataFrame, original_stats: Dict = None) -> Dict:
    """Compute total summary statistics from comparison DataFrame."""
    
    totals = {
        'total_expected_tags': len(comparison_df),
        'tags_in_original': 0,
        'tags_in_results': 0,
        'tags_in_both': 0,
        'missing_in_both': 0,
        'missing_in_results': 0,
        'total_original_rdp': 0,
        'total_original_santa': 0,
        'total_original_events': 0,
        'total_result_rdp': 0,
        'total_result_santa': 0,
        'total_result_events': 0,
        'total_matched_events': 0,
        'total_unmatched_rdp': 0,
        'total_unmatched_santa': 0,
        'false_positive_rate': 0.0,
        'avg_rdp_match_rate': 0.0,
        'avg_santa_match_rate': 0.0,
        'zero_rdp_tags': [],
        'zero_santa_tags': [],
        'exact_matches': 0,
        'equal_count': 0,
    }
    
    if comparison_df.empty:
        return totals
    
    # Count tags in each category
    if 'In_Original' in comparison_df.columns:
        totals['tags_in_original'] = int(comparison_df['In_Original'].sum())
    
    if 'In_Results' in comparison_df.columns:
        totals['tags_in_results'] = int(comparison_df['In_Results'].sum())
    
    totals['tags_in_both'] = int(((comparison_df['In_Original'] == True) & (comparison_df['In_Results'] == True)).sum())
    totals['missing_in_both'] = int(((comparison_df['In_Original'] == False) & (comparison_df['In_Results'] == False)).sum())
    totals['missing_in_results'] = int(((comparison_df['In_Original'] == True) & (comparison_df['In_Results'] == False)).sum())
    
    # ===== USE ORIGINAL STATS FOR TRUE TOTALS (includes ALL tags) =====
    if original_stats:
        totals['total_original_rdp'] = original_stats.get('total_rdp_events', 0)
        totals['total_original_santa'] = original_stats.get('total_santa_events', 0)
        totals['total_original_events'] = original_stats.get('total_events', 0)
        totals['zero_rdp_tags'] = original_stats.get('zero_rdp_tags', [])
        totals['zero_santa_tags'] = original_stats.get('zero_santa_tags', [])
    else:
        # Fallback to summing from comparison_df
        if 'Original_RDP_True' in comparison_df.columns:
            totals['total_original_rdp'] = int(comparison_df['Original_RDP_True'].sum())
        if 'Original_Santa_True' in comparison_df.columns:
            totals['total_original_santa'] = int(comparison_df['Original_Santa_True'].sum())
        totals['total_original_events'] = totals['total_original_rdp'] + totals['total_original_santa']
        
        for _, row in comparison_df.iterrows():
            if row.get('In_Original', False):
                if row.get('Original_RDP_True', 0) == 0:
                    totals['zero_rdp_tags'].append(int(row['Tag_Number']))
                if row.get('Original_Santa_True', 0) == 0:
                    totals['zero_santa_tags'].append(int(row['Tag_Number']))
    
    # Result totals (from CSV - only tags in results)
    if 'Result_RDP' in comparison_df.columns:
        totals['total_result_rdp'] = int(comparison_df['Result_RDP'].sum())
    
    if 'Result_Santa' in comparison_df.columns:
        totals['total_result_santa'] = int(comparison_df['Result_Santa'].sum())
    
    totals['total_result_events'] = totals['total_result_rdp'] + totals['total_result_santa']
    totals['total_matched_events'] = totals['total_result_events']
    
    if 'Remaining_RDP' in comparison_df.columns:
        totals['total_unmatched_rdp'] = int(comparison_df['Remaining_RDP'].sum())
    
    if 'Remaining_Santa' in comparison_df.columns:
        totals['total_unmatched_santa'] = int(comparison_df['Remaining_Santa'].sum())
    
    # False Positive Rate
    if totals['total_result_rdp'] > 0:
        totals['false_positive_rate'] = totals['total_unmatched_rdp'] / totals['total_result_rdp']
    
    if 'Is_Equal' in comparison_df.columns:
        totals['equal_count'] = int(comparison_df['Is_Equal'].sum())
    
    if 'Match_Status' in comparison_df.columns:
        totals['exact_matches'] = int((comparison_df['Match_Status'] == '✅ Exact Match').sum())
    
    # Average match rates
    if 'RDP_Match_Rate' in comparison_df.columns:
        valid_rates = comparison_df[comparison_df['In_Results'] == True]['RDP_Match_Rate'].dropna()
        valid_rates = valid_rates[valid_rates > 0]
        totals['avg_rdp_match_rate'] = valid_rates.mean() if len(valid_rates) > 0 else 0.0
        
    if 'Santa_Match_Rate' in comparison_df.columns:
        valid_rates = comparison_df[comparison_df['In_Results'] == True]['Santa_Match_Rate'].dropna()
        valid_rates = valid_rates[valid_rates > 0]
        totals['avg_santa_match_rate'] = valid_rates.mean() if len(valid_rates) > 0 else 0.0
    
    return totals


def get_missing_tags_list(comparison_df: pd.DataFrame) -> Dict:
    """Get organized lists of missing tags."""
    missing = {
        'missing_in_both': [],
        'missing_in_original': [],
        'missing_in_results': [],
    }
    
    if comparison_df.empty:
        return missing
    
    for _, row in comparison_df.iterrows():
        tag_num = int(row['Tag_Number'])
        in_orig = row.get('In_Original', False)
        in_res = row.get('In_Results', False)
        
        if not in_orig and not in_res:
            missing['missing_in_both'].append(tag_num)
        elif not in_orig:
            missing['missing_in_original'].append(tag_num)
        elif not in_res:
            missing['missing_in_results'].append(tag_num)
    
    return missing

def calculate_circular_distance(pos1: float, pos2: float, genome_length: float = 10000.0) -> float:
    """
    Calculate the circular distance between two positions.
    Accounts for circular genomes where position 1 is adjacent to position genome_length.
    """
    if pd.isna(pos1) or pd.isna(pos2):
        return np.nan
    
    direct_distance = abs(pos1 - pos2)
    circular_distance = genome_length - direct_distance
    return min(direct_distance, circular_distance)


def calculate_breakpoint_distance_min(santa_ini: float, santa_end: float, 
                                       rdp_ini: float, rdp_end: float, 
                                       genome_length: float = 10000.0) -> Dict:
    """
    Calculate breakpoint distances accounting for reversed breakpoints.
    
    Professor's requirement:
    "find the shortest distances of the 50 breakpoint to BOTH 5000 and 10000 
     and select whichever distance is shorter"
    
    For EACH RDP breakpoint, find minimum distance to BOTH Santa breakpoints.
    This properly handles cases where RDP reverses the breakpoint order.
    """
    if pd.isna(santa_ini) or pd.isna(santa_end) or pd.isna(rdp_ini) or pd.isna(rdp_end):
        return {
            'start_distance': np.nan,
            'end_distance': np.nan,
            'total_distance': np.nan,
            'is_reversed': False
        }
    
    # For RDP start breakpoint: find minimum distance to Santa_INI OR Santa_END
    rdp_start_to_santa_ini = calculate_circular_distance(rdp_ini, santa_ini, genome_length)
    rdp_start_to_santa_end = calculate_circular_distance(rdp_ini, santa_end, genome_length)
    start_distance = min(rdp_start_to_santa_ini, rdp_start_to_santa_end)
    
    # For RDP end breakpoint: find minimum distance to Santa_INI OR Santa_END
    rdp_end_to_santa_ini = calculate_circular_distance(rdp_end, santa_ini, genome_length)
    rdp_end_to_santa_end = calculate_circular_distance(rdp_end, santa_end, genome_length)
    end_distance = min(rdp_end_to_santa_ini, rdp_end_to_santa_end)
    
    # A case is "reversed" if RDP start is closer to Santa end than Santa start
    # OR RDP end is closer to Santa start than Santa end
    is_reversed = (rdp_start_to_santa_end < rdp_start_to_santa_ini) or \
                  (rdp_end_to_santa_ini < rdp_end_to_santa_end)
    
    return {
        'start_distance': start_distance,
        'end_distance': end_distance,
        'total_distance': (start_distance + end_distance) / 2,
        'is_reversed': is_reversed
    }


def analyze_breakpoint_distances(result_df: pd.DataFrame, genome_length: float = 10000.0) -> Dict:
    """
    Analyze breakpoint distances between Santa and RDP.
    
    For each matched event:
    - Start distance = min(circular_dist(RDP_ini, Santa_INI), circular_dist(RDP_ini, Santa_END))
    - End distance = min(circular_dist(RDP_end, Santa_INI), circular_dist(RDP_end, Santa_END))
    - Total distance = (start_distance + end_distance) / 2
    
    This correctly handles reversed breakpoints as required.
    """
    distances = {
        'start_distances': [],
        'end_distances': [],
        'total_distances': [],
        'mean_start_distance': 0.0,
        'mean_end_distance': 0.0,
        'mean_total_distance': 0.0,
        'median_start_distance': 0.0,
        'median_end_distance': 0.0,
        'median_total_distance': 0.0,
        'reversed_cases': 0,
        'total_cases': 0,
    }
    
    if result_df is None or len(result_df) == 0:
        return distances
    
    required_cols = ['Santa_INI', 'Santa_END', 'RDP_INI', 'RDP_END']
    if not all(col in result_df.columns for col in required_cols):
        return distances
    
    for _, row in result_df.iterrows():
        try:
            santa_ini = float(row['Santa_INI']) if not pd.isna(row['Santa_INI']) else np.nan
            santa_end = float(row['Santa_END']) if not pd.isna(row['Santa_END']) else np.nan
            rdp_ini = float(row['RDP_INI']) if not pd.isna(row['RDP_INI']) else np.nan
            rdp_end = float(row['RDP_END']) if not pd.isna(row['RDP_END']) else np.nan
            
            if pd.isna(santa_ini) or pd.isna(santa_end) or pd.isna(rdp_ini) or pd.isna(rdp_end):
                continue
            
            distances['total_cases'] += 1
            
            # Calculate distances using minimum to either breakpoint
            dist = calculate_breakpoint_distance_min(santa_ini, santa_end, rdp_ini, rdp_end, genome_length)
            
            distances['start_distances'].append(dist['start_distance'])
            distances['end_distances'].append(dist['end_distance'])
            distances['total_distances'].append(dist['total_distance'])
            
            if dist['is_reversed']:
                distances['reversed_cases'] += 1
                
        except (ValueError, TypeError):
            continue
    
    if distances['start_distances']:
        distances['mean_start_distance'] = np.mean(distances['start_distances'])
        distances['median_start_distance'] = np.median(distances['start_distances'])
    
    if distances['end_distances']:
        distances['mean_end_distance'] = np.mean(distances['end_distances'])
        distances['median_end_distance'] = np.median(distances['end_distances'])
    
    if distances['total_distances']:
        distances['mean_total_distance'] = np.mean(distances['total_distances'])
        distances['median_total_distance'] = np.median(distances['total_distances'])
    
    return distances

def analyze_breakpoint_distances(result_df: pd.DataFrame, genome_length: float = 10000.0) -> Dict:
    """
    Analyze breakpoint distances between Santa and RDP.
    
    For each matched event:
    - Start distance = min(circular_dist(RDP_ini, Santa_INI), circular_dist(RDP_ini, Santa_END))
    - End distance = min(circular_dist(RDP_end, Santa_INI), circular_dist(RDP_end, Santa_END))
    - Total distance = (start_distance + end_distance) / 2
    
    This correctly handles reversed breakpoints as required.
    """
    distances = {
        'start_distances': [],
        'end_distances': [],
        'total_distances': [],
        'mean_start_distance': 0.0,
        'mean_end_distance': 0.0,
        'mean_total_distance': 0.0,
        'median_start_distance': 0.0,
        'median_end_distance': 0.0,
        'median_total_distance': 0.0,
        'reversed_cases': 0,
        'total_cases': 0,
    }
    
    if result_df is None or len(result_df) == 0:
        return distances
    
    required_cols = ['Santa_INI', 'Santa_END', 'RDP_INI', 'RDP_END']
    if not all(col in result_df.columns for col in required_cols):
        return distances
    
    for _, row in result_df.iterrows():
        try:
            santa_ini = float(row['Santa_INI']) if not pd.isna(row['Santa_INI']) else np.nan
            santa_end = float(row['Santa_END']) if not pd.isna(row['Santa_END']) else np.nan
            rdp_ini = float(row['RDP_INI']) if not pd.isna(row['RDP_INI']) else np.nan
            rdp_end = float(row['RDP_END']) if not pd.isna(row['RDP_END']) else np.nan
            
            if pd.isna(santa_ini) or pd.isna(santa_end) or pd.isna(rdp_ini) or pd.isna(rdp_end):
                continue
            
            distances['total_cases'] += 1
            
            # Calculate distances using minimum to either breakpoint
            dist = calculate_breakpoint_distance_min(santa_ini, santa_end, rdp_ini, rdp_end, genome_length)
            
            distances['start_distances'].append(dist['start_distance'])
            distances['end_distances'].append(dist['end_distance'])
            distances['total_distances'].append(dist['total_distance'])
            
            if dist['is_reversed']:
                distances['reversed_cases'] += 1
                
        except (ValueError, TypeError):
            continue
    
    if distances['start_distances']:
        distances['mean_start_distance'] = np.mean(distances['start_distances'])
        distances['median_start_distance'] = np.median(distances['start_distances'])
    
    if distances['end_distances']:
        distances['mean_end_distance'] = np.mean(distances['end_distances'])
        distances['median_end_distance'] = np.median(distances['end_distances'])
    
    if distances['total_distances']:
        distances['mean_total_distance'] = np.mean(distances['total_distances'])
        distances['median_total_distance'] = np.median(distances['total_distances'])
    
    return distances
def analyze_recombinant_accuracy(result_df: pd.DataFrame) -> Dict:
    """
    Analyze recombinant identification accuracy.
    Uses Source_Tab column: "Step 7" = incorrectly identified as parental sequence.
    
    Simple counting: Each Step 7 row = 1 incorrect parental event.
    
    Formula:
    - Incorrect Parental = Count of rows with Source_Tab = "Step 7"
    - Accuracy = 1 - (Incorrect Parental / Total Matched RDP)
    """
    accuracy = {
        'total_matches': 0,
        'total_matched_rdp': 0,
        'correct_recombinant': 0,
        'incorrect_parental': 0,
        'incorrect_parental_rate': 0.0,
        'correct_rate': 0.0,
        'step7_count': 0,
    }
    
    if result_df is None or len(result_df) == 0:
        return accuracy
    
    # Total matched RDP comes from per-tag aggregates
    tag_stats = extract_tag_level_stats(result_df)
    if not tag_stats.empty and 'Result_RDP' in tag_stats.columns:
        accuracy['total_matched_rdp'] = int(tag_stats['Result_RDP'].sum())
    
    accuracy['total_matches'] = len(result_df)
    
    # Count Step 7 rows directly
    if 'Source_Tab' in result_df.columns:
        for _, row in result_df.iterrows():
            source_tab = str(row['Source_Tab']).lower() if not pd.isna(row['Source_Tab']) else ''
            
            if 'step 7' in source_tab or 'step7' in source_tab:
                accuracy['step7_count'] += 1
    
    accuracy['incorrect_parental'] = accuracy['step7_count']
    
    # Formula: Accuracy = 1 - (Step7 / Total Matched RDP)
    if accuracy['total_matched_rdp'] > 0:
        accuracy['incorrect_parental_rate'] = accuracy['incorrect_parental'] / accuracy['total_matched_rdp']
        accuracy['correct_rate'] = 1 - accuracy['incorrect_parental_rate']
        accuracy['correct_recombinant'] = accuracy['total_matched_rdp'] - accuracy['incorrect_parental']
    else:
        accuracy['incorrect_parental_rate'] = 0.0
        accuracy['correct_rate'] = 0.0
        accuracy['correct_recombinant'] = 0
    
    return accuracy


def compute_study_metrics(result_df: pd.DataFrame, original_stats: Dict) -> Dict:
    """
    Compute the three key metrics for the study:
    (1) False Positive Rate = Unmatched RDP / Total Original RDP
    (2) Recombinant Accuracy = 1 - (Step7 rows / Total Matched RDP)
    (3) Breakpoint Distance - circular distance between inferred and simulated breakpoints
    """
    metrics = {
        # (1) False Positive Rate
        'false_positive_rate': 0.0,
        'total_original_rdp': 0,
        'total_rdp_events': 0,
        'unmatched_rdp': 0,
        'matched_rdp': 0,
        
        # (2) Recombinant Accuracy - SIMPLIFIED
        'total_matches': 0,
        'total_matched_rdp': 0,
        'correct_recombinant': 0,
        'incorrect_parental': 0,
        'incorrect_parental_rate': 0.0,
        'step7_count': 0,
        'recombinant_accuracy': 0.0,
        
        # (3) Breakpoint Distance
        'mean_start_distance': 0.0,
        'mean_end_distance': 0.0,
        'mean_breakpoint_distance': 0.0,
        'median_breakpoint_distance': 0.0,
        'total_breakpoints_analyzed': 0,
    }
    
    if result_df is None or len(result_df) == 0:
        return metrics
    
    # Get per-tag statistics
    tag_stats = extract_tag_level_stats(result_df)
    
    # (1) False Positive Rate
    if original_stats:
        metrics['total_original_rdp'] = original_stats.get('total_rdp_events', 0)
    else:
        if not tag_stats.empty and 'Original_RDP' in tag_stats.columns:
            metrics['total_original_rdp'] = int(tag_stats['Original_RDP'].sum())
    
    if not tag_stats.empty:
        if 'Result_RDP' in tag_stats.columns:
            metrics['matched_rdp'] = int(tag_stats['Result_RDP'].sum())
        if 'Remaining_RDP' in tag_stats.columns:
            metrics['unmatched_rdp'] = int(tag_stats['Remaining_RDP'].sum())
    
    metrics['total_rdp_events'] = metrics['matched_rdp'] + metrics['unmatched_rdp']
    
    if metrics['total_original_rdp'] > 0:
        metrics['false_positive_rate'] = metrics['unmatched_rdp'] / metrics['total_original_rdp']
    
    # (2) Recombinant Accuracy - SIMPLE COUNT
    accuracy = analyze_recombinant_accuracy(result_df)
    metrics['total_matches'] = accuracy['total_matches']
    metrics['total_matched_rdp'] = accuracy['total_matched_rdp']
    metrics['correct_recombinant'] = accuracy['correct_recombinant']
    metrics['incorrect_parental'] = accuracy['incorrect_parental']
    metrics['incorrect_parental_rate'] = accuracy['incorrect_parental_rate']
    metrics['step7_count'] = accuracy['step7_count']
    metrics['recombinant_accuracy'] = accuracy['correct_rate']
    
    # (3) Breakpoint Distance
    bp_distances = analyze_breakpoint_distances(result_df)
    metrics['mean_start_distance'] = bp_distances['mean_start_distance']
    metrics['mean_end_distance'] = bp_distances['mean_end_distance']
    metrics['mean_breakpoint_distance'] = bp_distances['mean_total_distance']
    metrics['median_breakpoint_distance'] = bp_distances['median_start_distance']
    metrics['total_breakpoints_analyzed'] = len(bp_distances['total_distances'])
    
    return metrics


def compute_study_summary_all_conditions() -> pd.DataFrame:
    """
    Compute study metrics for all 9 conditions and 3 models.
    Returns a DataFrame with summary for each condition.
    """
    from config import CONDITIONS_MAP, MODELS
    from data_loader import load_both_files
    
    summary_data = []
    
    for condition in CONDITIONS_MAP.keys():
        for model in MODELS:
            original_df, result_df = load_both_files(condition, model)
            
            if result_df is not None:
                orig_stats = analyze_original_by_tool(original_df) if original_df is not None else {}
                metrics = compute_study_metrics(result_df, orig_stats)
                
                summary_data.append({
                    'Condition': condition,
                    'Model': model,
                    'μ': f"{CONDITIONS_MAP[condition]['mutation_rate']:.2e}",
                    'r': CONDITIONS_MAP[condition]['recomb_rate'],
                    
                    # (1) False Positive Rate
                    'Orig_RDP': metrics['total_original_rdp'],
                    'Matched_RDP': metrics['matched_rdp'],
                    'Unmatched_RDP': metrics['unmatched_rdp'],
                    'FPR': f"{metrics['false_positive_rate']:.3%}",
                    
                    # (2) Recombinant Accuracy
                    'Total_Matches': metrics['total_matches'],
                    'Correct_Recombinant': metrics['correct_recombinant'],
                    'Incorrect_Parental': metrics['incorrect_parental'],
                    'Step7_Count': metrics['step7_count'],
                    'Accuracy': f"{metrics['recombinant_accuracy']:.3%}",
                    
                    # (3) Breakpoint Distance
                    'Mean_BP_Dist': f"{metrics['mean_breakpoint_distance']:.1f}" if metrics['mean_breakpoint_distance'] > 0 else 'N/A',
                    'BP_Analyzed': metrics['total_breakpoints_analyzed'],
                })
    
    return pd.DataFrame(summary_data)









    