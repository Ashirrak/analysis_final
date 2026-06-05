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
    Counts RDP events by splitting comma-separated RDP_Event_ID per row.
    Each occurrence counts (even if same RDP ID appears in multiple rows).
    """
    if result_df is None or len(result_df) == 0:
        return pd.DataFrame()
    
    all_tags = get_all_tags_from_result(result_df)
    
    if not all_tags:
        return pd.DataFrame()
    
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
        
        first_row = tag_rows[0]
        
        tag_stats = {'Tag_Number': tag_num, 'Row_Count': len(tag_rows)}
        
        # ====================================================================
        # COUNT RDP EVENTS: Split each row's RDP_Event_ID by comma
        # "1,2,13" = 3 events. Count EVERY occurrence (no dedup).
        # ====================================================================
        total_rdp_count = 0
        rdp_detail_list = []  # For debugging
        
        for row in tag_rows:
            rdp_event_id = str(row.get('RDP_Event_ID', ''))
            if rdp_event_id and rdp_event_id != 'nan':
                parts = [p.strip().rstrip('.0') for p in rdp_event_id.replace(',', ' ').split() 
                        if p.strip().rstrip('.0').isdigit()]
                total_rdp_count += len(parts)
                rdp_detail_list.extend(parts)
        
        tag_stats['Result_RDP'] = total_rdp_count
        # Store for debugging
        tag_stats['_rdp_detail'] = rdp_detail_list
        tag_stats['_rdp_unique_count'] = len(set(rdp_detail_list))
        
        # ====================================================================
        # COUNT SANTA EVENTS (unique Santa_ID)
        # ====================================================================
        santa_count = 0
        for row in tag_rows:
            santa_id = str(row.get('Santa_ID', ''))
            if santa_id and santa_id != 'nan':
                santa_count += 1
        
        tag_stats['Result_Santa'] = santa_count
        
        # Original counts
        if 'Tag_Original_RDP_Count' in result_df.columns:
            tag_stats['Original_RDP'] = safe_parse_numeric(first_row.get('Tag_Original_RDP_Count', 0))
        if 'Tag_Original_Santa_Count' in result_df.columns:
            tag_stats['Original_Santa'] = safe_parse_numeric(first_row.get('Tag_Original_Santa_Count', 0))
        
        # Remaining
        tag_stats['Remaining_RDP'] = max(0, tag_stats.get('Original_RDP', 0) - total_rdp_count)
        tag_stats['Remaining_Santa'] = max(0, tag_stats.get('Original_Santa', 0) - santa_count)
        
        # Match rates
        orig_rdp = tag_stats.get('Original_RDP', 0)
        tag_stats['RDP_Match_Rate'] = (total_rdp_count / orig_rdp) if orig_rdp > 0 else 0.0
        
        orig_santa = tag_stats.get('Original_Santa', 0)
        tag_stats['Santa_Match_Rate'] = (santa_count / orig_santa) if orig_santa > 0 else 0.0
        
        # Totals
        tag_stats['Original_Total'] = tag_stats.get('Original_RDP', 0) + tag_stats.get('Original_Santa', 0)
        tag_stats['Result_Total'] = total_rdp_count + santa_count
        
        per_tag_data.append(tag_stats)
    
    result_df_out = pd.DataFrame(per_tag_data)
    
    # ====================================================================
    # DEBUG: Print summary to help identify missing RDPs
    # ====================================================================
    total_result_rdp = result_df_out['Result_RDP'].sum() if not result_df_out.empty else 0
    total_unique_rdp = sum(len(set(row['_rdp_detail'])) for _, row in result_df_out.iterrows()) if not result_df_out.empty else 0
    
    # Remove debug columns for final output
    if '_rdp_detail' in result_df_out.columns:
        result_df_out = result_df_out.drop(columns=['_rdp_detail', '_rdp_unique_count'], errors='ignore')
    
    return result_df_out
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
        'rdp_match_rate_eligible': 0.0,
        'total_rdp_in_eligible_tags': 0,
        'total_matched_rdp_in_eligible_tags': 0,
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
    totals['missing_in_results'] = int(((comparison_df['In_Original'] == True) & (comparison_df['In_Results'] == False)).sum())
    
    # ===== USE ORIGINAL STATS FOR TRUE TOTALS =====
    if original_stats:
        totals['total_original_rdp'] = original_stats.get('total_rdp_events', 0)
        totals['total_original_santa'] = original_stats.get('total_santa_events', 0)
        totals['total_original_events'] = original_stats.get('total_events', 0)
        totals['zero_rdp_tags'] = original_stats.get('zero_rdp_tags', [])
        totals['zero_santa_tags'] = original_stats.get('zero_santa_tags', [])
        
        # ===== RDP+ SANTA+ TAGS =====
        tags_with_both = original_stats.get('tags_with_both', [])
        
        # ===== CRITICAL FIX: Count RDP events in eligible tags =====
        # This should count ALL RDP rows for tags that have both RDP and SANTA
        total_rdp_eligible = 0
        for tag_num in tags_with_both:
            tag_data = original_stats.get('per_tag', {}).get(tag_num, {})
            total_rdp_eligible += tag_data.get('rdp_count', 0)
        totals['total_rdp_in_eligible_tags'] = total_rdp_eligible
        
        # ===== Count matched RDP in those same eligible tags =====
        matched_rdp_eligible = 0
        for _, row in comparison_df.iterrows():
            if row.get('In_Results', False):
                tag_num = int(row['Tag_Number'])
                if tag_num in tags_with_both:
                    # Use Result_RDP which already counts chains properly (split by comma)
                    matched_rdp_eligible += row.get('Result_RDP', 0)
        totals['total_matched_rdp_in_eligible_tags'] = matched_rdp_eligible
        
        # RDP Match Rate for eligible tags
        if total_rdp_eligible > 0:
            totals['rdp_match_rate_eligible'] = (matched_rdp_eligible / total_rdp_eligible) * 100
    else:
        if 'Original_RDP_True' in comparison_df.columns:
            totals['total_original_rdp'] = int(comparison_df['Original_RDP_True'].sum())
        if 'Original_Santa_True' in comparison_df.columns:
            totals['total_original_santa'] = int(comparison_df['Original_Santa_True'].sum())
        totals['total_original_events'] = totals['total_original_rdp'] + totals['total_original_santa']
    
    # Result totals from comparison_df (already properly counted in extract_tag_level_stats)
    if 'Result_RDP' in comparison_df.columns:
        totals['total_result_rdp'] = int(comparison_df['Result_RDP'].sum())
    
    if 'Result_Santa' in comparison_df.columns:
        totals['total_result_santa'] = int(comparison_df['Result_Santa'].sum())
    
    totals['total_result_events'] = totals['total_result_rdp'] + totals['total_result_santa']
    
    if 'Remaining_RDP' in comparison_df.columns:
        totals['total_unmatched_rdp'] = int(comparison_df['Remaining_RDP'].sum())
    
    if 'Remaining_Santa' in comparison_df.columns:
        totals['total_unmatched_santa'] = int(comparison_df['Remaining_Santa'].sum())
    
    # False Positive Rate
    if totals['total_result_rdp'] > 0:
        totals['false_positive_rate'] = (totals['total_unmatched_rdp'] / totals['total_result_rdp']) * 100
    
    if 'Is_Equal' in comparison_df.columns:
        totals['equal_count'] = int(comparison_df['Is_Equal'].sum())
    
    if 'Match_Status' in comparison_df.columns:
        totals['exact_matches'] = int((comparison_df['Match_Status'] == '✅ Exact Match').sum())
    
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


"""
Statistical Analysis Module for Per-Replicate Comparisons
=========================================================
Performs paired hypothesis tests, confidence intervals,
and effect size calculations across classifiers and conditions.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def compute_per_replicate_metrics(
    result_df: pd.DataFrame, 
    original_stats: Dict,
    replicate_ids: List[int]
    ) -> pd.DataFrame:
    """
    Extract per-replicate metrics from the result DataFrame.
    
    Each row in result_df has a Santa_Tag or Tag column containing the replicate ID.
    Groups by replicate and computes:
    - Accuracy (correct recombinant identification)
    - FPR (false positive rate)
    - BP Distance (mean breakpoint distance)
    
    Args:
        result_df: Results DataFrame with Tag/Replicate column
        original_stats: Original dataset statistics per tag
        replicate_ids: List of expected replicate IDs
        
    Returns:
        DataFrame with columns: Replicate, Accuracy, FPR, BP_Distance, N_Matches
    """
    from analysis import (
        compute_study_metrics, 
        analyze_original_by_tool,
        extract_tag_number_from_row,
        analyze_breakpoint_distances,
        analyze_recombinant_accuracy
    )
    
    per_replicate_data = []
    
    if result_df is None or len(result_df) == 0:
        return pd.DataFrame()
    
    # Find tag/replicate column
    tag_col = None
    for col in ['Tag', 'tag', 'Santa_Tag', 'Replicate', 'replicate']:
        if col in result_df.columns:
            tag_col = col
            break
    
    if tag_col is None:
        return pd.DataFrame()
    
    # Extract tag numbers
    def extract_tag(val):
        if pd.isna(val):
            return None
        import re
        numbers = re.findall(r'\d+', str(val))
        return int(numbers[0]) if numbers else None
    
    result_df_copy = result_df.copy()
    result_df_copy['_tag_num'] = result_df_copy[tag_col].apply(extract_tag)
    
    # Group by tag number
    for tag_num in replicate_ids:
        tag_rows = result_df_copy[result_df_copy['_tag_num'] == tag_num]
        
        if len(tag_rows) == 0:
            # No results for this replicate
            per_replicate_data.append({
                'Replicate': tag_num,
                'Accuracy': np.nan,
                'FPR': np.nan,
                'BP_Distance': np.nan,
                'N_Matches': 0,
                'N_Incorrect': 0,
            })
            continue
        
        # Get original counts for this tag
        orig_tag_data = original_stats.get('per_tag', {}).get(tag_num, {})
        orig_rdp = orig_tag_data.get('rdp_count', 0)
        
        # Count matched RDP events (split comma-separated IDs)
        total_matched = 0
        for _, row in tag_rows.iterrows():
            rdp_id = str(row.get('RDP_Event_ID', ''))
            if rdp_id and rdp_id != 'nan':
                parts = [p.strip().rstrip('.0') for p in rdp_id.replace(',', ' ').split()
                        if p.strip().rstrip('.0').isdigit()]
                total_matched += len(parts)
        
        # Count incorrect parental (Step 7)
        incorrect = 0
        if 'Source_Tab' in tag_rows.columns:
            for _, row in tag_rows.iterrows():
                source = str(row['Source_Tab']).lower() if not pd.isna(row['Source_Tab']) else ''
                if 'step 7' in source or 'step7' in source:
                    incorrect += 1
        
        # Accuracy
        accuracy = (total_matched - incorrect) / total_matched if total_matched > 0 else np.nan
        
        # FPR (false positive = unmatched / original)
        fpr = (orig_rdp - total_matched) / orig_rdp if orig_rdp > 0 else np.nan
        
        # Breakpoint distance (for this replicate's rows)
        bp_dists = analyze_breakpoint_distances(tag_rows)
        mean_bp = bp_dists['mean_total_distance'] if bp_dists['total_distances'] else np.nan
        
        per_replicate_data.append({
            'Replicate': tag_num,
            'Accuracy': accuracy,
            'FPR': fpr,
            'BP_Distance': mean_bp,
            'N_Matches': total_matched,
            'N_Incorrect': incorrect,
        })
    
    return pd.DataFrame(per_replicate_data)


def paired_comparison_test(
    data1: np.ndarray, 
    data2: np.ndarray,
    test_type: str = 'auto',
    alpha: float = 0.05
    ) -> Dict:
    """
    Perform paired comparison between two classifiers.
    
    Args:
        data1, data2: Paired metric values (same replicates)
        test_type: 'auto' (chooses based on normality), 'ttest', or 'wilcoxon'
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    # Remove NaN pairs
    mask = ~(np.isnan(data1) | np.isnan(data2))
    d1 = data1[mask]
    d2 = data2[mask]
    
    if len(d1) < 3:
        return {
            'test': 'insufficient_data',
            'n_pairs': len(d1),
            'statistic': np.nan,
            'p_value': np.nan,
            'significant': False,
            'mean_diff': np.nan,
            'ci_95': (np.nan, np.nan),
        }
    
    mean_diff = np.mean(d1 - d2)
    
    if test_type == 'auto':
        # Test normality of differences
        _, p_norm = stats.shapiro(d1 - d2)
        test_type = 'ttest' if p_norm > 0.05 else 'wilcoxon'
    
    if test_type == 'ttest':
        statistic, p_value = stats.ttest_rel(d1, d2)
        test_name = 'paired_t_test'
        # Confidence interval for mean difference
        se = stats.sem(d1 - d2)
        ci = stats.t.interval(0.95, len(d1)-1, loc=mean_diff, scale=se)
    else:
        statistic, p_value = stats.wilcoxon(d1, d2, alternative='two-sided')
        test_name = 'wilcoxon_signed_rank'
        # Bootstrap CI for median difference
        diffs = d1 - d2
        boot_diffs = np.random.choice(diffs, size=(10000, len(diffs)), replace=True)
        boot_medians = np.median(boot_diffs, axis=1)
        ci = (np.percentile(boot_medians, 2.5), np.percentile(boot_medians, 97.5))
    
    # Effect size (Cohen's d for paired)
    d_pooled = np.mean(d1 - d2) / np.std(d1 - d2, ddof=1) if np.std(d1 - d2) > 0 else 0
    
    return {
        'test': test_name,
        'n_pairs': len(d1),
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < alpha,
        'mean_diff': mean_diff,
        'ci_95': ci,
        'cohens_d': d_pooled,
        'alpha': alpha,
    }


def compute_all_pairwise_tests(
    metrics_df: pd.DataFrame,
    metric_col: str = 'Accuracy',
    alpha: float = 0.05
    ) -> pd.DataFrame:
    """
    Compute all pairwise comparisons between classifiers.
    
    Args:
        metrics_df: DataFrame with columns: Replicate, Model, Accuracy, FPR, BP_Distance
        metric_col: Which metric to compare
        alpha: Significance level
        
    Returns:
        DataFrame with pairwise comparison results
    """
    models = metrics_df['Model'].unique()
    results = []
    
    for i, m1 in enumerate(models):
        for m2 in models[i+1:]:
            d1 = metrics_df[metrics_df['Model'] == m1][metric_col].values
            d2 = metrics_df[metrics_df['Model'] == m2][metric_col].values
            
            # Ensure same replicates
            rep1 = metrics_df[metrics_df['Model'] == m1]['Replicate'].values
            rep2 = metrics_df[metrics_df['Model'] == m2]['Replicate'].values
            common_reps = np.intersect1d(rep1, rep2)
            
            if len(common_reps) < 3:
                continue
            
            mask1 = np.isin(rep1, common_reps)
            mask2 = np.isin(rep2, common_reps)
            
            test_result = paired_comparison_test(
                d1[mask1], d2[mask2], 
                test_type='auto', 
                alpha=alpha
            )
            
            # Bonferroni correction
            n_comparisons = len(models) * (len(models) - 1) / 2
            bonferroni_alpha = alpha / n_comparisons
            
            results.append({
                'Comparison': f'{m1} vs {m2}',
                'Model_1': m1,
                'Model_2': m2,
                'Metric': metric_col,
                'Test': test_result['test'],
                'N_Pairs': test_result['n_pairs'],
                'Mean_Diff': test_result['mean_diff'],
                'CI_95_Lower': test_result['ci_95'][0],
                'CI_95_Upper': test_result['ci_95'][1],
                'Statistic': test_result['statistic'],
                'P_Value': test_result['p_value'],
                'Significant (α=0.05)': test_result['significant'],
                'Significant (Bonferroni)': test_result['p_value'] < bonferroni_alpha,
                "Cohen's d": test_result['cohens_d'],
            })
    
    return pd.DataFrame(results)


def compute_summary_with_ci(
    per_replicate_df: pd.DataFrame,
    metric_col: str = 'Accuracy'
    ) -> Dict:
    """
    Compute summary statistics with confidence intervals.
    
    Args:
        per_replicate_df: DataFrame with per-replicate metrics
        metric_col: Metric column name
        
    Returns:
        Dictionary with mean, std, CI, median, IQR
    """
    values = per_replicate_df[metric_col].dropna().values
    
    if len(values) < 3:
        return {
            'mean': np.nan, 'std': np.nan, 'sem': np.nan,
            'ci_95_lower': np.nan, 'ci_95_upper': np.nan,
            'median': np.nan, 'q1': np.nan, 'q3': np.nan,
            'n': len(values),
        }
    
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    sem = stats.sem(values)
    ci = stats.t.interval(0.95, len(values)-1, loc=mean, scale=sem)
    median = np.median(values)
    q1, q3 = np.percentile(values, [25, 75])
    
    return {
        'mean': mean,
        'std': std,
        'sem': sem,
        'ci_95_lower': ci[0],
        'ci_95_upper': ci[1],
        'median': median,
        'q1': q1,
        'q3': q3,
        'iqr': q3 - q1,
        'n': len(values),
    }


def friedman_test_across_conditions(
    metrics_df: pd.DataFrame,
    metric_col: str = 'Accuracy'
    ) -> Dict:
    """
    Friedman test (non-parametric repeated measures ANOVA).
    Tests if classifiers differ significantly across conditions.
    
    Args:
        metrics_df: DataFrame with Model, Condition, and metric
        metric_col: Metric to test
        
    Returns:
        Dictionary with test results
    """
    from scipy.stats import friedmanchisquare
    
    models = sorted(metrics_df['Model'].unique())
    
    if len(models) < 2:
        return {'test': 'insufficient_models', 'n_models': len(models)}
    
    # Pivot to get models as columns
    pivot = metrics_df.pivot_table(
        index=['Condition', 'Replicate'],
        columns='Model',
        values=metric_col
    ).dropna()
    
    if len(pivot) < 3:
        return {'test': 'insufficient_data', 'n_blocks': len(pivot)}
    
    # Extract columns for each model
    groups = [pivot[m].values for m in models]
    
    try:
        statistic, p_value = friedmanchisquare(*groups)
        
        # Effect size (Kendall's W)
        n = len(pivot)
        k = len(models)
        W = statistic / (n * (k - 1)) if n > 0 and k > 1 else 0
        
        return {
            'test': 'friedman',
            'n_blocks': n,
            'n_models': k,
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'kendall_w': W,
            'interpretation': f'{"Significant" if p_value < 0.05 else "No significant"} difference among classifiers'
        }
    except Exception as e:
        return {'test': 'error', 'error': str(e)}


def compute_all_statistics(
    all_per_replicate_data: Dict,
    conditions: List[str],
    models: List[str],
    metrics: List[str] = ['Accuracy', 'FPR', 'BP_Distance']
    ) -> Dict:
    """
    Master function: compute all statistics for all conditions and models.
    
    Args:
        all_per_replicate_data: Dict[condition][model] = per_replicate DataFrame
        conditions: List of condition names
        models: List of model names
        metrics: List of metric column names
        
    Returns:
        Dictionary with all statistical results
    """
    results = {
        'summaries': {},        # Per condition, per model summaries
        'pairwise_tests': {},   # Per condition, per metric pairwise tests
        'friedman_tests': {},   # Per condition, per metric Friedman tests
        'global_pairwise': {},  # Across all conditions
    }
    
    for condition in conditions:
        results['summaries'][condition] = {}
        results['pairwise_tests'][condition] = {}
        results['friedman_tests'][condition] = {}
        
        # Combine all models for this condition
        cond_data = []
        for model in models:
            df = all_per_replicate_data.get(condition, {}).get(model)
            if df is not None and not df.empty:
                df_copy = df.copy()
                df_copy['Model'] = model
                df_copy['Condition'] = condition
                cond_data.append(df_copy)
                
                # Summary per model
                results['summaries'][condition][model] = {}
                for metric in metrics:
                    if metric in df_copy.columns:
                        results['summaries'][condition][model][metric] = \
                            compute_summary_with_ci(df_copy, metric)
        
        if len(cond_data) >= 2:
            combined = pd.concat(cond_data, ignore_index=True)
            
            # Pairwise tests per metric
            for metric in metrics:
                if metric in combined.columns:
                    results['pairwise_tests'][condition][metric] = \
                        compute_all_pairwise_tests(combined, metric)
                    
                    # Friedman test
                    results['friedman_tests'][condition][metric] = \
                        friedman_test_across_conditions(combined, metric)
    
    # Global pairwise (across all conditions)
    all_combined = []
    for condition in conditions:
        for model in models:
            df = all_per_replicate_data.get(condition, {}).get(model)
            if df is not None and not df.empty:
                df_copy = df.copy()
                df_copy['Model'] = model
                df_copy['Condition'] = condition
                all_combined.append(df_copy)
    
    if all_combined:
        global_df = pd.concat(all_combined, ignore_index=True)
        results['global_pairwise'] = {}
        for metric in metrics:
            if metric in global_df.columns:
                results['global_pairwise'][metric] = \
                    compute_all_pairwise_tests(global_df, metric)
    
    return results


def format_p_value(p: float, alpha: float = 0.05) -> str:
    """Format p-value with significance stars."""
    if pd.isna(p):
        return 'N/A'
    
    if p < 0.001:
        stars = '***'
    elif p < 0.01:
        stars = '**'
    elif p < 0.05:
        stars = '*'
    else:
        stars = 'ns'
    
    return f'{p:.4f} {stars}'


def create_results_table_apa_style(
    summaries: Dict,
    pairwise_tests: Dict,
    metric: str = 'Accuracy',
    condition: str = None
    ) -> pd.DataFrame:
    """
    Create an APA-style results table.
    
    Args:
        summaries: Summary statistics
        pairwise_tests: Pairwise test results
        metric: Metric name
        condition: Specific condition (or None for global)
        
    Returns:
        Formatted DataFrame
    """
    rows = []
    
    if condition:
        cond_summaries = summaries.get(condition, {})
    else:
        # Aggregate across conditions
        cond_summaries = {}
        # This would require aggregation logic
    
    for model, model_stats in cond_summaries.items():
        metric_stats = model_stats.get(metric, {})
        
        rows.append({
            'Model': model,
            'M': f"{metric_stats.get('mean', 0):.4f}",
            'SD': f"{metric_stats.get('std', 0):.4f}",
            '95% CI': f"[{metric_stats.get('ci_95_lower', 0):.4f}, {metric_stats.get('ci_95_upper', 0):.4f}]",
            'Mdn': f"{metric_stats.get('median', 0):.4f}",
            'IQR': f"[{metric_stats.get('q1', 0):.4f}, {metric_stats.get('q3', 0):.4f}]",
            'N': metric_stats.get('n', 0),
        })
    
    return pd.DataFrame(rows)




"""
Additional Analysis Functions
==============================
Correlation analysis, boxplot data preparation, and condition-specific comparisons.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional


def compute_metric_correlations(per_replicate_df: pd.DataFrame) -> Dict:
    """
    Compute correlations between the three metrics (Accuracy, FPR, BP Distance).
    
    Args:
        per_replicate_df: DataFrame with columns Accuracy, FPR, BP_Distance
        
    Returns:
        Dict with correlation matrices (Pearson, Spearman) and p-values
    """
    metrics = ['Accuracy', 'FPR', 'BP_Distance']
    available = [m for m in metrics if m in per_replicate_df.columns]
    
    if len(available) < 2:
        return {'error': 'Need at least 2 metrics'}
    
    data = per_replicate_df[available].dropna()
    
    results = {
        'pearson': {'matrix': None, 'p_values': None},
        'spearman': {'matrix': None, 'p_values': None},
        'n_samples': len(data),
    }
    
    if len(data) >= 3:
        # Pearson correlation
        pearson_r = np.zeros((len(available), len(available)))
        pearson_p = np.zeros((len(available), len(available)))
        
        for i, m1 in enumerate(available):
            for j, m2 in enumerate(available):
                if i == j:
                    pearson_r[i, j] = 1.0
                    pearson_p[i, j] = 0.0
                else:
                    r, p = stats.pearsonr(data[m1], data[m2])
                    pearson_r[i, j] = r
                    pearson_p[i, j] = p
        
        results['pearson']['matrix'] = pd.DataFrame(pearson_r, index=available, columns=available)
        results['pearson']['p_values'] = pd.DataFrame(pearson_p, index=available, columns=available)
        
        # Spearman correlation
        spearman_r = np.zeros((len(available), len(available)))
        spearman_p = np.zeros((len(available), len(available)))
        
        for i, m1 in enumerate(available):
            for j, m2 in enumerate(available):
                if i == j:
                    spearman_r[i, j] = 1.0
                    spearman_p[i, j] = 0.0
                else:
                    r, p = stats.spearmanr(data[m1], data[m2])
                    spearman_r[i, j] = r
                    spearman_p[i, j] = p
        
        results['spearman']['matrix'] = pd.DataFrame(spearman_r, index=available, columns=available)
        results['spearman']['p_values'] = pd.DataFrame(spearman_p, index=available, columns=available)
    
    return results


def compute_correlation_with_biological_params(
    summary_df: pd.DataFrame
) -> Dict:
    """
    Compute correlations between model performance and biological parameters (μ, r).
    
    Args:
        summary_df: DataFrame with columns: Condition, Model, μ, r, Accuracy, FPR, BP_Distance
        
    Returns:
        Dict with correlation results per model
    """
    results = {}
    
    for model in summary_df['Model'].unique():
        model_data = summary_df[summary_df['Model'] == model]
        
        # Initialize correlations dictionary for this model
        correlations = {}
        
        for metric in ['Accuracy', 'FPR', 'BP_Distance']:
            # Correlation with μ
            if 'μ' in model_data.columns:
                mu_values = model_data['μ'].values
                metric_values = model_data[metric].values
                
                # Remove NaN values
                valid_mask = ~np.isnan(mu_values) & ~np.isnan(metric_values)
                mu_clean = mu_values[valid_mask]
                metric_clean = metric_values[valid_mask]
                
                if len(mu_clean) > 1:  # Need at least 2 points for correlation
                    r_mu, p_mu = stats.spearmanr(mu_clean, metric_clean)
                    correlations[f'{metric}_vs_μ'] = {
                        'spearman_r': r_mu,
                        'p_value': p_mu,
                        'significant': p_mu < 0.05,
                    }
            
            # Correlation with r (recombination rate)
            if 'r' in model_data.columns:
                r_values = model_data['r'].values
                metric_values = model_data[metric].values
                
                # Remove NaN values
                valid_mask = ~np.isnan(r_values) & ~np.isnan(metric_values)
                r_clean = r_values[valid_mask]
                metric_clean = metric_values[valid_mask]
                
                if len(r_clean) > 1:  # Need at least 2 points for correlation
                    r_r, p_r = stats.spearmanr(r_clean, metric_clean)
                    correlations[f'{metric}_vs_r'] = {
                        'spearman_r': r_r,
                        'p_value': p_r,
                        'significant': p_r < 0.05,
                    }
        
        results[model] = correlations
    
    return results

def prepare_correlation_heatmap_data(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a correlation matrix for all metrics × biological parameters.
    """
    metrics = ['Accuracy', 'FPR', 'BP_Distance']
    bio_params = ['μ', 'r']
    
    all_cols = [c for c in metrics + bio_params if c in summary_df.columns]
    corr_data = summary_df[all_cols].dropna()
    
    if len(corr_data) < 3:
        return pd.DataFrame()
    
    # Compute Spearman correlation matrix
    corr_matrix = corr_data.corr(method='spearman')
    
    return corr_matrix


def compute_condition_ranking(
    per_replicate_data: Dict,
    conditions: List[str],
    models: List[str],
    metric: str = 'Accuracy'
    ) -> pd.DataFrame:
    """
    Rank models within each condition based on mean performance.
    Also computes pairwise differences and whether they are significant.
    """
    from analysis import compute_summary_with_ci
    
    ranking_data = []
    
    for condition in conditions:
        cond_results = []
        for model in models:
            df = per_replicate_data.get(condition, {}).get(model)
            if df is not None and not df.empty and metric in df.columns:
                summary = compute_summary_with_ci(df, metric)
                cond_results.append({
                    'Condition': condition,
                    'Model': model,
                    'Mean': summary['mean'],
                    'SD': summary['std'],
                    'CI_Lower': summary['ci_95_lower'],
                    'CI_Upper': summary['ci_95_upper'],
                    'Median': summary['median'],
                })
        
        if cond_results:
            # Rank by mean (higher = better for Accuracy, lower = better for FPR)
            reverse = metric != 'FPR'
            sorted_results = sorted(cond_results, key=lambda x: x['Mean'], reverse=reverse)
            
            for rank, res in enumerate(sorted_results, 1):
                res['Rank'] = rank
                ranking_data.append(res)
    
    return pd.DataFrame(ranking_data)


def compute_model_advantage_matrix(
    per_replicate_data: Dict,
    conditions: List[str],
    models: List[str],
    metric: str = 'Accuracy'
) -> pd.DataFrame:
    """
    Compute pairwise advantage: how much better is model A over model B
    in each condition. Positive = A better, Negative = B better.
    """
    advantages = []
    
    for condition in conditions:
        model_means = {}
        for model in models:
            df = per_replicate_data.get(condition, {}).get(model)
            if df is not None and not df.empty and metric in df.columns:
                model_means[model] = df[metric].mean()
        
        for m1 in models:
            for m2 in models:
                if m1 != m2 and m1 in model_means and m2 in model_means:
                    diff = model_means[m1] - model_means[m2]
                    # For FPR, reverse the sign (lower is better)
                    if metric == 'FPR':
                        diff = -diff
                    
                    advantages.append({
                        'Condition': condition,
                        'Comparison': f'{m1} vs {m2}',
                        'Model_Better': m1 if diff > 0 else m2,
                        'Advantage': abs(diff),
                        'Raw_Difference': diff,
                    })
    
    return pd.DataFrame(advantages)

    