import sys
import pandas as pd
import os
from datetime import datetime

def read_file(file_path):
    """
    Read CSV or Excel file and return dataframe
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.csv':
        return pd.read_csv(file_path)
    elif file_ext in ['.xls', '.xlsx']:
        return pd.read_excel(file_path)
    else:
        print(f"Warning: Unsupported file format '{file_ext}' for {file_path}")
        return None

def concatenate_files(input_files, output_file=None):
    """
    Concatenate multiple CSV/Excel files into one output file
    """
    if len(input_files) < 2:
        print("Please provide at least 2 files to concatenate")
        return
    
    # Check if all input files exist
    for file in input_files:
        if not os.path.exists(file):
            print(f"Error: File '{file}' not found!")
            return
    
    # If no output file specified, create one with timestamp
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Check first file type to determine output format
        first_ext = os.path.splitext(input_files[0])[1].lower()
        if first_ext in ['.xls', '.xlsx']:
            output_file = f"concatenated_{timestamp}.xlsx"
        else:
            output_file = f"concatenated_{timestamp}.csv"
    
    try:
        # Read and concatenate all files
        dfs = []
        for file in input_files:
            df = read_file(file)
            if df is not None:
                dfs.append(df)
                print(f"Read {len(df)} rows from {file}")
        
        if not dfs:
            print("No valid files to concatenate")
            return
        
        # Concatenate all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Determine output format based on extension
        output_ext = os.path.splitext(output_file)[1].lower()
        
        # Save to output file
        if output_ext == '.csv':
            combined_df.to_csv(output_file, index=False)
        elif output_ext in ['.xls', '.xlsx']:
            combined_df.to_excel(output_file, index=False, engine='openpyxl')
        else:
            # Default to CSV if no extension or unknown extension
            output_file += '.csv'
            combined_df.to_csv(output_file, index=False)
        
        print(f"\n✅ Successfully concatenated {len(input_files)} files into '{output_file}'")
        print(f"📊 Total rows: {len(combined_df)}")
        print(f"📋 Total columns: {len(combined_df.columns)}")
        print(f"📁 Columns: {', '.join(combined_df.columns.tolist())}")
        
    except Exception as e:
        print(f"❌ Error during concatenation: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("="*50)
        print("CSV/Excel File Concatenator")
        print("="*50)
        print("Usage: python concate.py file1.csv file2.xls [file3.xlsx ...]")
        print("\nExamples:")
        print("  python concate.py 1.csv 2.csv 3.csv")
        print("  python concate.py data1.xlsx data2.xlsx data3.csv")
        print("  python concate.py report1.xls report2.xls")
        print("\nSupports: .csv, .xls, .xlsx files")
        print("Output: Auto-detects format or uses timestamp filename")
    else:
        input_files = sys.argv[1:]  # All arguments except script name
        concatenate_files(input_files)


