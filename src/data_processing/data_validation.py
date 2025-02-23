import pandas as pd
import json

def main():
    # Load the unified dataset
    data_path = 'data/unified/unified_dataset_cleaned.csv'
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error loading data from {data_path}: {e}")
        return

    # Basic overview: shape and info
    print("=== Basic Overview ===")
    print(f"Data shape (rows, columns): {df.shape}\n")
    print("Data info:")
    df.info()
    print("\n")

    # Count missing values per column
    print("=== Missing Values per Column ===")
    missing_counts = df.isnull().sum()
    print(missing_counts)
    print("\n")

    # Descriptive statistics for numeric columns
    print("=== Descriptive Statistics (Numeric Columns) ===")
    print(df.describe())
    print("\n")

    # Define critical columns for the models
    critical_columns = [
        'Transport_direction',  # needed for BBB and Transport State Models
        'CellLoc',              # needed for BBB and Transport State Models
        'SMILES',               # used in BBB model (if available)
        'logBB',                # blood-brain barrier coefficient
        'Direction',            # for Transport State Model (producing/degrading)
        'aggregated_brain_expression',  # for Brain Region Effect Model
        'is_neurotransmitter',  # for Brain Region Effect Model
        'is_hormone'            # for Brain Region Effect Model
    ]

    # Check for missing values in the critical columns
    print("=== Missing Values in Critical Columns ===")
    for col in critical_columns:
        if col in df.columns:
            missing = df[col].isnull().sum()
            print(f"{col}: {missing} missing value(s)")
        else:
            print(f"{col}: Column not found in the dataset!")
    print("\n")

    # Print unique values for the critical columns
    print("=== Unique Values in Critical Columns ===")
    for col in critical_columns:
        if col in df.columns:
            unique_vals = df[col].dropna().unique()
            print(f"{col} ({len(unique_vals)} unique values): {unique_vals}")
        else:
            print(f"{col}: Column not found in the dataset!")
    print("\n")

    # Inspect a sample of the 'aggregated_brain_expression' column to verify JSON/dict formatting
    if 'aggregated_brain_expression' in df.columns:
        sample_value = df['aggregated_brain_expression'].dropna().iloc[0]
        try:
            # Replace single quotes with double quotes if necessary
            expr_dict = json.loads(sample_value.replace("'", "\""))
            print("=== Sample aggregated_brain_expression Parsed as JSON ===")
            print(expr_dict)
        except Exception as e:
            print("Could not parse 'aggregated_brain_expression' as JSON. Check data formatting.")
            print(f"Error: {e}")

if __name__ == '__main__':
    main()
