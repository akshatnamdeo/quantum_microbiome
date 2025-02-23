import pandas as pd
import numpy as np

def main():
    data_path = 'data/unified/unified_dataset_cleaned.csv'
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error loading data from {data_path}: {e}")
        return

    # Fill missing brain_tissues with a placeholder since it's used in grouping.
    df['brain_tissues_filled'] = df['brain_tissues'].fillna('Unknown')

    # Create a group key based on is_neurotransmitter, is_hormone, and brain_tissues.
    df['group_key'] = df['is_neurotransmitter'].astype(str) + "_" + df['is_hormone'].astype(str) + "_" + df['brain_tissues_filled'].astype(str)

    # For rows with non-missing CellLoc, compute the mode for each group.
    non_missing = df[df['CellLoc'].notnull()]
    group_mode = non_missing.groupby('group_key')['CellLoc'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)

    # For rows with missing CellLoc, check if their group has a candidate mode for imputation.
    missing = df[df['CellLoc'].isnull()]
    imputation_candidates = missing['group_key'].apply(lambda key: key in group_mode.index and pd.notnull(group_mode.loc[key]))
    num_possible = imputation_candidates.sum()
    total_missing = missing.shape[0]

    print("=== Imputation Analysis for CellLoc ===")
    print(f"Total missing CellLoc rows: {total_missing}")
    print(f"Number of missing CellLoc rows with candidate imputation based on grouping: {num_possible}")

    if total_missing > 0:
        ratio = num_possible / total_missing
        print(f"Imputation feasibility ratio: {ratio:.2%}")
        if ratio > 0:
            print("Conclusion: Alternative imputation based on similar metabolites is possible for at least some missing CellLoc values.")
        else:
            print("Conclusion: No alternative imputation option is possible based on the current grouping criteria.")
    else:
        print("No missing CellLoc values detected.")

if __name__ == '__main__':
    main()
