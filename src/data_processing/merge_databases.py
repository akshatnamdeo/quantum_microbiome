import pandas as pd
import json
import re
from tqdm import tqdm  # Add this import

def clean_compound_name(name):
    if pd.isna(name):
        return ""
    # Convert to lowercase
    name = str(name).lower().strip()
    # Remove special characters and extra spaces
    name = re.sub(r'[^a-z0-9]', ' ', name)
    # Remove multiple spaces
    name = re.sub(r'\s+', ' ', name)
    return name.strip()

def aggregate_brain_expression(gene_names, brain_expr_df):
    if pd.isna(gene_names):
        return json.dumps({})
    # Split gene_names by "|" and convert to lowercase for matching
    genes = [g.strip().lower() for g in gene_names.split("|") if g.strip()]
    # Filter brain expression for matching Gene name (converted to lowercase)
    sub_df = brain_expr_df[brain_expr_df["Gene name"].str.lower().isin(genes)]
    expr_dict = {}
    for _, row in sub_df.iterrows():
        region = row["Brain region"]
        nTPM = row["nTPM"]
        if region in expr_dict:
            expr_dict[region].append(nTPM)
        else:
            expr_dict[region] = [nTPM]
    # Compute average nTPM per region if there are multiple values
    aggregated = {region: sum(values)/len(values) for region, values in expr_dict.items()}
    return json.dumps(aggregated)

def has_brain_data(row):
    # Check if at least one brain-related field is not empty/unknown
    brain_fields = [
        row.get("brain_tissues", ""), 
        row.get("CellLoc", ""), 
        row.get("aggregated_brain_expression", "")
    ]
    return any(str(field).strip().lower() not in ["", "none", "unknown", "{}", "nan"] for field in brain_fields)

def main():
    print("Loading datasets...")
    # Load datasets
    metabolites = pd.read_csv("data/cleaned/metabolites_cleaned_processed.csv")
    transport = pd.read_csv("data/cleaned/transport_cleaned_processed.csv")
    bbb = pd.read_csv("data/cleaned/bbb_crossing_cleaned_processed.csv")
    brain_expr = pd.read_csv("data/cleaned/brain_expression_cleaned_processed.csv")

    print("Cleaning and normalizing HMDB IDs...")
    # Clean and normalize HMDB IDs
    metabolites["hmdb_id_lower"] = metabolites["hmdb_id"].str.lower().str.strip()
    transport["HMDB_lower"] = transport["HMDB"].str.lower().str.strip()
    
    # Merge transport information
    print("Merging transport information...")
    merged = pd.merge(metabolites, transport, 
                     left_on="hmdb_id_lower", 
                     right_on="HMDB_lower", 
                     how="inner",
                     validate="many_to_many")
    print("After merging metabolites and transport:", merged.shape)
    
    print("Cleaning compound names...")
    # Clean and normalize compound names for better matching
    tqdm.pandas(desc="Cleaning metabolite names")
    merged["name_clean"] = merged["name"].progress_apply(clean_compound_name)
    
    tqdm.pandas(desc="Cleaning BBB compound names")
    bbb["compound_name_clean"] = bbb["compound_name"].progress_apply(clean_compound_name)
    
    # Remove rows with empty compound names
    bbb = bbb[bbb["compound_name_clean"] != ""]
    
    # Log unique values in the keys to inspect for mismatches
    print("\nSample cleaned metabolite names:")
    print(merged["name_clean"].head().tolist())
    print("\nSample cleaned BBB compound names:")
    print(bbb["compound_name_clean"].head().tolist())
    
    # Merge BBB crossing data using a left join with cleaned names
    print("\nMerging BBB crossing data...")
    merged = pd.merge(merged, bbb, 
                     left_on="name_clean", 
                     right_on="compound_name_clean", 
                     how="left",
                     validate="many_to_many")
    print("After merging with BBB crossing (left join):", merged.shape)
    
    # Aggregate brain expression data
    print("\nAggregating brain expression data...")
    tqdm.pandas(desc="Processing brain expression")
    merged["aggregated_brain_expression"] = merged["gene_names"].progress_apply(
        lambda x: aggregate_brain_expression(x, brain_expr)
    )
    
    # Final filtering: Remove rows where all brain-related fields are empty
    print("\nPerforming final filtering...")
    before_filter = merged.shape[0]
    final_merged = merged[merged.apply(has_brain_data, axis=1)]
    after_filter = final_merged.shape[0]
    print(f"Rows before final brain data filtering: {before_filter}")
    print(f"Rows after filtering: {after_filter}")

    # Save the final unified dataset
    print("\nSaving unified dataset...")
    output_path = "data/cleaned/unified_dataset.csv"
    final_merged.to_csv(output_path, index=False)
    print(f"Unified dataset saved to {output_path}")

    # Print sample of successful matches
    print("\nSample of successful BBB matches:")
    matched = final_merged[~final_merged["compound_name"].isna()]
    if not matched.empty:
        print(matched[["name", "compound_name"]].head())
    else:
        print("No successful BBB matches found!")

if __name__ == "__main__":
    main()