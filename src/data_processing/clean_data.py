import pandas as pd
import numpy as np

def clean_transport():
    # Read the processed BBB crossing file
    df = pd.read_csv("data/processed/transport_processed.csv")
    
    # Define critical columns that must be non-empty and not 'unknown'
    critical_columns = ["Direction", "CellLoc", "TissueLoc"]
    
    # Ensure HMDB and MetName are also non-empty
    key_columns = ["HMDB", "MetName"]
    
    # Remove rows where any key column is missing or marked as 'unknown'
    for col in key_columns:
        df[col] = df[col].astype(str).str.strip()
        df = df[~df[col].str.lower().isin(["", "none", "unknown"])]
    
    # Remove rows where any critical column is missing or marked as 'unknown'
    for col in critical_columns:
        df[col] = df[col].astype(str).str.strip()
        df = df[~df[col].str.lower().isin(["", "none", "unknown"])]
    
    # Clean up extraneous quotes in CellLoc and TissueLoc columns if they exist
    for col in ["CellLoc", "TissueLoc"]:
        # Remove double-double quotes that may appear because of embedded commas
        df[col] = df[col].str.replace('""', '"', regex=False)
    
    # Save the cleaned data to a new CSV file
    df.to_csv("data/cleaned/transport_cleaned_processed.csv", index=False)
    print("Cleaned transport data saved to data/cleaned/transport_cleaned_processed.csv")
    
    
def clean_microbes():
    # Read the processed microbes dataset
    df = pd.read_csv("data/processed/microbes_processed.csv")
    
    # Define key columns that must be non-empty and not 'unknown'
    key_columns = ["organism", "metabolites"]
    
    # Remove rows where any key column is missing or marked as 'unknown' or 'none'
    for col in key_columns:
        df[col] = df[col].astype(str).str.strip()
        df = df[~df[col].str.lower().isin(["", "none", "unknown"])]
    
    # Optionally, clean up any extra spaces in the metabolism column
    if "metabolism" in df.columns:
        df["metabolism"] = df["metabolism"].astype(str).str.strip()
    
    # Save the cleaned data to a new CSV file
    df.to_csv("data/cleaned/microbes_cleaned_processed.csv", index=False)
    print("Cleaned microbes data saved to data/cleaned/microbes_cleaned_processed.csv")

def clean_metabolites():
    # Define the dtype for columns expected to be strings to avoid mixed-type warnings
    dtype_spec = {
        "gene_names": str,
        "uniprot_ids": str,
        "brain_tissues": str,
        "cellular_locations": str,
        "pathways": str,
        "hmdb_id": str,
        "name": str,
        "is_neurotransmitter": str,
        "is_hormone": str,
        "description": str
    }
    
    # Read the processed metabolites dataset with explicit dtypes and low_memory disabled
    df = pd.read_csv("data/processed/metabolites_processed.csv", dtype=dtype_spec, low_memory=False)
    
    # Define key columns that must be non-empty and not 'unknown'
    key_columns = ["hmdb_id", "name"]
    for col in key_columns:
        df[col] = df[col].astype(str).str.strip()
        df = df[~df[col].str.lower().isin(["", "none", "unknown"])]
    
    # Define brain-related columns that are important for downstream analysis
    brain_columns = ["gene_names", "brain_tissues", "cellular_locations"]
    for col in brain_columns:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()
    
    # Remove rows where ALL brain-related columns are empty or marked as 'none' or 'unknown'
    def has_brain_info(row):
        return any(row[col].lower() not in ["", "none", "unknown"] for col in brain_columns)
    df = df[df.apply(has_brain_info, axis=1)]
    
    # Optionally, trim whitespace from the remaining columns
    other_columns = ["is_neurotransmitter", "is_hormone", "uniprot_ids", "pathways", "description"]
    for col in other_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    # Save the cleaned data to a new CSV file
    df.to_csv("data/cleaned/metabolites_cleaned_processed.csv", index=False)
    print("Cleaned metabolites data saved to data/cleaned/metabolites_cleaned_processed.csv")

def clean_brain_expression():
    # Read the processed brain expression dataset
    df = pd.read_csv("data/processed/brain_expression_processed.csv", low_memory=False)
    
    # Define key columns that must be non-empty and not 'unknown'
    key_columns = ["Gene", "Gene name", "Brain region"]
    for col in key_columns:
        df[col] = df[col].astype(str).str.strip()
        df = df[~df[col].str.lower().isin(["", "none", "unknown"])]
    
    # Convert nTPM column to numeric, dropping rows where conversion fails
    df["nTPM"] = pd.to_numeric(df["nTPM"], errors="coerce")
    df = df.dropna(subset=["nTPM"])
    
    # Optionally, drop duplicate rows
    df = df.drop_duplicates()
    
    # Save the cleaned data to a new CSV file
    df.to_csv("data/cleaned/brain_expression_cleaned_processed.csv", index=False)
    print("Cleaned brain expression data saved to data/cleaned/brain_expression_cleaned_processed.csv")
    
def clean_bbb_crossing():
    # Read the processed BBB crossing dataset
    df = pd.read_csv("data/processed/bbb_crossing_processed.csv", low_memory=False)
    
    # Define key columns that are required: compound_name and SMILES
    key_columns = ["compound_name"]
    for col in key_columns:
        df[col] = df[col].astype(str).str.strip()
        df = df[~df[col].str.lower().isin(["", "none", "unknown"])]
    
    # Clean the CID column: remove trailing "|" and convert to numeric.
    def clean_numeric(value):
        if isinstance(value, str):
            value = value.strip().replace("|", "")
        try:
            return float(value)
        except Exception:
            return np.nan
    
    df["CID"] = df["CID"].apply(clean_numeric)
    df["logBB"] = df["logBB"].apply(clean_numeric)
    
    # Drop rows where CID or logBB are missing (NaN) after conversion
    # df = df.dropna(subset=["CID", "logBB"])
    
    # Save the cleaned data to a new CSV file
    df.to_csv("data/cleaned/bbb_crossing_cleaned_processed.csv", index=False)
    print("Cleaned BBB crossing data saved to data/cleaned/bbb_crossing_cleaned_processed.csv")

if __name__ == "__main__":
    clean_bbb_crossing()
    clean_brain_expression()
    clean_metabolites()
    clean_microbes()
    clean_transport()
