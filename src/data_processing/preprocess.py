import pandas as pd

def process_microbes():
    # Load the microbe-metabolite links dataset and extract relevant columns.
    df = pd.read_csv("data/raw/recon-store-microbes-1.tsv", sep="\t")
    df_microbes = df[[
        "organism", 
        "metabolites", 
        "metabolism"
    ]]
    df_microbes.to_csv("data/processed/microbes_processed.csv", index=False)
    return df_microbes

def process_metabolites():
    # Load the HMDB neuro/hormone dataset as the core metabolite info.
    df = pd.read_csv("data/processed/hmdb_neuro_hormone_gene_data.csv")
    df_metabolites = df[[
        "hmdb_id",              # Primary identifier
        "name",                 # Metabolite name
        "is_neurotransmitter",  # Yes/No flag for neurotransmitter
        "is_hormone",           # Yes/No flag for hormone
        "gene_names",           # Linked genes for brain expression connection
        "uniprot_ids",          # Linked proteins
        "brain_tissues",        # Target brain areas
        "cellular_locations",   # Cellular locations where the metabolite appears
        "pathways",             # Affected pathways
        "description"           # Additional details if needed
    ]]
    df_metabolites.to_csv("data/processed/metabolites_processed.csv", index=False)
    return df_metabolites

def process_brain_expression():
    # Extract key columns from the brain region expression dataset.
    df = pd.read_csv("data/raw/rna_brain_region_hpa.tsv", sep="\t")
    df_brain = df[[
        "Gene",         # ENSG ID
        "Gene name",    # Gene symbol
        "Brain region", # Specific brain region
        "nTPM"          # Normalized expression level
    ]]
    df_brain.to_csv("data/processed/brain_expression_processed.csv", index=False)
    return df_brain

def process_bbb_crossing():
    # Process the blood-brain barrier crossing dataset to extract required columns.
    df = pd.read_csv("data/raw/B3DB_regression.tsv", sep="\t")
    df_bbb = df[[
        "compound_name",  # Name of the compound
        "SMILES",         # SMILES notation
        "CID",            # Compound identifier
        "logBB"           # Blood-brain barrier coefficient
    ]]
    df_bbb.to_csv("data/processed/bbb_crossing_processed.csv", index=False)
    return df_bbb

def process_transport():
    # Extract relevant columns from the transport & location dataset.
    df = pd.read_csv("data/raw/PD_0.4.5_processed.csv")
    df_transport = df[[
        "HMDB",               # Links to HMDB dataset (metabolite id)
        "MetName",            # Metabolite name
        "Transport_direction",# Transport direction
        "Direction",          # Producing or degrading
        "CellLoc",            # Cellular locations
        "TissueLoc"           # Tissue presence
    ]]
    df_transport.to_csv("data/processed/transport_processed.csv", index=False)
    return df_transport

def main():
    process_microbes()
    process_metabolites()
    process_brain_expression()
    process_bbb_crossing()
    process_transport()
    print("Processing complete. All filtered files have been saved to data/processed/.")

if __name__ == "__main__":
    main()
