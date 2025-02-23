import os
import json
import numpy as np
import pandas as pd
import sys

# Get the absolute path to your project's root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) # go up one directory.

# Add the project root to sys.path
sys.path.append(project_root)

from src.pipeline import model_orchestrator, aggregator, output_formatter

def parse_cellloc(cellloc_str):
    try:
        locs = json.loads(cellloc_str.replace("'", "\""))
        if isinstance(locs, list):
            return locs
        else:
            return [str(locs)]
    except:
        return []

def parse_brain_tissues(bt_str):
    if pd.isnull(bt_str):
        return ['Unknown']
    return [s.strip() for s in bt_str.split('|')]

def parse_json_expression(expr_str):
    try:
        expr_dict = json.loads(expr_str.replace("'", "\""))
    except Exception as e:
        expr_dict = {}
    # Fixed order for brain regions
    BRAIN_REGIONS = [
        "amygdala", "basal ganglia", "cerebellum", "cerebral cortex", "choroid plexus",
        "hippocampal formation", "hypothalamus", "medulla oblongata", "midbrain", "pons",
        "spinal cord", "thalamus", "white matter"
    ]
    return [expr_dict.get(region, 0.0) for region in BRAIN_REGIONS]

def one_hot_transport_direction(value):
    # Fixed order: ["in", "out", "in | out", "unknown"]
    mapping = {
        "in": [1, 0, 0, 0],
        "out": [0, 1, 0, 0],
        "in | out": [0, 0, 1, 0],
        "unknown": [0, 0, 0, 1]
    }
    v = value.strip().lower()
    return mapping.get(v, mapping["unknown"])

def one_hot_direction(val):
    # For the Direction column; possible values: "in", "out", "degrading", "producing", "degrading | producing"
    mapping = {
        "in": [1, 0, 0, 0, 0],
        "out": [0, 1, 0, 0, 0],
        "degrading": [0, 0, 1, 0, 0],
        "producing": [0, 0, 0, 1, 0],
        "degrading | producing": [0, 0, 0, 0, 1]
    }
    v = val.strip().lower()
    return mapping.get(v, [0, 0, 0, 0, 1])

def process_record_to_inputs(record):
    """
    Convert a metabolite record (Pandas Series) into the input vectors expected by the models.
    Assumes the record contains columns:
        - "Transport_direction"
        - "CellLoc"
        - "Direction"
        - "is_neurotransmitter"
        - "is_hormone"
        - "brain_tissues"
        - "aggregated_brain_expression"
    For simplicity, some encodings are simulated with fixed dummy vectors.
    """
    # --- BBB Transport Model input (34 dims) ---
    # one-hot encoding for Transport_direction (4 dims)
    transport_dir = one_hot_transport_direction(record["Transport_direction"])
    
    # For CellLoc, we parse the value and create a dummy multi-hot encoding.
    # For example, if "Mitochondria" is present, we encode as [1, 0, 0, 0, 0]; otherwise zeros.
    cellloc_list = parse_cellloc(record["CellLoc"])
    cellloc_encoding = [1, 0, 0, 0, 0] if "Mitochondria" in cellloc_list else [0, 0, 0, 0, 0]
    
    # Additional dummy vector for CellLoc (5 dims)
    additional_cellloc = [0, 0, 0, 0, 0]
    
    # Binary flags for is_neurotransmitter and is_hormone (2 dims)
    binary_flags = [
        1 if record["is_neurotransmitter"].strip().lower() == "yes" else 0,
        1 if record["is_hormone"].strip().lower() == "yes" else 0
    ]
    
    # Dummy encoding for brain_tissues (simulate with a fixed vector of length 5)
    brain_tissues_dummy = [1, 1, 1, 1, 1]
    
    # Parse aggregated_brain_expression into a 13-dimensional vector.
    expr_vector = parse_json_expression(record["aggregated_brain_expression"])
    
    # Concatenate for BBB input: 4 + 5 + 5 + 2 + 5 + 13 = 34 dims.
    bbb_input = transport_dir + cellloc_encoding + additional_cellloc + binary_flags + brain_tissues_dummy + expr_vector
    
    # --- Transport State Model input (21 dims) ---
    # It is constructed as: one-hot for Transport_direction (4 dims) +
    # dummy CellLoc encoding (5 dims) + one-hot for Direction (5 dims) + dummy current state (3 dims)
    # Total so far: 4+5+5+3 = 17; pad with 4 zeros to reach 21.
    ts_transport = transport_dir  # 4 dims
    ts_cellloc = cellloc_encoding   # 5 dims (reuse)
    ts_direction = one_hot_direction(record["Direction"])  # 5 dims
    current_state = [0.33, 0.33, 0.34]  # dummy distribution (3 dims)
    ts_input = ts_transport + ts_cellloc + ts_direction + current_state
    # Pad with zeros if needed:
    ts_input += [0.0] * (21 - len(ts_input))
    
    # --- Brain Effect Model input (19 dims) ---
    # It is: aggregated expression (13 dims) + binary flags (2 dims) + dummy current state (3 dims) + dummy BBB probability (1 dim)
    be_input = expr_vector + binary_flags + current_state + [0.5]
    # Total: 13+2+3+1 = 19 dims.
    
    return {
        "bbb_input": bbb_input,
        "transport_state_input": ts_input,
        "brain_effect_input": be_input
    }

def load_metabolite_record(metabolite_name, csv_path='data/unified/unified_dataset_cleaned.csv'):
    """
    Load the unified dataset and return the first record matching the given metabolite name (case-insensitive).
    """
    df = pd.read_csv(csv_path)
    matches = df[df['name'].str.lower() == metabolite_name.lower()]
    if len(matches) > 0:
        return matches.iloc[0]
    return None

def create_pipeline_inputs(llm_metabolites, csv_path='data/unified/unified_dataset_cleaned.csv'):
    """
    Given a list of metabolites from the LLM (each a dict with "name" and "amount"),
    look up the full record in the unified dataset, process it into input vectors,
    and attach the concentration.
    Returns a list of metabolite input dictionaries.
    """
    inputs = []
    for met in llm_metabolites:
        record = load_metabolite_record(met["name"], csv_path=csv_path)
        if record is None:
            print(f"Metabolite '{met['name']}' not found in dataset.")
            continue
        processed = process_record_to_inputs(record)
        processed["concentration"] = met["amount"]
        inputs.append(processed)
    return inputs

def main():
    # Simulate LLM output with two metabolites: Estrone sulfate and Formaldehyde.
    llm_output = [
        {"name": "Estrone sulfate", "amount": 1.0},
        {"name": "Formaldehyde", "amount": 0.8}
    ]
    
    # Create pipeline inputs by looking up the full record for each metabolite.
    metabolites = create_pipeline_inputs(llm_output)
    if not metabolites:
        print("No valid metabolites found. Exiting.")
        return
    
    # Run the individual models pipeline.
    predictions = model_orchestrator.run_pipeline(metabolites)
    print("Individual Pipeline Predictions:")
    for idx, pred in enumerate(predictions):
        print(f"Metabolite {idx+1} ({llm_output[idx]['name']}):")
        print(pred)
    
    # Aggregate brain effect predictions from all metabolites.
    aggregated = aggregator.aggregate_predictions(predictions)
    print("\nAggregated Brain Effect Predictions:")
    print(aggregated)
    
    # Format final output.
    json_output, text_summary = output_formatter.format_output(aggregated)
    print("\nFinal Formatted Output (JSON):")
    print(json_output)
    print("\nFinal Formatted Output (Text Summary):")
    print(text_summary)

if __name__ == '__main__':
    main()
