import os
import sys
import json
import numpy as np
import pandas as pd
from flask import Flask, request, Response, stream_with_context, jsonify

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Import pipeline modules
from src.pipeline import model_orchestrator, aggregator, output_formatter, llm_integration

# ---------------------------
# Helper functions for processing records
# ---------------------------
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
    BRAIN_REGIONS = [
        "amygdala", "basal ganglia", "cerebellum", "cerebral cortex", "choroid plexus",
        "hippocampal formation", "hypothalamus", "medulla oblongata", "midbrain", "pons",
        "spinal cord", "thalamus", "white matter"
    ]
    return [expr_dict.get(region, 0.0) for region in BRAIN_REGIONS]

def one_hot_transport_direction(value):
    mapping = {
        "in": [1, 0, 0, 0],
        "out": [0, 1, 0, 0],
        "in | out": [0, 0, 1, 0],
        "unknown": [0, 0, 0, 1]
    }
    v = value.strip().lower()
    return mapping.get(v, mapping["unknown"])

def one_hot_direction(val):
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
    Assumes the record contains:
      - "Transport_direction", "CellLoc", "Direction",
      - "is_neurotransmitter", "is_hormone",
      - "brain_tissues", "aggregated_brain_expression"
    Some encodings use fixed dummy vectors where needed.
    """
    # BBB Transport Model input (34 dims)
    transport_dir = one_hot_transport_direction(record["Transport_direction"])
    cellloc_list = parse_cellloc(record["CellLoc"])
    cellloc_encoding = [1, 0, 0, 0, 0] if "mitochondria" in [l.lower() for l in cellloc_list] else [0, 0, 0, 0, 0]
    additional_cellloc = [0, 0, 0, 0, 0]
    binary_flags = [
        1 if record["is_neurotransmitter"].strip().lower() == "yes" else 0,
        1 if record["is_hormone"].strip().lower() == "yes" else 0
    ]
    brain_tissues_dummy = [1, 1, 1, 1, 1]
    expr_vector = parse_json_expression(record["aggregated_brain_expression"])
    bbb_input = transport_dir + cellloc_encoding + additional_cellloc + binary_flags + brain_tissues_dummy + expr_vector

    # Transport State Model input (21 dims)
    ts_transport = transport_dir
    ts_cellloc = cellloc_encoding
    ts_direction = one_hot_direction(record["Direction"])
    current_state = [0.33, 0.33, 0.34]  # Dummy current state
    ts_input = ts_transport + ts_cellloc + ts_direction + current_state
    ts_input += [0.0] * (21 - len(ts_input))

    # Brain Effect Model input (19 dims)
    be_input = expr_vector + binary_flags + current_state + [0.5]  # Dummy BBB probability
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
    if not matches.empty:
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

# ---------------------------
# Flask application with streaming response
# ---------------------------
app = Flask(__name__)

@app.route("/process", methods=["POST"])
def process():
    """
    Endpoint that accepts user input in JSON format.
    Expects a JSON object with key "user_input".
    The llm_integration module converts this into an llm_output (a list of metabolites).
    Then the pipeline looks up records, processes inputs, runs predictions,
    aggregates results, and streams final summary, charts data, and status messages in real time.
    """
    data = request.get_json()
    if not data or "user_input" not in data:
        return jsonify({"error": "Invalid input. Expected JSON with 'user_input' key."}), 400

    user_input = data["user_input"]

    # Get metabolites list from llm_integration (assumed implemented)
    try:
        llm_metabolites = llm_integration.get_llm_output(user_input)
    except Exception as e:
        return jsonify({"error": "LLM processing failed.", "details": str(e)}), 500

    if not llm_metabolites:
        return jsonify({"error": "No metabolites extracted from input."}), 400

    # Create pipeline inputs by looking up records in the unified dataset.
    pipeline_inputs = create_pipeline_inputs(llm_metabolites)
    if not pipeline_inputs:
        return jsonify({"error": "No valid metabolite records found."}), 400

    # Create a generator to stream messages.
    def generate_stream():
        # Stream status messages from each metabolite's processing.
        statuses = []
        # Run predictions for each metabolite individually.
        predictions = []
        for idx, met in enumerate(pipeline_inputs):
            pred = model_orchestrator.predict_metabolite(met)
            predictions.append(pred)
            # Generate a status message for this metabolite.
            status_msg = llm_integration.generate_status(pred)
            statuses.append(status_msg)
            # Yield a JSON message for this metabolite's status.
            yield json.dumps({"status": status_msg}) + "\n"
        
        # Once all metabolites are processed, aggregate the brain effect predictions.
        aggregated = aggregator.aggregate_predictions(predictions)
        # Generate chart data (assume this function exists).
        charts = llm_integration.generate_chart_data(aggregated)
        # Generate final summary.
        final_summary = llm_integration.generate_final_summary(aggregated)
        
        # Build final response message.
        final_output = {
            "summary": final_summary,
            "charts": charts,
            "status": statuses
        }
        yield json.dumps({"final_output": final_output}) + "\n"
    
    return Response(stream_with_context(generate_stream()), mimetype="application/json")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
