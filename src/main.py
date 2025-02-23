import os
import sys
import json
import numpy as np
import pandas as pd
from flask import Flask, request, Response, stream_with_context, jsonify
from flask_cors import CORS

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

def normalize_hmdb_id(hmdb_id: str) -> str:
    """
    Ensures HMDB IDs have the correct zero-padding.
    Converts 'HMDB00190' to 'HMDB0000190' for matching with dataset.
    """
    if hmdb_id.startswith("HMDB") and len(hmdb_id) < 11:  # HMDB IDs should be 11 characters long
        return "HMDB" + hmdb_id[4:].zfill(7)
    return hmdb_id  # Return unchanged if already correct

def load_metabolite_record(metabolite_name, hmdb_id=None, csv_path='data/unified/unified_dataset_cleaned.csv'):
    """
    Load the dataset and return the first matching record by HMDB ID or metabolite name.
    Prioritizes HMDB ID matching over name matching.
    """
    df = pd.read_csv(csv_path)

    # Try matching by HMDB ID first (if provided)
    if hmdb_id:
        normalized_hmdb = normalize_hmdb_id(hmdb_id)
        matches = df[df['hmdb_id'].astype(str) == normalized_hmdb]
        if not matches.empty:
            return matches.iloc[0]

    # If no match by HMDB ID, try matching by name
    matches = df[df['name'].str.lower() == metabolite_name.lower()]
    return matches.iloc[0] if not matches.empty else None

def create_pipeline_inputs(llm_metabolites, csv_path='data/unified/unified_dataset_cleaned.csv'):
    """
    Given a list of metabolites from the LLM (each a dict with "name", "hmdb_id", and "amount"),
    look up the full record in the dataset, process it into input vectors,
    and attach the concentration.
    """
    inputs = []
    for met in llm_metabolites:
        record = load_metabolite_record(met["name"], met.get("hmdb_id"), csv_path)
        
        if record is None:
            print(f"⚠️ Metabolite '{met['name']}' (HMDB: {met.get('hmdb_id')}) not found in dataset.")
            continue
        
        processed = process_record_to_inputs(record)
        processed["concentration"] = met["amount"]
        processed["name"] = met["name"]  # Ensure 'name' key exists for later stages
        inputs.append(processed)

    return inputs

def generate_chart_data(aggregated_results):
    """
    Generates structured chart data for visualization.
    """
    if not isinstance(aggregated_results, dict):
        print(f"❌ Error: Expected dict, got {type(aggregated_results)}")
        return {}

    print("DEBUG: Aggregated Results Structure:", json.dumps(aggregated_results, indent=4))

    required_keys = ["region_concentration", "accumulation_rates", "interference_patterns"]
    for key in required_keys:
        if key not in aggregated_results:
            print(f"❌ Error: Missing key '{key}' in aggregated_results.")
            return {}

        if not isinstance(aggregated_results[key], dict):
            print(f"❌ Error: '{key}' should be a dictionary but got {type(aggregated_results[key])}")
            return {}

    # Extract necessary data
    region_concentration = aggregated_results['region_concentration']
    accumulation_rates = aggregated_results['accumulation_rates']
    interference_patterns = aggregated_results['interference_patterns']

    # Validate that extracted values are dictionaries
    if not all(isinstance(val, dict) for val in [region_concentration, accumulation_rates, interference_patterns]):
        print("Error: Expected dict values in aggregated_results.")
        return {}

    # Get metabolite names
    metabolites = list(region_concentration.keys())

    # Brain regions list
    brain_regions = [
        "Amygdala", "Basal Ganglia", "Cerebellum", "Cerebral Cortex", "Choroid Plexus",
        "Hippocampal Formation", "Hypothalamus", "Medulla Oblongata", "Midbrain", "Pons",
        "Spinal Cord", "Thalamus", "White Matter"
    ]

    # Stacked Bar Chart Data: Contributions by Metabolites
    stacked_bar_data = {
        "brain_regions": brain_regions,
        "metabolites": metabolites,
        "values": {met: [region_concentration[met][i] for i in range(len(brain_regions))] for met in metabolites}
    }

    # Radar Chart Data: Overall Aggregated Effects
    radar_chart_data = {
        "brain_regions": brain_regions,
        "values": [sum(region_concentration[met][i] for met in metabolites) for i in range(len(brain_regions))]
    }

    # Metabolite Contribution Heatmap Data
    heatmap_data = {
        "metabolites": metabolites,
        "brain_regions": brain_regions,
        "values": {met: [region_concentration[met][i] for i in range(len(brain_regions))] for met in metabolites}
    }

    # Grouped Bar Chart Dashboard Data
    grouped_bar_data = {
        "brain_regions": brain_regions,
        "region_concentration": {met: [region_concentration[met][i] for i in range(len(brain_regions))] for met in metabolites},
        "accumulation_rates": {met: [accumulation_rates[met][i] for i in range(len(brain_regions))] for met in metabolites},
        "interference_patterns": {met: [interference_patterns[met][i] for i in range(len(brain_regions))] for met in metabolites}
    }

    return {
        "stacked_bar_chart": stacked_bar_data,
        "radar_chart": radar_chart_data,
        "heatmap": heatmap_data,
        "grouped_bar_chart": grouped_bar_data
    }

# ---------------------------
# Flask application with streaming response
# ---------------------------
app = Flask(__name__)

CORS(app, resources={r"/process": {"origins": "http://localhost:3000"}})

@app.route("/process", methods=["POST"])
def process():
    """
    Endpoint that accepts user input in JSON format.
    """
    data = request.get_json()
    if not data or "user_input" not in data:
        return jsonify({"error": "Invalid input. Expected JSON with 'user_input' key."}), 400

    user_input = data["user_input"]

    # Step 1: Extract metabolites using LLM
    try:
        llm_output = llm_integration.get_llm_output(user_input)
        print(llm_output)  # Debug print
        
        # Extract just the metabolites list for pipeline processing
        llm_metabolites = llm_output.get("metabolites", [])
        
    except Exception as e:
        return jsonify({"error": "LLM processing failed.", "details": str(e)}), 500

    if not llm_metabolites:
        return jsonify({"error": "No metabolites extracted from input."}), 400

    # Step 2: Look up records in the dataset to create pipeline inputs
    pipeline_inputs = create_pipeline_inputs(llm_metabolites)
    if not pipeline_inputs:
        return jsonify({"error": "No valid metabolite records found."}), 400

    # Create a generator to stream messages
    def generate_stream():
        statuses = []
        
        # Notify that the simulation is starting
        simulation_start_msg = llm_integration.generate_status_message("simulation_start", {})
        statuses.append(simulation_start_msg)
        yield json.dumps({"status": simulation_start_msg}) + "\n"

        predictions = []
        for idx, met in enumerate(pipeline_inputs):
            # Notify that we are processing this metabolite
            processing_msg = llm_integration.generate_status_message(
                "processing_metabolite", 
                {"name": met["name"], "index": idx + 1, "total": len(pipeline_inputs)}
            )
            statuses.append(processing_msg)
            yield json.dumps({"status": processing_msg}) + "\n"

            # Run prediction for the metabolite
            pred = model_orchestrator.predict_metabolite(met)
            predictions.append(pred)

            # Generate a custom LLM status message about the prediction
            prediction_msg = llm_integration.generate_status_message(
                "prediction_complete", 
                {"name": met["name"], "prediction": pred}
            )
            statuses.append(prediction_msg)
            yield json.dumps({"status": prediction_msg}) + "\n"

        # Step 3: Aggregate results
        aggregation_msg = llm_integration.generate_status_message("aggregation_complete", {})
        statuses.append(aggregation_msg)
        yield json.dumps({"status": aggregation_msg}) + "\n"

        aggregated = aggregator.aggregate_predictions(predictions)

        # Format aggregated results before generating charts and final summary
        json_out, text_out = output_formatter.format_output(aggregated)

        # Step 4: Generate Charts
        chart_msg = llm_integration.generate_status_message("chart_generation", {})
        statuses.append(chart_msg)
        yield json.dumps({"status": chart_msg}) + "\n"

        charts = generate_chart_data(aggregated)

        # Step 5: Generate Final Summary (Streaming)
        yield json.dumps({"status": "📜 Generating final output..."}) + "\n"

        summary_chunks = []
        for chunk in llm_integration.generate_final_summary(
            user_input=user_input,
            extracted_data=llm_output,  # Pass full LLM output here
            pipeline_results=text_out
        ):
            if chunk:
                summary_chunks.append(chunk)
            
        final_summary = "".join(summary_chunks)

        # Final Response
        final_output = {
            "summary": final_summary,
            "charts": charts,
            "status": statuses
        }
        yield json.dumps({"final_output": final_output}) + "\n"

    return Response(stream_with_context(generate_stream()), mimetype="application/json")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)