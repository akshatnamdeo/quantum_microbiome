import json

def flatten_if_nested(values):
    """
    If values is a list containing a single list, return that inner list.
    Otherwise, return values unchanged.
    """
    if isinstance(values, list) and len(values) == 1 and isinstance(values[0], list):
        return values[0]
    return values

def format_output(aggregated_predictions):
    """
    Format the aggregated predictions into a user-friendly output.
    
    The aggregated_predictions dictionary is expected to have keys:
        - "region_concentration"
        - "accumulation_rates"
        - "interference_patterns"
    Each is a list of 13 values corresponding to the following brain regions:
        amygdala, basal ganglia, cerebellum, cerebral cortex, choroid plexus,
        hippocampal formation, hypothalamus, medulla oblongata, midbrain, pons,
        spinal cord, thalamus, white matter.
    
    Returns a tuple (json_output, text_summary):
        - json_output: a JSON-formatted string of the aggregated results.
        - text_summary: a plain text summary.
    """
    regions = [
        "amygdala", "basal ganglia", "cerebellum", "cerebral cortex", "choroid plexus",
        "hippocampal formation", "hypothalamus", "medulla oblongata", "midbrain", "pons",
        "spinal cord", "thalamus", "white matter"
    ]
    
    formatted = {}
    for key, values in aggregated_predictions.items():
        # Flatten if the list is nested
        flat_values = flatten_if_nested(values)
        formatted[key] = {region: round(val, 3) for region, val in zip(regions, flat_values)}
    
    json_output = json.dumps(formatted, indent=4)
    
    # Create a text summary
    summary_lines = []
    for key, region_values in formatted.items():
        summary_lines.append(f"{key.replace('_', ' ').capitalize()}:")
        for region, value in region_values.items():
            summary_lines.append(f"    {region}: {value}")
    text_summary = "\n".join(summary_lines)
    
    return json_output, text_summary

if __name__ == '__main__':
    # Test formatter with dummy aggregated predictions.
    dummy_aggregated = {'region_concentration': [[0.8734175562858582, 0.4191829264163971, 0.8617703318595886, 0.724186360836029, 0.7022993564605713, 0.37027469277381897, 1.162819504737854, 0.5733842849731445, 0.18155436217784882, 0.2589813768863678, 0.9765827059745789, 0.6800806522369385, 0.7717145681381226]], 'accumulation_rates': [[0.5846498012542725, 0.6081812381744385, 0.4510299265384674, 0.5695000886917114, 0.7397624254226685, 0.5292357802391052, 0.3652268052101135, 0.16228431463241577, 0.42442286014556885, 0.6604949235916138, 0.6830834150314331, 0.5472559928894043, 0.43672096729278564]], 'interference_patterns': [[0.14822709560394287, 0.07604321837425232, 0.0841614156961441, 0.03911794349551201, 0.10183227062225342, 0.06738702207803726, 0.07815752923488617, 0.027974383905529976, 0.07830190658569336, 0.06109462305903435, 0.0359550341963768, 0.05338321998715401, 0.1483643651008606]]}
    json_out, text_out = format_output(dummy_aggregated)
    print("JSON Output:")
    print(json_out)
    print("\nText Summary:")
    print(text_out)
