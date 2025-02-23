import numpy as np

def aggregate_predictions(predictions):
    """
    Aggregates brain effect predictions across all metabolites.
    - Sums the weighted outputs across all metabolites.
    - Normalizes by the total concentration.
    - Stores each metabolite's values in a dictionary.

    Returns:
        A dictionary with:
            - "region_concentration": {Metabolite_Name: [values]}
            - "accumulation_rates": {Metabolite_Name: [values]}
            - "interference_patterns": {Metabolite_Name: [values]}
    """
    aggregated = {
        "region_concentration": {},
        "accumulation_rates": {},
        "interference_patterns": {}
    }
    total_concentration = 0.0

    for pred in predictions:
        # Use the actual metabolite name from the prediction, with a more descriptive fallback
        metabolite_name = pred.get("name", "Unknown_Metabolite")
        conc = pred.get("concentration", 1.0)
        total_concentration += conc

        be_pred = pred.get("brain_effect_prediction", {})

        # Convert lists to NumPy arrays and flatten to 1D
        region_values = np.array(be_pred.get("region_concentration", [[0.0] * 13])).flatten()
        acc_values = np.array(be_pred.get("accumulation_rates", [[0.0] * 13])).flatten()
        int_values = np.array(be_pred.get("interference_patterns", [[0.0] * 13])).flatten()

        # Store results per metabolite instead of lists
        aggregated["region_concentration"][metabolite_name] = region_values.tolist()
        aggregated["accumulation_rates"][metabolite_name] = acc_values.tolist()
        aggregated["interference_patterns"][metabolite_name] = int_values.tolist()

    return aggregated

if __name__ == '__main__':
    # Test aggregator with dummy predictions.
    dummy_preds = [{'bbb_prediction': {'crossing_probability': [[0.9613933563232422]], 'transport_rate': [[-0.8243483304977417]], 'state_distribution': [[0.7824651598930359, 0.19023825228214264, 0.027296600863337517]]}, 'transport_state_prediction': {'time_series': [[0.18526431918144226, 0.20591998100280762, 0.2075623869895935, 0.18108084797859192, 0.19758453965187073, 0.19582217931747437, 0.19279098510742188, 0.1959656924009323, 0.20179331302642822, 0.16795668005943298, 0.20009955763816833, 0.20976267755031586, 0.20055246353149414, 0.20289921760559082, 0.17908595502376556, 0.20212413370609283, 0.18935705721378326, 0.17437872290611267, 0.1742105633020401, 0.20962834358215332, 0.19459281861782074, 0.17059800028800964, 0.20230236649513245, 0.2074505239725113, 0.20029966533184052, 0.20153430104255676, 0.2117278128862381, 0.1803825944662094, 0.17170380055904388, 0.2153051197528839]], 'transition_rates': [[0.23157626390457153, 0.2260204255580902, 0.25138697028160095, 0.20851358771324158, 0.235065758228302, 0.22172996401786804, 0.22455748915672302, 0.1974298506975174, 0.22083145380020142]], 'steady_state': [[0.3339472711086273, 0.31658434867858887, 0.3494683802127838]]}, 'brain_effect_prediction': {'region_concentration': [[0.8734175562858582, 0.4191829264163971, 0.8617703318595886, 0.724186360836029, 0.7022993564605713, 0.37027469277381897, 1.162819504737854, 0.5733842849731445, 0.18155436217784882, 0.2589813768863678, 0.9765827059745789, 0.6800806522369385, 0.7717145681381226]], 'accumulation_rates': [[0.5846498012542725, 0.6081812381744385, 0.4510299265384674, 0.5695000886917114, 0.7397624254226685, 0.5292357802391052, 0.3652268052101135, 0.16228431463241577, 0.42442286014556885, 0.6604949235916138, 0.6830834150314331, 0.5472559928894043, 0.43672096729278564]], 'interference_patterns': [[0.14822709560394287, 0.07604321837425232, 0.0841614156961441, 0.03911794349551201, 0.10183227062225342, 0.06738702207803726, 0.07815752923488617, 0.027974383905529976, 0.07830190658569336, 0.06109462305903435, 0.0359550341963768, 0.05338321998715401, 0.1483643651008606]]}, 'concentration': 1.0}]
    aggregated_output = aggregate_predictions(dummy_preds)
    print("Aggregated output:")
    print(aggregated_output)
