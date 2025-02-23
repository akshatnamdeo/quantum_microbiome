import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils import custom_object_scope

# Import the custom layer so that we can register it when loading models.
# Adjust the import path if your QuantumDense is defined elsewhere.
class QuantumDense(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(QuantumDense, self).__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
    
    def build(self, input_shape):
        self.w_real = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='w_real'
        )
        self.w_imag = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='w_imag'
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )
        super(QuantumDense, self).build(input_shape)
    
    def call(self, inputs):
        real_part = tf.matmul(inputs, self.w_real)
        imag_part = tf.matmul(inputs, self.w_imag)
        magnitude = tf.sqrt(tf.square(real_part) + tf.square(imag_part)) + self.bias
        if self.activation is not None:
            return self.activation(magnitude)
        return magnitude

# Define paths to the final saved models
BBB_MODEL_PATH = os.path.join("src", "checkpoints", "h5", "final_bbb_transport_model.h5")
TRANSPORT_STATE_MODEL_PATH = os.path.join("src", "checkpoints", "h5", "final_transport_state_model.h5")
BRAIN_EFFECT_MODEL_PATH = os.path.join("src", "checkpoints", "h5", "final_brain_effect_model.h5")

def load_models():
    """
    Load the three pre-trained models from disk using a custom object scope
    so that the QuantumDense layer is recognized.
    """
    custom_objects = {"QuantumDense": QuantumDense}
    with custom_object_scope(custom_objects):
        bbb_model = keras.models.load_model(BBB_MODEL_PATH, compile=False)
        transport_state_model = keras.models.load_model(TRANSPORT_STATE_MODEL_PATH, compile=False)
        brain_effect_model = keras.models.load_model(BRAIN_EFFECT_MODEL_PATH, compile=False)
    return bbb_model, transport_state_model, brain_effect_model

def predict_metabolite(metabolite_features, models=None):
    """
    Given a dictionary of preprocessed input features for one metabolite, predict its effects.
    The metabolite_features dictionary must contain:
        - "bbb_input": Input vector for the BBB Transport Model (e.g., dimension 34)
        - "transport_state_input": Input vector for the Transport State Model (e.g., dimension 21)
        - "brain_effect_input": Input vector for the Brain Region Effect Model (e.g., dimension 19)
        - "concentration": A scalar representing the metabolite's amount
    Returns a dictionary of predictions from each model.
    """
    if models is None:
        models = load_models()
    bbb_model, transport_state_model, brain_effect_model = models

    # Ensure inputs are numpy arrays with a batch dimension.
    bbb_in = np.array(metabolite_features["bbb_input"]).reshape(1, -1)
    ts_in = np.array(metabolite_features["transport_state_input"]).reshape(1, -1)
    be_in = np.array(metabolite_features["brain_effect_input"]).reshape(1, -1)

    # Get predictions (each model returns a list of outputs)
    bbb_pred = bbb_model.predict(bbb_in)
    ts_pred = transport_state_model.predict(ts_in)
    be_pred = brain_effect_model.predict(be_in)

    # Package predictions into a dictionary.
    prediction = {
        "bbb_prediction": {
            "crossing_probability": bbb_pred[0].tolist(),
            "transport_rate": bbb_pred[1].tolist(),
            "state_distribution": bbb_pred[2].tolist()
        },
        "transport_state_prediction": {
            "time_series": ts_pred[0].tolist(),
            "transition_rates": ts_pred[1].tolist(),
            "steady_state": ts_pred[2].tolist()
        },
        "brain_effect_prediction": {
            "region_concentration": be_pred[0].tolist(),
            "accumulation_rates": be_pred[1].tolist(),
            "interference_patterns": be_pred[2].tolist()
        },
        "concentration": metabolite_features.get("concentration", 1.0)
    }
    return prediction

def run_pipeline(metabolites):
    """
    Given a list of metabolite feature dictionaries, run each through the pipeline.
    Returns a list of prediction dictionaries.
    """
    models = load_models()
    predictions = []
    for met in metabolites:
        pred = predict_metabolite(met, models=models)
        predictions.append(pred)
    return predictions

if __name__ == '__main__':
    # Instead of a random dummy, use a sample metabolite corresponding to Putrescine (HMDB0001414).
    # For demonstration purposes, we simulate preprocessed input vectors.
    # In your actual system, you would process the unified dataset row to generate these inputs.
    
    # For BBB Transport Model: 34-dimensional input vector.
    # Components: one-hot encoded Transport_direction ("in" -> e.g., [0, 0, 1, 0]),
    # binary flags for is_neurotransmitter ("No" → 0) and is_hormone ("No" → 0),
    # multi-hot encoded CellLoc from ["Mitochondria"],
    # multi-hot encoded brain_tissues from ["Brain", "Neuron"],
    # and aggregated brain expression from the provided JSON.
    # (Here we simulate a plausible fixed vector.)
    bbb_input = [0, 0, 1, 0] + [1]*5 + [0]*5 + [0, 0] + [1]*5 + [30.292, 34.929, 32.604, 35.838, 29.342, 30.442, 45.117, 36.883, 35.488, 32.800, 34.117, 36.717, 40.104]
    # This should sum up to 4 + 10 + 2 + 5 + 13 = 34. (Adjust as needed.)
    
    # For Transport State Model: 21-dimensional input vector.
    # (Simulated values.)
    transport_state_input = [0.5]*21
    
    # For Brain Region Effect Model: 19-dimensional input vector.
    # (Simulated values; e.g., aggregated brain expression, binary flags, dummy state, dummy BBB probability)
    brain_effect_input = [30.292, 34.929, 32.604, 35.838, 29.342, 30.442, 45.117, 36.883, 35.488, 32.800, 34.117, 36.717, 40.104, 0, 0, 0.33, 0.33, 0.34, 0.6]
    
    sample_metabolite = {
        "bbb_input": bbb_input,
        "transport_state_input": transport_state_input,
        "brain_effect_input": brain_effect_input,
        "concentration": 1.0
    }
    
    preds = run_pipeline([sample_metabolite])
    print("Pipeline predictions:")
    print(preds)