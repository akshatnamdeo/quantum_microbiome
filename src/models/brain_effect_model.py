import os
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Global constant for fixed brain regions order (13 regions)
BRAIN_REGIONS = [
    "amygdala", "basal ganglia", "cerebellum", "cerebral cortex", "choroid plexus",
    "hippocampal formation", "hypothalamus", "medulla oblongata", "midbrain", "pons",
    "spinal cord", "thalamus", "white matter"
]

def parse_json_expression(expr_str):
    """
    Parse the aggregated_brain_expression JSON string into a fixed-order vector
    of brain regions.
    """
    try:
        expr_dict = json.loads(expr_str.replace("'", "\""))
    except Exception as e:
        expr_dict = {}
    return [expr_dict.get(region, 0.0) for region in BRAIN_REGIONS]

def load_and_preprocess_data(csv_path):
    """
    Load the unified dataset and process features for the Brain Region Effect Model.
    Input features:
        - aggregated_brain_expression (parsed into a 13-element vector)
        - is_neurotransmitter (binary)
        - is_hormone (binary)
        - Dummy current state distribution (3 values, summing to 1)
        - Dummy BBB crossing probability (1 value between 0 and 1)
    These are concatenated to form a feature vector of length 19.
    Dummy target outputs (for now):
        - region_concentration: vector of 13 values
        - accumulation_rates: vector of 13 values
        - interference_patterns: vector of 13 values (softmax output)
    """
    df = pd.read_csv(csv_path)
    
    # Process aggregated_brain_expression
    expr_array = np.array(df['aggregated_brain_expression'].apply(parse_json_expression).tolist())
    
    # Process binary features for neurotransmitter and hormone status
    df['is_neurotransmitter_bin'] = df['is_neurotransmitter'].map({'Yes': 1, 'No': 0})
    df['is_hormone_bin'] = df['is_hormone'].map({'Yes': 1, 'No': 0})
    binary_features = df[['is_neurotransmitter_bin', 'is_hormone_bin']].values  # shape (n,2)
    
    # Generate a dummy current state distribution (3 compartments; normalized)
    rand_state = np.random.rand(df.shape[0], 3)
    current_state = rand_state / rand_state.sum(axis=1, keepdims=True)
    
    # Generate a dummy BBB crossing probability (one value per sample)
    bbb_prob = np.random.rand(df.shape[0], 1)
    
    # Concatenate all features: expr_array (13) + binary_features (2) + current_state (3) + bbb_prob (1)
    X = np.concatenate([expr_array, binary_features, current_state, bbb_prob], axis=1)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Generate dummy targets:
    num_samples = df.shape[0]
    # 1. Region concentration predictions (vector of 13 values)
    y_concentration = np.random.rand(num_samples, 13)
    # 2. Accumulation rates (vector of 13 values)
    y_accumulation = np.random.rand(num_samples, 13)
    # 3. Interference patterns (vector of 13 values; here we simulate probabilities with softmax later)
    random_interference = np.random.rand(num_samples, 13)
    y_interference = random_interference / random_interference.sum(axis=1, keepdims=True)
    
    y = {
        'region_concentration': y_concentration,
        'accumulation_rates': y_accumulation,
        'interference_patterns': y_interference
    }
    
    return X, y, scaler

# --- Quantum-Inspired Dense Layer (reuse from previous models) ---
class QuantumDense(layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(QuantumDense, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
    
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

def build_brain_effect_model(input_dim):
    """
    Build the Brain Region Effect Model.
    The model outputs three predictions:
        1. region_concentration: Vector of 13 values (one per brain region)
        2. accumulation_rates: Vector of 13 values
        3. interference_patterns: Vector of 13 values (softmax activation)
    """
    inputs = layers.Input(shape=(input_dim,))
    
    # Shared layers
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    
    # Quantum-inspired branch
    q = QuantumDense(32, activation='relu')(x)
    shared = layers.Concatenate()([x, q])
    shared = layers.Dense(64, activation='relu')(shared)
    
    # --- Branch for Region Concentration ---
    branch1 = layers.Dense(32, activation='relu')(shared)
    concentration_output = layers.Dense(13, activation='linear', name='region_concentration')(branch1)
    
    # --- Branch for Accumulation Rates ---
    branch2 = layers.Dense(32, activation='relu')(shared)
    accumulation_output = layers.Dense(13, activation='linear', name='accumulation_rates')(branch2)
    
    # --- Branch for Interference Patterns ---
    branch3 = layers.Dense(32, activation='relu')(shared)
    # Use softmax to simulate a probability distribution over brain regions
    interference_output = layers.Dense(13, activation='softmax', name='interference_patterns')(branch3)
    
    model = models.Model(inputs=inputs, outputs=[concentration_output, accumulation_output, interference_output])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            'region_concentration': 'mse',
            'accumulation_rates': 'mse',
            'interference_patterns': 'mse'
        },
        metrics={
            'region_concentration': 'mse',
            'accumulation_rates': 'mse',
            'interference_patterns': 'mse'
        }
    )
    return model

def train_model(model, X, y):
    """
    Train the Brain Region Effect Model using a training/validation split and callbacks.
    """
    indices = np.arange(X.shape[0])
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    X_train = X[train_idx]
    X_val = X[val_idx]
    y_train = {
        'region_concentration': y['region_concentration'][train_idx],
        'accumulation_rates': y['accumulation_rates'][train_idx],
        'interference_patterns': y['interference_patterns'][train_idx]
    }
    y_val = {
        'region_concentration': y['region_concentration'][val_idx],
        'accumulation_rates': y['accumulation_rates'][val_idx],
        'interference_patterns': y['interference_patterns'][val_idx]
    }
    
    log_callback = callbacks.CSVLogger('brain_effect_training_log.csv', separator=',', append=False)
    checkpoint = callbacks.ModelCheckpoint('brain_effect_model.h5', save_best_only=True, monitor='val_loss', verbose=1)
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[checkpoint, early_stop, reduce_lr, log_callback],
        verbose=1
    )
    return history

def main():
    try:
        csv_path = 'data/unified/unified_dataset_cleaned.csv'
        X, y, scaler = load_and_preprocess_data(csv_path)
        print(f"Input feature shape: {X.shape}")
        print("Target shapes:")
        for key, val in y.items():
            print(f"{key}: {val.shape}")
        
        model = build_brain_effect_model(input_dim=X.shape[1])
        model.summary()
        
        history = train_model(model, X, y)
        
        # Plot training history
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Overall Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history['interference_patterns_mse'], label='Interference MSE')
        plt.plot(history.history['val_interference_patterns_mse'], label='Val Interference MSE')
        plt.title('Interference Patterns MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        plt.tight_layout()
        plt.savefig('brain_effect_training_history.png')
        plt.close()
        
        print("\nEvaluating model...")
        evaluation = model.evaluate(X, y, verbose=1)
        for metric, value in zip(model.metrics_names, evaluation):
            print(f"{metric}: {value:.4f}")
        
        model.save('final_brain_effect_model.h5')
        print("\nModel saved to final_brain_effect_model.h5")
        
        # Save the preprocessing scaler
        with open('brain_effect_preprocessing.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        print("Preprocessing objects saved to brain_effect_preprocessing.pkl")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main()
