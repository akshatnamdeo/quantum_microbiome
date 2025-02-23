import os
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, StandardScaler

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Global constant for fixed brain regions (if needed later)
BRAIN_REGIONS = [
    "amygdala", "basal ganglia", "cerebellum", "cerebral cortex", "choroid plexus",
    "hippocampal formation", "hypothalamus", "medulla oblongata", "midbrain", "pons",
    "spinal cord", "thalamus", "white matter"
]

def impute_cellloc(df):
    """
    Impute missing values in the CellLoc column based on grouping by:
    is_neurotransmitter, is_hormone, and brain_tissues.
    """
    df['brain_tissues_filled'] = df['brain_tissues'].fillna('Unknown')
    df['group_key'] = (
        df['is_neurotransmitter'].astype(str) + "_" +
        df['is_hormone'].astype(str) + "_" +
        df['brain_tissues_filled'].astype(str)
    )
    non_missing = df[df['CellLoc'].notnull()]
    group_mode = non_missing.groupby('group_key')['CellLoc'].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
    )
    def impute_value(row):
        if pd.isnull(row['CellLoc']):
            key = row['group_key']
            return group_mode.get(key, 'Unknown')
        else:
            return row['CellLoc']
    df['CellLoc'] = df.apply(impute_value, axis=1)
    return df

def load_and_preprocess_data(csv_path):
    """
    Load the unified dataset and process features for the Transport State Model.
    Input features:
        - Transport_direction (one-hot encoded)
        - CellLoc (imputed and multi-hot encoded)
        - Direction (one-hot encoded)
        - Dummy current state distribution (3 values summing to 1)
    Dummy target outputs:
        - time_series: Time-series of state probabilities (flattened vector of 10 time steps × 3 compartments = 30)
        - transition_rates: Flattened transition rate matrix (3x3 = 9 values)
        - steady_state: Steady-state distribution (3 probabilities)
    """
    df = pd.read_csv(csv_path)
    df = impute_cellloc(df)
    
    # Process Transport_direction using OneHotEncoder
    transport_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    transport_dir = transport_encoder.fit_transform(df[['Transport_direction']])
    
    # Process CellLoc: convert string representation of list to list
    def parse_cellloc(cellloc_str):
        try:
            locs = json.loads(cellloc_str.replace("'", "\""))
            if isinstance(locs, list):
                return locs
            else:
                return [str(locs)]
        except:
            return []
    df['CellLoc_list'] = df['CellLoc'].apply(parse_cellloc)
    mlb_cellloc = MultiLabelBinarizer()
    cellloc_encoded = mlb_cellloc.fit_transform(df['CellLoc_list'])
    
    # Process Direction using OneHotEncoder (e.g., values like "degrading", "producing", "degrading | producing")
    direction_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    direction_encoded = direction_encoder.fit_transform(df[['Direction']])
    
    # Generate a dummy "current state distribution" for each sample (3 compartments)
    # Here, we generate random probabilities that sum to 1.
    rand_states = np.random.rand(df.shape[0], 3)
    current_state = rand_states / rand_states.sum(axis=1, keepdims=True)
    
    # Concatenate input features
    X = np.concatenate([
        transport_dir,         # Encoded Transport_direction
        cellloc_encoded,       # Multi-hot encoded CellLoc
        direction_encoded,     # Encoded Direction
        current_state          # Dummy current state distribution
    ], axis=1)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Generate dummy targets for Transport State Model
    num_samples = df.shape[0]
    # 1. Time-series state probabilities: shape (num_samples, 10, 3) flattened to (num_samples, 30)
    T = 10  # number of time steps
    random_series = np.random.rand(num_samples, T, 3)
    # Normalize each time step's 3 values to sum to 1
    for i in range(num_samples):
        for t in range(T):
            random_series[i, t] /= random_series[i, t].sum() + 1e-8
    y_time_series = random_series.reshape(num_samples, T * 3)
    
    # 2. Transition rates: flattened 3x3 matrix (9 values per sample)
    y_transition = np.random.rand(num_samples, 9)
    
    # 3. Steady-state distribution: vector of 3 probabilities
    random_steady = np.random.rand(num_samples, 3)
    y_steady = random_steady / random_steady.sum(axis=1, keepdims=True)
    
    y = {
        'time_series': y_time_series,
        'transition_rates': y_transition,
        'steady_state': y_steady
    }
    
    return X, y, transport_encoder, mlb_cellloc, direction_encoder, scaler

# --- Define a Quantum-Inspired Dense Layer (reuse from previous model) ---
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

def build_transport_state_model(input_dim):
    """
    Build the Transport State Model with corrected output shapes.
    The model outputs three predictions:
        1. time_series (shape: [batch_size, 30])
        2. transition_rates (shape: [batch_size, 9])
        3. steady_state (shape: [batch_size, 3])
    """
    inputs = layers.Input(shape=(input_dim,))
    
    # Shared base layers
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    
    # Quantum-inspired parallel pathways
    q1 = QuantumDense(32, activation='relu')(x)
    q2 = QuantumDense(32, activation='relu')(x)
    quantum_out = layers.Concatenate()([q1, q2])
    quantum_out = layers.Dense(64, activation='relu')(quantum_out)
    
    # Combine shared representation with quantum pathway output
    shared = layers.Concatenate()([x, quantum_out])
    shared = layers.Dense(64, activation='relu')(shared)
    
    # --- Time Series Output ---
    # Remove the Lambda layer and keep it as a flat output
    time_series_branch = layers.Dense(128, activation='relu')(shared)
    time_series_branch = layers.Dense(64, activation='relu')(time_series_branch)
    time_series = layers.Dense(30, name='time_series')(time_series_branch)
    
    # --- Transition Rates Output ---
    rates_branch = layers.Dense(32, activation='relu')(shared)
    transition_rates = layers.Dense(9, name='transition_rates')(rates_branch)
    
    # --- Steady State Output ---
    steady_branch = layers.Dense(32, activation='relu')(shared)
    steady_state = layers.Dense(3, activation='softmax', name='steady_state')(steady_branch)
    
    model = models.Model(
        inputs=inputs, 
        outputs=[time_series, transition_rates, steady_state]
    )
    
    # Custom loss weights to balance the different tasks
    loss_weights = {
        'time_series': 1.0,
        'transition_rates': 0.5,
        'steady_state': 1.0
    }
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            'time_series': MeanSquaredError(),
            'transition_rates': MeanSquaredError(),
            'steady_state': MeanSquaredError()
        },
        loss_weights=loss_weights,
        metrics={
            'time_series': 'mse',
            'transition_rates': 'mse',
            'steady_state': 'mse'
        }
    )
    return model

def train_model(model, X, y):
    """
    Train the model with proper data handling
    """
    # Split indices for consistent splitting across multiple outputs
    indices = np.arange(X.shape[0])
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    # Split features and all targets
    X_train, X_val = X[train_idx], X[val_idx]
    y_train = {
        'time_series': y['time_series'][train_idx],
        'transition_rates': y['transition_rates'][train_idx],
        'steady_state': y['steady_state'][train_idx]
    }
    y_val = {
        'time_series': y['time_series'][val_idx],
        'transition_rates': y['transition_rates'][val_idx],
        'steady_state': y['steady_state'][val_idx]
    }
    
    # Callbacks for training
    checkpoint = callbacks.ModelCheckpoint(
        'transport_state_model.h5',
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )
    
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    log_callback = callbacks.CSVLogger(
        'transport_state_training_log.csv', 
        separator=',',
        append=False
    )
    
    try:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[checkpoint, early_stop, reduce_lr, log_callback],
            verbose=1
        )
        return history
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        # Log shapes for debugging
        print(f"Training shapes:")
        print(f"X_train: {X_train.shape}")
        for key, val in y_train.items():
            print(f"{key}: {val.shape}")
        raise

def main():
    try:
        csv_path = 'data/unified/unified_dataset_cleaned.csv'
        X, y, transport_enc, cellloc_mlb, direction_enc, scaler = load_and_preprocess_data(csv_path)
        print(f"Input feature shape: {X.shape}")
        print("Target shapes:")
        for key, val in y.items():
            print(f"{key}: {val.shape}")
        
        model = build_transport_state_model(input_dim=X.shape[1])
        model.summary()
        
        history = train_model(model, X, y)
        
        # Plot training history
        if history is not None:
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
            plt.plot(history.history['steady_state_mse'], label='Steady-State MSE')
            plt.plot(history.history['val_steady_state_mse'], label='Val Steady-State MSE')
            plt.title('Steady-State MSE')
            plt.xlabel('Epoch')
            plt.ylabel('MSE')
            plt.legend()
            plt.tight_layout()
            plt.savefig('transport_state_training_history.png')
            plt.close()
        
        print("\nEvaluating model...")
        evaluation = model.evaluate(X, y, verbose=1)
        for metric, value in zip(model.metrics_names, evaluation):
            print(f"{metric}: {value:.4f}")
        
        model.save('final_transport_state_model.h5')
        print("\nModel saved to final_transport_state_model.h5")
        
        preprocessing = {
            'transport_encoder': transport_enc,
            'cellloc_mlb': cellloc_mlb,
            'direction_encoder': direction_enc,
            'scaler': scaler
        }
        with open('transport_state_preprocessing.pkl', 'wb') as f:
            pickle.dump(preprocessing, f)
        print("Preprocessing objects saved to transport_state_preprocessing.pkl")
    
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main()
