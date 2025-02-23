import os
import json
import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks  # Updated imports
from keras.optimizers import Adam  # Updated optimizer import
from keras.losses import BinaryCrossentropy, MeanSquaredError  # Updated loss imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, StandardScaler

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Global constant for fixed brain regions order (for aggregated expression features)
BRAIN_REGIONS = [
	"amygdala", "basal ganglia", "cerebellum", "cerebral cortex", "choroid plexus",
	"hippocampal formation", "hypothalamus", "medulla oblongata", "midbrain", "pons",
	"spinal cord", "thalamus", "white matter"
]

def parse_json_expression(expr_str):
	"""
	Parse the aggregated_brain_expression JSON string into a vector
	with fixed ordering of brain regions.
	"""
	try:
		expr_dict = json.loads(expr_str.replace("'", "\""))
	except Exception as e:
		expr_dict = {}
	return [expr_dict.get(region, 0.0) for region in BRAIN_REGIONS]

def parse_state_distribution(state_str):
	"""
	Parse the state_distribution JSON string into a fixed-order vector.
	Expected keys: "blood", "bbb_interface", "brain_tissue".
	"""
	try:
		state_dict = json.loads(state_str.replace("'", "\""))
	except Exception as e:
		state_dict = {}
	keys = ["blood", "bbb_interface", "brain_tissue"]
	return [state_dict.get(k, 0.0) for k in keys]

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
	Load the unified dataset and process features for the BBB Transport Model.
	Features used:
	    - Transport_direction (one-hot encoded)
	    - CellLoc (imputed and multi-hot encoded)
	    - is_neurotransmitter and is_hormone (binary)
	    - brain_tissues (multi-hot encoded)
	    - aggregated_brain_expression (vectorized)
	Targets (dummy outputs):
	    - crossing_probability (binary, probability)
	    - transport_rate (regression)
	    - state_distribution (probabilities over compartments)
	"""
	df = pd.read_csv(csv_path)
	# Impute missing CellLoc values
	df = impute_cellloc(df)
	
	# --- Process Transport_direction ---
	transport_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
	transport_dir = transport_encoder.fit_transform(df[['Transport_direction']])
	
	# --- Process CellLoc ---
	# Convert CellLoc (string representation of list) to an actual list
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
	
	# --- Process is_neurotransmitter and is_hormone ---
	df['is_neurotransmitter_bin'] = df['is_neurotransmitter'].map({'Yes': 1, 'No': 0})
	df['is_hormone_bin'] = df['is_hormone'].map({'Yes': 1, 'No': 0})
	binary_features = df[['is_neurotransmitter_bin', 'is_hormone_bin']].values
	
	# --- Process brain_tissues ---
	# Assume brain_tissues are separated by "|" if there are multiple entries
	def parse_brain_tissues(bt_str):
		if pd.isnull(bt_str):
			return ['Unknown']
		return [s.strip() for s in bt_str.split('|')]
	df['brain_tissues_list'] = df['brain_tissues'].apply(parse_brain_tissues)
	mlb_brain_tissues = MultiLabelBinarizer()
	brain_tissues_encoded = mlb_brain_tissues.fit_transform(df['brain_tissues_list'])
	
	# --- Process aggregated_brain_expression ---
	expr_array = np.array(df['aggregated_brain_expression'].apply(parse_json_expression).tolist())
	
	# Concatenate all feature vectors into a final feature matrix
	X = np.concatenate([
		transport_dir,         # Encoded Transport_direction
		cellloc_encoded,       # Multi-hot encoded CellLoc
		binary_features,       # is_neurotransmitter and is_hormone binary features
		brain_tissues_encoded, # Multi-hot encoded brain_tissues
		expr_array             # Aggregated brain expression vector
	], axis=1)
	
	# Scale the final feature matrix
	scaler = StandardScaler()
	X = scaler.fit_transform(X)
	
	# --- Prepare target outputs ---
	# Since the model is designed to simulate probabilistic outputs, we generate dummy targets.
	# In production, these can be replaced by simulation outputs or derived from domain-specific heuristics.
	y_cross = np.random.randint(0, 2, size=df.shape[0])
	y_rate = np.random.rand(df.shape[0])
	random_states = np.random.rand(df.shape[0], 3)
	y_state = random_states / random_states.sum(axis=1, keepdims=True)
	
	y = {
		'crossing_probability': y_cross,
		'transport_rate': y_rate,
		'state_distribution': y_state
	}
	
	return X, y, transport_encoder, mlb_cellloc, mlb_brain_tissues, scaler

# --- Define Custom Quantum-Inspired Dense Layer ---
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
		# Compute magnitude to simulate interference effects
		magnitude = tf.sqrt(tf.square(real_part) + tf.square(imag_part)) + self.bias
		if self.activation is not None:
			return self.activation(magnitude)
		return magnitude

def build_bbb_transport_model(input_dim):
    """
    Build the quantum-inspired BBB Transport Model.
    The network processes input features and branches into three outputs:
        1. bbb_crossing (sigmoid for probability)
        2. transport_rate (linear activation for regression)
        3. state_distribution (softmax over 3 compartments)
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
    
    # --- Output Heads ---
    # 1. BBB Crossing Probability
    crossing_branch = layers.Dense(32, activation='relu')(shared)
    crossing_output = layers.Dense(1, activation='sigmoid', name='crossing_probability')(crossing_branch)
    
    # 2. Transport Rate
    rate_branch = layers.Dense(32, activation='relu')(shared)
    rate_output = layers.Dense(1, activation='linear', name='transport_rate')(rate_branch)
    
    # 3. State Distribution (probabilities; softmax ensures sum=1)
    state_branch = layers.Dense(32, activation='relu')(shared)
    state_output = layers.Dense(3, activation='softmax', name='state_distribution')(state_branch)
    
    model = models.Model(inputs=inputs, outputs=[crossing_output, rate_output, state_output])
    
    # Compile the model with combined loss functions using updated imports
    model.compile(
        optimizer=Adam(learning_rate=0.001),  # Updated optimizer
        loss={
            'crossing_probability': BinaryCrossentropy(),  # Updated loss
            'transport_rate': MeanSquaredError(),         # Updated loss
            'state_distribution': MeanSquaredError()      # Updated loss
        },
        metrics={
            'crossing_probability': 'accuracy',
            'transport_rate': 'mse',
            'state_distribution': 'mse'
        }
    )
    return model

def train_model(model, X, y):
    """
    Train the model with a training/validation split and callbacks.
    Handles dictionary target variables correctly.
    
    Args:
        model: Compiled tensorflow model
        X: Input features array
        y: Dictionary of target variables
        
    Returns:
        history: Training history
    """
    # Split indices instead of data directly
    indices = np.arange(X.shape[0])
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    # Split X data
    X_train = X[train_idx]
    X_val = X[val_idx]
    
    # Split y data - maintain dictionary structure
    y_train = {
        'crossing_probability': y['crossing_probability'][train_idx],
        'transport_rate': y['transport_rate'][train_idx],
        'state_distribution': y['state_distribution'][train_idx]
    }
    
    y_val = {
        'crossing_probability': y['crossing_probability'][val_idx],
        'transport_rate': y['transport_rate'][val_idx],
        'state_distribution': y['state_distribution'][val_idx]
    }
    
    # Add logging callback
    log_callback = callbacks.CSVLogger('training_log.csv', separator=',', append=False)
    
    # Callbacks for checkpointing and early stopping
    checkpoint = callbacks.ModelCheckpoint(
        'bbb_transport_model.h5',
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
    
    # Add learning rate reduction on plateau
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
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
        X, y, transport_enc, cellloc_mlb, brain_tissues_mlb, scaler = load_and_preprocess_data(csv_path)
        print(f"Input feature shape: {X.shape}")
        print("Target shapes:")
        for key, val in y.items():
            print(f"{key}: {val.shape}")
        
        model = build_bbb_transport_model(input_dim=X.shape[1])
        model.summary()
        
        history = train_model(model, X, y)
        
        # Plot training history
        if history is not None:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['crossing_probability_accuracy'], 
                    label='Crossing Probability Accuracy')
            plt.plot(history.history['val_crossing_probability_accuracy'], 
                    label='Val Crossing Probability Accuracy')
            plt.title('Crossing Probability Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('training_history.png')
            plt.close()
        
        # Evaluate the model
        print("\nEvaluating model...")
        evaluation = model.evaluate(X, y, verbose=1)
        for metric, value in zip(model.metrics_names, evaluation):
            print(f"{metric}: {value:.4f}")
        
        # Save the final model
        model.save('final_bbb_transport_model.h5')
        print("\nModel saved to final_bbb_transport_model.h5")
        
        # Save the preprocessing objects
        preprocessing = {
            'transport_encoder': transport_enc,
            'cellloc_mlb': cellloc_mlb,
            'brain_tissues_mlb': brain_tissues_mlb,
            'scaler': scaler
        }
        with open('preprocessing_objects.pkl', 'wb') as f:
            pickle.dump(preprocessing, f)
        print("Preprocessing objects saved to preprocessing_objects.pkl")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
	main()
