import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. DATA GENERATION WITH REALISTIC PATTERNS
# =============================================================================

def generate_realistic_power_data(n_samples=10000, n_test_runs=10):
    """
    Generate synthetic power data with realistic patterns for each state
    """
    data = []
    samples_per_run = n_samples // n_test_runs
    
    for run_id in range(n_test_runs):
        run_data = []
        
        # Each run has different state transitions
        for i in range(samples_per_run):
            # Determine state based on position in run
            if i < samples_per_run * 0.3:
                state = 'stabilizing'
                # Stabilizing: oscillations that gradually decrease
                t = i / 100
                voltage = 400 + 20 * np.exp(-t/10) * np.sin(t*5) + np.random.randn() * 2
                current = 50 + 5 * np.exp(-t/10) * np.cos(t*5) + np.random.randn() * 0.5
            elif i < samples_per_run * 0.4:
                state = 'de-energized'
                # De-energized: sharp drop to near zero
                decay_factor = max(0, 1 - (i - samples_per_run * 0.3) / (samples_per_run * 0.05))
                voltage = 400 * decay_factor + np.random.randn() * 0.1
                current = 50 * decay_factor + np.random.randn() * 0.05
            else:
                state = 'steady_state'
                # Steady state: stable values with low noise
                voltage = 400 + np.random.randn() * 0.5
                current = 50 + np.random.randn() * 0.1
            
            run_data.append({
                'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(milliseconds=i),
                'dc1_voltage': voltage,
                'dc1_current': current,
                'status': state,
                'test_run': run_id
            })
        
        data.extend(run_data)
    
    return pd.DataFrame(data)

# =============================================================================
# 2. ENHANCED DATA PROCESSOR WITH DUAL PATHS
# =============================================================================

class PowerDataProcessor:
    """Handles data windowing with options for both feature engineering and raw sequences"""
    
    def __init__(self, window_size=100, stride=10):
        """
        Args:
            window_size: Number of milliseconds in each window
            stride: Step size for sliding window
        """
        self.window_size = window_size
        self.stride = stride
        self.scaler = StandardScaler()
        self.feature_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def create_windows(self, df, group_col='test_run', engineer_features=True):
        """
        Create sliding windows from time series data
        
        Args:
            df: DataFrame with columns [timestamp, dc1_voltage, dc1_current, status, test_run]
            group_col: Column to group by
            engineer_features: If True, return engineered features; if False, return raw sequences
        """
        windows = []
        labels = []
        groups = []
        
        for group_name, group_df in df.groupby(group_col):
            group_df = group_df.sort_values('timestamp').reset_index(drop=True)
            
            # Create windows for this test run
            for i in range(0, len(group_df) - self.window_size + 1, self.stride):
                window_data = group_df.iloc[i:i + self.window_size]
                
                # Extract raw signals
                voltage = window_data['dc1_voltage'].values
                current = window_data['dc1_current'].values
                
                if engineer_features:
                    # Path A: Engineer features for traditional ML
                    features = self._engineer_features(voltage, current)
                    windows.append(features)
                else:
                    # Path B: Keep raw sequences for deep learning
                    raw_sequence = np.stack([voltage, current], axis=1)
                    windows.append(raw_sequence)
                
                # Use the label at the end of the window
                label = window_data['status'].iloc[-1]
                labels.append(label)
                groups.append(group_name)
        
        return np.array(windows), np.array(labels), np.array(groups)
    
    def _engineer_features(self, voltage, current):
        """
        Engineer features from a window of voltage and current data
        """
        # Calculate power
        power = voltage * current
        
        features = []
        
        for signal, name in [(voltage, 'v'), (current, 'i'), (power, 'p')]:
            # Basic statistics
            features.extend([
                np.mean(signal),
                np.std(signal),
                np.min(signal),
                np.max(signal),
                np.median(signal),
                np.percentile(signal, 25),
                np.percentile(signal, 75),
                signal[-1] - signal[0],  # Delta
            ])
            
            # Rate of change features
            diff = np.diff(signal)
            features.extend([
                np.mean(diff),
                np.std(diff),
                np.max(np.abs(diff)),
            ])
            
            # Trend features
            x = np.arange(len(signal))
            slope, intercept = np.polyfit(x, signal, 1)
            features.append(slope)
            
            # Additional time-series features
            features.extend([
                np.sum(np.abs(diff)),  # Total variation
                len(np.where(np.diff(np.sign(diff)))[0]),  # Number of peaks
            ])
        
        # Cross-signal features
        features.append(np.corrcoef(voltage, current)[0, 1])
        
        return np.array(features)
    
    def prepare_data(self, X, y, groups, test_runs_for_test, is_sequence=False):
        """
        Split data by test runs and scale features
        """
        # Create masks for train and test
        test_mask = np.isin(groups, test_runs_for_test)
        train_mask = ~test_mask
        
        # Split data
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        # Scale features
        if is_sequence:
            # For sequences, scale along the feature dimension
            original_shape = X_train.shape
            X_train_flat = X_train.reshape(-1, X_train.shape[-1])
            X_test_flat = X_test.reshape(-1, X_test.shape[-1])
            
            X_train_flat = self.scaler.fit_transform(X_train_flat)
            X_test_flat = self.scaler.transform(X_test_flat)
            
            X_train = X_train_flat.reshape(original_shape)
            X_test = X_test_flat.reshape(X_test.shape)
        else:
            # For engineered features, standard scaling
            X_train = self.feature_scaler.fit_transform(X_train)
            X_test = self.feature_scaler.transform(X_test)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        return X_train, X_test, y_train_encoded, y_test_encoded, y_train, y_test

# =============================================================================
# 3. PATH A: TRADITIONAL ML MODELS FOR ENGINEERED FEATURES
# =============================================================================

class TraditionalMLModels:
    """Traditional ML models for engineered features"""
    
    def __init__(self, n_classes):
        self.n_classes = n_classes
        
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model"""
        print("\nTraining XGBoost...")
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective='multi:softprob',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return model, y_pred, accuracy
    
    def train_lightgbm(self, X_train, y_train, X_test, y_test):
        """Train LightGBM model"""
        print("\nTraining LightGBM...")
        
        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
        )
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return model, y_pred, accuracy
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model"""
        print("\nTraining Random Forest...")
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return model, y_pred, accuracy
    
    def build_mlp(self, n_features):
        """Build Multi-Layer Perceptron for engineered features"""
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(n_features,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

# =============================================================================
# 4. PATH B: SEQUENCE MODELS FOR RAW TIME SERIES
# =============================================================================

class SequenceModels:
    """Deep learning models for raw sequence data"""
    
    def __init__(self, window_size, n_features, n_classes):
        self.window_size = window_size
        self.n_features = n_features  # Number of raw features (e.g., 2 for voltage and current)
        self.n_classes = n_classes
        
    def build_lstm_model(self):
        """Build LSTM model for actual sequences"""
        model = models.Sequential([
            # LSTM expects (batch, timesteps, features)
            layers.LSTM(128, return_sequences=True, 
                       input_shape=(self.window_size, self.n_features)),
            layers.Dropout(0.2),
            layers.LSTM(64, return_sequences=False),
            layers.Dropout(0.2),
            
            # Dense layers
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_cnn1d_model(self):
        """Build 1D CNN model for sequences"""
        model = models.Sequential([
            # 1D Conv expects (batch, timesteps, features)
            layers.Conv1D(64, kernel_size=5, activation='relu', 
                         input_shape=(self.window_size, self.n_features)),
            layers.MaxPooling1D(2),
            layers.Conv1D(128, kernel_size=3, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, kernel_size=3, activation='relu'),
            
            # Global pooling and dense layers
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_cnn_lstm_model(self):
        """Build CNN-LSTM hybrid for sequences"""
        model = models.Sequential([
            # CNN for feature extraction
            layers.Conv1D(64, kernel_size=3, activation='relu',
                         input_shape=(self.window_size, self.n_features)),
            layers.Conv1D(64, kernel_size=3, activation='relu'),
            layers.MaxPooling1D(2),
            
            # LSTM for sequence modeling
            layers.LSTM(50, return_sequences=False),
            layers.Dropout(0.2),
            
            # Dense layers
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_transformer_model(self):
        """Build Transformer model for sequences"""
        inputs = layers.Input(shape=(self.window_size, self.n_features))
        
        # Linear projection to embedding dimension
        d_model = 64
        x = layers.Dense(d_model)(inputs)
        
        # Positional encoding (simplified)
        positions = tf.range(start=0, limit=self.window_size, delta=1)
        positions = tf.expand_dims(positions, 0)
        positions = tf.cast(positions, tf.float32)
        position_embedding = layers.Embedding(self.window_size, d_model)(positions)
        x = x + position_embedding
        
        # Transformer block
        attention_output = layers.MultiHeadAttention(
            num_heads=4, key_dim=d_model
        )(x, x)
        x1 = layers.Dropout(0.1)(attention_output)
        x1 = layers.LayerNormalization(epsilon=1e-6)(x + x1)
        
        # Feed forward network
        ff = layers.Dense(128, activation='relu')(x1)
        ff = layers.Dense(d_model)(ff)
        x2 = layers.Dropout(0.1)(ff)
        x2 = layers.LayerNormalization(epsilon=1e-6)(x1 + x2)
        
        # Global pooling and output
        x = layers.GlobalAveragePooling1D()(x2)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.n_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

# =============================================================================
# 5. TRAINING AND EVALUATION UTILITIES
# =============================================================================

def train_neural_network(model, X_train, y_train, X_test, y_test, 
                        model_name="Model", epochs=50, batch_size=32):
    """Train and evaluate a neural network model"""
    print(f"\nTraining {model_name}...")
    
    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=0
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print(f"   {model_name} Test Accuracy: {test_accuracy:.4f}")
    
    return history, y_pred_classes, test_accuracy

# =============================================================================
# 6. MAIN EXECUTION SCRIPT
# =============================================================================

def main():
    """
    Main execution function comparing both paths
    """
    print("=" * 80)
    print("POWER DATA CLASSIFICATION: COMPARING FEATURE ENGINEERING VS RAW SEQUENCES")
    print("=" * 80)
    
    # Generate realistic synthetic data
    print("\n1. Generating realistic power data...")
    df = generate_realistic_power_data(n_samples=20000, n_test_runs=10)
    print(f"   Generated {len(df)} samples from {df['test_run'].nunique()} test runs")
    print(f"   Status distribution:")
    for status, count in df['status'].value_counts().items():
        print(f"      {status}: {count} ({count/len(df)*100:.1f}%)")
    
    # Initialize processor
    processor = PowerDataProcessor(window_size=100, stride=20)
    
    # Define test runs (30% for testing)
    unique_runs = df['test_run'].unique()
    n_test_runs = max(1, int(len(unique_runs) * 0.3))
    test_runs = np.random.choice(unique_runs, n_test_runs, replace=False)
    print(f"\n   Using test runs {sorted(test_runs)} for testing")
    
    results = {}
    
    # =========================================================================
    # PATH A: ENGINEERED FEATURES WITH TRADITIONAL ML
    # =========================================================================
    print("\n" + "=" * 80)
    print("PATH A: ENGINEERED FEATURES WITH TRADITIONAL ML MODELS")
    print("=" * 80)
    
    # Create windows with feature engineering
    print("\n2. Creating windows with engineered features...")
    X_features, y, groups = processor.create_windows(df, engineer_features=True)
    print(f"   Created {len(X_features)} windows with {X_features.shape[1]} engineered features")
    
    # Prepare data
    X_train_feat, X_test_feat, y_train, y_test, y_train_orig, y_test_orig = processor.prepare_data(
        X_features, y, groups, test_runs, is_sequence=False
    )
    print(f"   Training samples: {len(X_train_feat)}")
    print(f"   Testing samples: {len(X_test_feat)}")
    
    # Get number of classes
    n_classes = len(np.unique(y_train))
    
    # Train traditional ML models
    print("\n3. Training traditional ML models on engineered features...")
    
    trad_ml = TraditionalMLModels(n_classes)
    
    # XGBoost
    xgb_model, xgb_pred, xgb_acc = trad_ml.train_xgboost(
        X_train_feat, y_train, X_test_feat, y_test
    )
    results['XGBoost (Features)'] = xgb_acc
    print(f"   XGBoost Accuracy: {xgb_acc:.4f}")
    
    # LightGBM
    lgb_model, lgb_pred, lgb_acc = trad_ml.train_lightgbm(
        X_train_feat, y_train, X_test_feat, y_test
    )
    results['LightGBM (Features)'] = lgb_acc
    print(f"   LightGBM Accuracy: {lgb_acc:.4f}")
    
    # Random Forest
    rf_model, rf_pred, rf_acc = trad_ml.train_random_forest(
        X_train_feat, y_train, X_test_feat, y_test
    )
    results['Random Forest (Features)'] = rf_acc
    print(f"   Random Forest Accuracy: {rf_acc:.4f}")
    
    # MLP on features
    mlp_model = trad_ml.build_mlp(X_train_feat.shape[1])
    mlp_hist, mlp_pred, mlp_acc = train_neural_network(
        mlp_model, X_train_feat, y_train, X_test_feat, y_test,
        model_name="MLP", epochs=50
    )
    results['MLP (Features)'] = mlp_acc
    
    # =========================================================================
    # PATH B: RAW SEQUENCES WITH DEEP LEARNING
    # =========================================================================
    print("\n" + "=" * 80)
    print("PATH B: RAW SEQUENCES WITH DEEP LEARNING MODELS")
    print("=" * 80)
    
    # Create windows with raw sequences
    print("\n4. Creating windows with raw sequences...")
    X_sequences, y_seq, groups_seq = processor.create_windows(df, engineer_features=False)
    print(f"   Created {len(X_sequences)} sequences of shape {X_sequences.shape[1:]} (timesteps, features)")
    
    # Prepare sequence data
    X_train_seq, X_test_seq, y_train_seq, y_test_seq, _, _ = processor.prepare_data(
        X_sequences, y_seq, groups_seq, test_runs, is_sequence=True
    )
    print(f"   Training sequences: {X_train_seq.shape}")
    print(f"   Testing sequences: {X_test_seq.shape}")
    
    # Initialize sequence model builder
    seq_models = SequenceModels(
        window_size=X_train_seq.shape[1],
        n_features=X_train_seq.shape[2],
        n_classes=n_classes
    )
    
    print("\n5. Training deep learning models on raw sequences...")
    
    # LSTM
    lstm_model = seq_models.build_lstm_model()
    lstm_hist, lstm_pred, lstm_acc = train_neural_network(
        lstm_model, X_train_seq, y_train_seq, X_test_seq, y_test_seq,
        model_name="LSTM", epochs=50
    )
    results['LSTM (Sequences)'] = lstm_acc
    
    # 1D CNN
    cnn_model = seq_models.build_cnn1d_model()
    cnn_hist, cnn_pred, cnn_acc = train_neural_network(
        cnn_model, X_train_seq, y_train_seq, X_test_seq, y_test_seq,
        model_name="1D-CNN", epochs=50
    )
    results['1D-CNN (Sequences)'] = cnn_acc
    
    # CNN-LSTM
    cnn_lstm_model = seq_models.build_cnn_lstm_model()
    cnn_lstm_hist, cnn_lstm_pred, cnn_lstm_acc = train_neural_network(
        cnn_lstm_model, X_train_seq, y_train_seq, X_test_seq, y_test_seq,
        model_name="CNN-LSTM", epochs=50
    )
    results['CNN-LSTM (Sequences)'] = cnn_lstm_acc
    
    # Transformer
    transformer_model = seq_models.build_transformer_model()
    trans_hist, trans_pred, trans_acc = train_neural_network(
        transformer_model, X_train_seq, y_train_seq, X_test_seq, y_test_seq,
        model_name="Transformer", epochs=50
    )
    results['Transformer (Sequences)'] = trans_acc
    
    # =========================================================================
    # FINAL COMPARISON
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL RESULTS COMPARISON")
    print("=" * 80)
    
    print("\nPath A - Engineered Features:")
    for model_name, accuracy in results.items():
        if "Features" in model_name:
            print(f"   {model_name:30} {accuracy:.4f}")
    
    print("\nPath B - Raw Sequences:")
    for model_name, accuracy in results.items():
        if "Sequences" in model_name:
            print(f"   {model_name:30} {accuracy:.4f}")
    
    best_model = max(results, key=results.get)
    print(f"\n🏆 Best Overall Model: {best_model}")
    print(f"   Accuracy: {results[best_model]:.4f}")
    
    # Get best model predictions for detailed report
    if "XGBoost" in best_model:
        best_pred = xgb_pred
    elif "LightGBM" in best_model:
        best_pred = lgb_pred
    elif "Random Forest" in best_model:
        best_pred = rf_pred
    elif "MLP" in best_model:
        best_pred = mlp_pred
    elif "LSTM" in best_model and "CNN" not in best_model:
        best_pred = lstm_pred
    elif "1D-CNN" in best_model:
        best_pred = cnn_pred
    elif "CNN-LSTM" in best_model:
        best_pred = cnn_lstm_pred
    else:
        best_pred = trans_pred
    
    # Use the appropriate test labels
    if "Features" in best_model:
        test_labels = y_test
    else:
        test_labels = y_test_seq
    
    print(f"\nDetailed Classification Report for {best_model}:")
    print(classification_report(test_labels, best_pred, 
                              target_names=processor.label_encoder.classes_))
    
    # Feature importance for tree-based models
    if "XGBoost" in best_model or "LightGBM" in best_model or "Random Forest" in best_model:
        print("\nTop 10 Most Important Features:")
        if "XGBoost" in best_model:
            importance = xgb_model.feature_importances_
        elif "LightGBM" in best_model:
            importance = lgb_model.feature_importances_
        else:
            importance = rf_model.feature_importances_
        
        indices = np.argsort(importance)[::-1][:10]
        for i, idx in enumerate(indices):
            print(f"   {i+1}. Feature {idx}: {importance[idx]:.4f}")
    
    return results

if __name__ == "__main__":
    results = main()
