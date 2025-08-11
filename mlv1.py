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
# 1. DATA LOADING AND PREPROCESSING
# =============================================================================

def load_power_data(filepath):
    """
    Load power data from parquet file
    
    Expected columns:
    - timestamp or similar time column
    - dc1_voltage, dc1_current, dc1_status
    - dc2_voltage, dc2_current, dc2_status
    - test_run or similar grouping column
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_parquet(filepath)
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for required columns
    required_cols = ['dc1_voltage', 'dc1_current', 'dc1_status', 
                     'dc2_voltage', 'dc2_current', 'dc2_status']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}")
        print("Available columns:", df.columns.tolist())
    
    # Display data info
    print("\nData Overview:")
    print(f"Shape: {df.shape}")
    if 'dc1_status' in df.columns:
        print(f"DC1 Status distribution: {df['dc1_status'].value_counts().to_dict()}")
    if 'dc2_status' in df.columns:
        print(f"DC2 Status distribution: {df['dc2_status'].value_counts().to_dict()}")
    
    return df

# =============================================================================
# 2. DUAL DC SYSTEM DATA PROCESSOR
# =============================================================================

class DualDCPowerDataProcessor:
    """Handles data windowing for dual DC systems with separate processing"""
    
    def __init__(self, window_size=100, stride=10):
        """
        Args:
            window_size: Number of milliseconds/samples in each window
            stride: Step size for sliding window
        """
        self.window_size = window_size
        self.stride = stride
        
        # Separate scalers and encoders for each DC system
        self.dc1_scaler = StandardScaler()
        self.dc1_feature_scaler = StandardScaler()
        self.dc1_label_encoder = LabelEncoder()
        
        self.dc2_scaler = StandardScaler()
        self.dc2_feature_scaler = StandardScaler()
        self.dc2_label_encoder = LabelEncoder()
        
    def create_windows_for_dc(self, df, dc_num, group_col='test_run', engineer_features=True):
        """
        Create sliding windows for a specific DC system
        
        Args:
            df: DataFrame with power data
            dc_num: 1 or 2 for DC1 or DC2
            group_col: Column to group by (e.g., 'test_run')
            engineer_features: If True, return engineered features; if False, return raw sequences
        """
        voltage_col = f'dc{dc_num}_voltage'
        current_col = f'dc{dc_num}_current'
        status_col = f'dc{dc_num}_status'
        
        # Check if columns exist
        if not all(col in df.columns for col in [voltage_col, current_col, status_col]):
            print(f"Warning: DC{dc_num} columns not found in dataframe")
            return None, None, None
        
        windows = []
        labels = []
        groups = []
        
        # If no group column specified, treat all data as one group
        if group_col not in df.columns:
            print(f"Warning: '{group_col}' not found, treating all data as single group")
            df['temp_group'] = 0
            group_col = 'temp_group'
        
        for group_name, group_df in df.groupby(group_col):
            # Sort by timestamp if available
            if 'timestamp' in group_df.columns:
                group_df = group_df.sort_values('timestamp')
            group_df = group_df.reset_index(drop=True)
            
            # Create windows for this group
            for i in range(0, len(group_df) - self.window_size + 1, self.stride):
                window_data = group_df.iloc[i:i + self.window_size]
                
                # Extract raw signals
                voltage = window_data[voltage_col].values
                current = window_data[current_col].values
                
                # Skip windows with NaN values
                if np.any(np.isnan(voltage)) or np.any(np.isnan(current)):
                    continue
                
                if engineer_features:
                    # Path A: Engineer features for traditional ML
                    features = self._engineer_features(voltage, current)
                    windows.append(features)
                else:
                    # Path B: Keep raw sequences for deep learning
                    raw_sequence = np.stack([voltage, current], axis=1)
                    windows.append(raw_sequence)
                
                # Use the label at the end of the window
                label = window_data[status_col].iloc[-1]
                labels.append(label)
                groups.append(group_name)
        
        if len(windows) == 0:
            print(f"Warning: No valid windows created for DC{dc_num}")
            return None, None, None
            
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
            if len(diff) > 0:
                features.extend([
                    np.mean(diff),
                    np.std(diff),
                    np.max(np.abs(diff)),
                ])
            else:
                features.extend([0, 0, 0])
            
            # Trend features
            if len(signal) > 1:
                x = np.arange(len(signal))
                slope, intercept = np.polyfit(x, signal, 1)
                features.append(slope)
            else:
                features.append(0)
            
            # Additional time-series features
            if len(diff) > 0:
                features.extend([
                    np.sum(np.abs(diff)),  # Total variation
                    len(np.where(np.diff(np.sign(diff)))[0]),  # Number of peaks
                ])
            else:
                features.extend([0, 0])
        
        # Cross-signal features
        if len(voltage) > 1 and len(current) > 1:
            corr = np.corrcoef(voltage, current)[0, 1]
            features.append(corr if not np.isnan(corr) else 0)
        else:
            features.append(0)
        
        return np.array(features)
    
    def prepare_data_for_dc(self, X, y, groups, test_runs_for_test, dc_num, is_sequence=False):
        """
        Split data by test runs and scale features for a specific DC system
        """
        if X is None or len(X) == 0:
            return None, None, None, None, None, None
            
        # Select appropriate scaler and encoder based on DC number
        if dc_num == 1:
            scaler = self.dc1_scaler if is_sequence else self.dc1_feature_scaler
            label_encoder = self.dc1_label_encoder
        else:
            scaler = self.dc2_scaler if is_sequence else self.dc2_feature_scaler
            label_encoder = self.dc2_label_encoder
        
        # Create masks for train and test
        test_mask = np.isin(groups, test_runs_for_test)
        train_mask = ~test_mask
        
        # Split data
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        if len(X_train) == 0 or len(X_test) == 0:
            print(f"Warning: Empty train or test set for DC{dc_num}")
            return None, None, None, None, None, None
        
        # Scale features
        if is_sequence:
            # For sequences, scale along the feature dimension
            original_shape = X_train.shape
            X_train_flat = X_train.reshape(-1, X_train.shape[-1])
            X_test_flat = X_test.reshape(-1, X_test.shape[-1])
            
            X_train_flat = scaler.fit_transform(X_train_flat)
            X_test_flat = scaler.transform(X_test_flat)
            
            X_train = X_train_flat.reshape(original_shape)
            X_test = X_test_flat.reshape(X_test.shape)
        else:
            # For engineered features, standard scaling
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        # Encode labels
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
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
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective='multi:softprob' if self.n_classes > 2 else 'binary:logistic',
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
        self.n_features = n_features
        self.n_classes = n_classes
        
    def build_lstm_model(self):
        """Build LSTM model for actual sequences"""
        model = models.Sequential([
            layers.LSTM(128, return_sequences=True, 
                       input_shape=(self.window_size, self.n_features)),
            layers.Dropout(0.2),
            layers.LSTM(64, return_sequences=False),
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
    
    def build_cnn1d_model(self):
        """Build 1D CNN model for sequences"""
        model = models.Sequential([
            layers.Conv1D(64, kernel_size=5, activation='relu', 
                         input_shape=(self.window_size, self.n_features)),
            layers.MaxPooling1D(2),
            layers.Conv1D(128, kernel_size=3, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, kernel_size=3, activation='relu'),
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
            layers.Conv1D(64, kernel_size=3, activation='relu',
                         input_shape=(self.window_size, self.n_features)),
            layers.Conv1D(64, kernel_size=3, activation='relu'),
            layers.MaxPooling1D(2),
            layers.LSTM(50, return_sequences=False),
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
# 5. TRAINING AND EVALUATION UTILITIES
# =============================================================================

def train_neural_network(model, X_train, y_train, X_test, y_test, 
                        model_name="Model", epochs=50, batch_size=32):
    """Train and evaluate a neural network model"""
    
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
    
    return history, y_pred_classes, test_accuracy

def train_models_for_dc(processor, df, dc_num, test_runs, results_dict):
    """
    Train all models for a specific DC system
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING DC{dc_num} SYSTEM")
    print(f"{'='*80}")
    
    dc_results = {}
    
    # =========================================================================
    # PATH A: ENGINEERED FEATURES
    # =========================================================================
    print(f"\nPath A: Engineered Features for DC{dc_num}")
    print("-" * 40)
    
    # Create windows with feature engineering
    X_features, y, groups = processor.create_windows_for_dc(df, dc_num, engineer_features=True)
    
    if X_features is None:
        print(f"Skipping DC{dc_num} - no valid data")
        return
    
    print(f"Created {len(X_features)} windows with {X_features.shape[1]} engineered features")
    
    # Prepare data
    X_train_feat, X_test_feat, y_train, y_test, y_train_orig, y_test_orig = processor.prepare_data_for_dc(
        X_features, y, groups, test_runs, dc_num, is_sequence=False
    )
    
    if X_train_feat is None:
        print(f"Skipping DC{dc_num} - insufficient data for train/test split")
        return
    
    print(f"Training samples: {len(X_train_feat)}, Testing samples: {len(X_test_feat)}")
    
    # Get number of classes
    n_classes = len(np.unique(y_train))
    print(f"Number of classes: {n_classes}")
    
    # Train traditional ML models
    print(f"\nTraining traditional ML models for DC{dc_num}...")
    
    trad_ml = TraditionalMLModels(n_classes)
    
    # XGBoost
    print("  Training XGBoost...")
    xgb_model, xgb_pred, xgb_acc = trad_ml.train_xgboost(
        X_train_feat, y_train, X_test_feat, y_test
    )
    dc_results[f'XGBoost'] = xgb_acc
    print(f"    Accuracy: {xgb_acc:.4f}")
    
    # LightGBM
    print("  Training LightGBM...")
    lgb_model, lgb_pred, lgb_acc = trad_ml.train_lightgbm(
        X_train_feat, y_train, X_test_feat, y_test
    )
    dc_results[f'LightGBM'] = lgb_acc
    print(f"    Accuracy: {lgb_acc:.4f}")
    
    # Random Forest
    print("  Training Random Forest...")
    rf_model, rf_pred, rf_acc = trad_ml.train_random_forest(
        X_train_feat, y_train, X_test_feat, y_test
    )
    dc_results[f'Random Forest'] = rf_acc
    print(f"    Accuracy: {rf_acc:.4f}")
    
    # MLP
    print("  Training MLP...")
    mlp_model = trad_ml.build_mlp(X_train_feat.shape[1])
    mlp_hist, mlp_pred, mlp_acc = train_neural_network(
        mlp_model, X_train_feat, y_train, X_test_feat, y_test,
        model_name="MLP", epochs=50
    )
    dc_results[f'MLP'] = mlp_acc
    print(f"    Accuracy: {mlp_acc:.4f}")
    
    # =========================================================================
    # PATH B: RAW SEQUENCES
    # =========================================================================
    print(f"\nPath B: Raw Sequences for DC{dc_num}")
    print("-" * 40)
    
    # Create windows with raw sequences
    X_sequences, y_seq, groups_seq = processor.create_windows_for_dc(df, dc_num, engineer_features=False)
    
    if X_sequences is None:
        print(f"Skipping sequence models for DC{dc_num} - no valid data")
        results_dict[f'DC{dc_num}'] = dc_results
        return
    
    print(f"Created {len(X_sequences)} sequences of shape {X_sequences.shape[1:]} (timesteps, features)")
    
    # Prepare sequence data
    X_train_seq, X_test_seq, y_train_seq, y_test_seq, _, _ = processor.prepare_data_for_dc(
        X_sequences, y_seq, groups_seq, test_runs, dc_num, is_sequence=True
    )
    
    if X_train_seq is None:
        print(f"Skipping sequence models for DC{dc_num} - insufficient data")
        results_dict[f'DC{dc_num}'] = dc_results
        return
    
    print(f"Training sequences: {X_train_seq.shape}, Testing sequences: {X_test_seq.shape}")
    
    # Initialize sequence model builder
    seq_models = SequenceModels(
        window_size=X_train_seq.shape[1],
        n_features=X_train_seq.shape[2],
        n_classes=n_classes
    )
    
    print(f"\nTraining deep learning models for DC{dc_num}...")
    
    # LSTM
    print("  Training LSTM...")
    lstm_model = seq_models.build_lstm_model()
    lstm_hist, lstm_pred, lstm_acc = train_neural_network(
        lstm_model, X_train_seq, y_train_seq, X_test_seq, y_test_seq,
        model_name="LSTM", epochs=50
    )
    dc_results[f'LSTM'] = lstm_acc
    print(f"    Accuracy: {lstm_acc:.4f}")
    
    # 1D CNN
    print("  Training 1D-CNN...")
    cnn_model = seq_models.build_cnn1d_model()
    cnn_hist, cnn_pred, cnn_acc = train_neural_network(
        cnn_model, X_train_seq, y_train_seq, X_test_seq, y_test_seq,
        model_name="1D-CNN", epochs=50
    )
    dc_results[f'1D-CNN'] = cnn_acc
    print(f"    Accuracy: {cnn_acc:.4f}")
    
    # CNN-LSTM
    print("  Training CNN-LSTM...")
    cnn_lstm_model = seq_models.build_cnn_lstm_model()
    cnn_lstm_hist, cnn_lstm_pred, cnn_lstm_acc = train_neural_network(
        cnn_lstm_model, X_train_seq, y_train_seq, X_test_seq, y_test_seq,
        model_name="CNN-LSTM", epochs=50
    )
    dc_results[f'CNN-LSTM'] = cnn_lstm_acc
    print(f"    Accuracy: {cnn_lstm_acc:.4f}")
    
    # Find best model for this DC
    best_model = max(dc_results, key=dc_results.get)
    best_acc = dc_results[best_model]
    
    print(f"\nBest model for DC{dc_num}: {best_model} (Accuracy: {best_acc:.4f})")
    
    # Store results
    results_dict[f'DC{dc_num}'] = dc_results
    
    # Get best predictions for classification report
    if best_model == 'XGBoost':
        best_pred = xgb_pred
        best_y_test = y_test
    elif best_model == 'LightGBM':
        best_pred = lgb_pred
        best_y_test = y_test
    elif best_model == 'Random Forest':
        best_pred = rf_pred
        best_y_test = y_test
    elif best_model == 'MLP':
        best_pred = mlp_pred
        best_y_test = y_test
    elif best_model == 'LSTM':
        best_pred = lstm_pred
        best_y_test = y_test_seq
    elif best_model == '1D-CNN':
        best_pred = cnn_pred
        best_y_test = y_test_seq
    else:  # CNN-LSTM
        best_pred = cnn_lstm_pred
        best_y_test = y_test_seq
    
    # Print classification report
    label_encoder = processor.dc1_label_encoder if dc_num == 1 else processor.dc2_label_encoder
    print(f"\nClassification Report for DC{dc_num} ({best_model}):")
    print(classification_report(best_y_test, best_pred, 
                              target_names=label_encoder.classes_))

# =============================================================================
# 6. MAIN EXECUTION SCRIPT
# =============================================================================

def main(filepath='power_data.parquet'):
    """
    Main execution function for dual DC system classification
    """
    print("=" * 80)
    print("DUAL DC POWER SYSTEM CLASSIFICATION")
    print("Comparing Feature Engineering vs Raw Sequences")
    print("=" * 80)
    
    # Load data
    df = load_power_data(filepath)
    
    # Initialize processor
    processor = DualDCPowerDataProcessor(window_size=100, stride=20)
    
    # Determine test runs for splitting
    if 'test_run' in df.columns:
        unique_runs = df['test_run'].unique()
        n_test_runs = max(1, int(len(unique_runs) * 0.3))
        test_runs = np.random.choice(unique_runs, n_test_runs, replace=False)
        print(f"\nUsing test runs {sorted(test_runs)} for testing")
    else:
        # If no test_run column, use random split
        print("\nNo 'test_run' column found, will use time-based splitting")
        # Create synthetic test runs based on time chunks
        df['test_run'] = pd.qcut(range(len(df)), q=10, labels=False)
        unique_runs = df['test_run'].unique()
        n_test_runs = 3
        test_runs = np.random.choice(unique_runs, n_test_runs, replace=False)
    
    # Store all results
    all_results = {}
    
    # Train models for DC1
    if all(col in df.columns for col in ['dc1_voltage', 'dc1_current', 'dc1_status']):
        train_models_for_dc(processor, df, dc_num=1, test_runs=test_runs, results_dict=all_results)
    else:
        print("\nDC1 columns not found, skipping DC1 processing")
    
    # Train models for DC2
    if all(col in df.columns for col in ['dc2_voltage', 'dc2_current', 'dc2_status']):
        train_models_for_dc(processor, df, dc_num=2, test_runs=test_runs, results_dict=all_results)
    else:
        print("\nDC2 columns not found, skipping DC2 processing")
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    
    for dc_system, results in all_results.items():
        print(f"\n{dc_system} Results:")
        print("-" * 40)
        
        # Separate by approach
        feature_models = {k: v for k, v in results.items() if k in ['XGBoost', 'LightGBM', 'Random Forest', 'MLP']}
        sequence_models = {k: v for k, v in results.items() if k in ['LSTM', '1D-CNN', 'CNN-LSTM']}
        
        if feature_models:
            print("  Engineered Features:")
            for model, acc in feature_models.items():
                print(f"    {model:20} {acc:.4f}")
        
        if sequence_models:
            print("  Raw Sequences:")
            for model, acc in sequence_models.items():
                print(f"    {model:20} {acc:.4f}")
        
        best_model = max(results, key=results.get)
        print(f"\n  🏆 Best: {best_model} ({results[best_model]:.4f})")
    
    return all_results

if __name__ == "__main__":
    # Change this to your parquet file path
    results = main('your_power_data.parquet')
