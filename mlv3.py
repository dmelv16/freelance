import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import xgboost as xgb
import lightgbm as lgb
from typing import Generator, Tuple, Optional, Dict
import gc
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. MEMORY-EFFICIENT DATA GENERATOR
# =============================================================================
class MemoryEfficientDataGenerator:
    """Generator-based data processor for large parquet files"""
    
    def __init__(self, filepath: str, window_size: int = 100, stride: int = 20, 
                 chunk_size: int = 10000):
        """
        Args:
            filepath: Path to parquet file
            window_size: Number of samples in each window
            stride: Step size for sliding window
            chunk_size: Number of rows to process at once
        """
        self.filepath = filepath
        self.window_size = window_size
        self.stride = stride
        self.chunk_size = chunk_size
        
        # Get parquet file info
        self.parquet_file = pq.ParquetFile(filepath)
        self.num_row_groups = self.parquet_file.num_row_groups
        
        # Initialize scalers and encoders
        self.dc1_scaler = StandardScaler()
        self.dc2_scaler = StandardScaler()
        self.dc1_label_encoder = LabelEncoder()
        self.dc2_label_encoder = LabelEncoder()
        
        # Fit encoders on a sample
        self._fit_encoders()
        
    def _fit_encoders(self):
        """Fit label encoders on a sample of the data"""
        # Read first row group to get unique labels
        first_batch = self.parquet_file.read_row_group(0).to_pandas()
        
        if 'dc1_status' in first_batch.columns:
            # Get all unique statuses from first few row groups (for better coverage)
            dc1_statuses = []
            for i in range(min(3, self.num_row_groups)):
                batch = self.parquet_file.read_row_group(i).to_pandas()
                dc1_statuses.extend(batch['dc1_status'].unique())
            self.dc1_label_encoder.fit(list(set(dc1_statuses)))
            
        if 'dc2_status' in first_batch.columns:
            dc2_statuses = []
            for i in range(min(3, self.num_row_groups)):
                batch = self.parquet_file.read_row_group(i).to_pandas()
                dc2_statuses.extend(batch['dc2_status'].unique())
            self.dc2_label_encoder.fit(list(set(dc2_statuses)))
    
    def generate_windows_for_dc(self, dc_num: int, engineer_features: bool = True,
                               test_runs: Optional[list] = None, 
                               is_training: bool = True) -> Generator:
        """
        Generator that yields windows for a specific DC system
        
        Args:
            dc_num: 1 or 2 for DC1 or DC2
            engineer_features: If True, yield engineered features; if False, yield raw sequences
            test_runs: Test run IDs to include/exclude based on is_training
            is_training: If True, exclude test_runs; if False, only include test_runs
        
        Yields:
            Tuple of (window_features, label)
        """
        voltage_col = f'dc{dc_num}_voltage'
        current_col = f'dc{dc_num}_current'
        status_col = f'dc{dc_num}_status'
        
        # Process each row group
        for group_idx in range(self.num_row_groups):
            # Read row group
            df_chunk = self.parquet_file.read_row_group(group_idx).to_pandas()
            
            # Check if DC columns exist
            if not all(col in df_chunk.columns for col in [voltage_col, current_col, status_col]):
                continue
            
            # Filter by test runs if specified
            if test_runs is not None and 'test_run' in df_chunk.columns:
                if is_training:
                    df_chunk = df_chunk[~df_chunk['test_run'].isin(test_runs)]
                else:
                    df_chunk = df_chunk[df_chunk['test_run'].isin(test_runs)]
            
            if len(df_chunk) < self.window_size:
                continue
            
            # Sort by timestamp if available
            if 'timestamp' in df_chunk.columns:
                df_chunk = df_chunk.sort_values('timestamp')
            
            # Generate windows from this chunk
            for i in range(0, len(df_chunk) - self.window_size + 1, self.stride):
                window_data = df_chunk.iloc[i:i + self.window_size]
                
                # Extract signals
                voltage = window_data[voltage_col].values
                current = window_data[current_col].values
                
                # Skip if contains NaN
                if np.any(np.isnan(voltage)) or np.any(np.isnan(current)):
                    continue
                
                if engineer_features:
                    # Engineer features
                    features = self._engineer_features(voltage, current)
                else:
                    # Raw sequence - ensure correct shape
                    features = np.stack([voltage, current], axis=1).astype(np.float32)
                
                # Get label
                label = window_data[status_col].iloc[-1]
                
                # Encode label
                if dc_num == 1:
                    label_encoded = self.dc1_label_encoder.transform([label])[0]
                else:
                    label_encoded = self.dc2_label_encoder.transform([label])[0]
                
                yield features, label_encoded
            
            # Clear memory after processing chunk
            del df_chunk
            gc.collect()
    
    def _engineer_features(self, voltage: np.ndarray, current: np.ndarray) -> np.ndarray:
        """Engineer features from voltage and current window"""
        power = voltage * current
        features = []
        
        for signal in [voltage, current, power]:
            features.extend([
                np.mean(signal),
                np.std(signal),
                np.min(signal),
                np.max(signal),
                np.median(signal),
                np.percentile(signal, 25),
                np.percentile(signal, 75),
                signal[-1] - signal[0],
            ])
            
            diff = np.diff(signal)
            if len(diff) > 0:
                features.extend([
                    np.mean(diff),
                    np.std(diff),
                    np.max(np.abs(diff)),
                ])
                
                # Slope
                x = np.arange(len(signal))
                slope, _ = np.polyfit(x, signal, 1)
                features.append(slope)
                
                # Additional features
                features.extend([
                    np.sum(np.abs(diff)),
                    len(np.where(np.diff(np.sign(diff)))[0]),
                ])
            else:
                features.extend([0] * 6)
        
        # Correlation
        if len(voltage) > 1:
            corr = np.corrcoef(voltage, current)[0, 1]
            features.append(corr if not np.isnan(corr) else 0)
        else:
            features.append(0)
        
        return np.array(features, dtype=np.float32)
    
    def create_tf_dataset(self, dc_num: int, batch_size: int = 32, 
                         engineer_features: bool = False,
                         test_runs: Optional[list] = None,
                         is_training: bool = True) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset from the generator for efficient training
        
        Args:
            dc_num: DC system number (1 or 2)
            batch_size: Batch size for training
            engineer_features: Use engineered features or raw sequences
            test_runs: Test run IDs for train/test split
            is_training: Whether this is for training or testing
        
        Returns:
            tf.data.Dataset ready for model training
        """
        # Fit scaler if needed for sequences
        if not engineer_features:
            scaler = self.dc1_scaler if dc_num == 1 else self.dc2_scaler
            if is_training and not hasattr(scaler, 'mean_'):
                print(f"Fitting scaler for DC{dc_num}...")
                sample_data = []
                for features, _ in self.generate_windows_for_dc(
                    dc_num, False, test_runs, is_training
                ):
                    sample_data.append(features.reshape(-1, 2))
                    if len(sample_data) >= 100:
                        break
                if sample_data:
                    sample_array = np.vstack(sample_data)
                    scaler.fit(sample_array)
        
        # Define generator function with scaling
        def data_generator():
            scaler = self.dc1_scaler if dc_num == 1 else self.dc2_scaler
            for features, label in self.generate_windows_for_dc(
                dc_num, engineer_features, test_runs, is_training
            ):
                if not engineer_features and hasattr(scaler, 'mean_'):
                    # Scale sequences
                    original_shape = features.shape
                    features_flat = features.reshape(-1, 2)
                    features_scaled = scaler.transform(features_flat)
                    features = features_scaled.reshape(original_shape).astype(np.float32)
                
                yield features, label
        
        # Determine output signature
        if engineer_features:
            # Number of engineered features
            n_features = 43  # Based on our feature engineering
            output_signature = (
                tf.TensorSpec(shape=(n_features,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        else:
            # Raw sequences
            output_signature = (
                tf.TensorSpec(shape=(self.window_size, 2), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        
        # Create dataset from generator
        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=output_signature
        )
        
        # Batch and prefetch for performance
        if is_training:
            dataset = dataset.shuffle(buffer_size=1000)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

# =============================================================================
# 2. MEMORY-EFFICIENT MODEL TRAINING
# =============================================================================
class MemoryEfficientTrainer:
    """Handles training with generators and limited memory"""
    
    def __init__(self, data_generator: MemoryEfficientDataGenerator):
        self.generator = data_generator
        self.all_reports = {}  # Store all classification reports
        
    def train_deep_learning_model(self, model: tf.keras.Model, dc_num: int,
                                 test_runs: list, epochs: int = 30,
                                 batch_size: int = 32) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Train a deep learning model using data generators
        
        Returns:
            Tuple of (accuracy, predictions, true_labels)
        """
        # Create training dataset
        train_dataset = self.generator.create_tf_dataset(
            dc_num=dc_num,
            batch_size=batch_size,
            engineer_features=False,
            test_runs=test_runs,
            is_training=True
        )
        
        # Create validation dataset
        val_dataset = self.generator.create_tf_dataset(
            dc_num=dc_num,
            batch_size=batch_size,
            engineer_features=False,
            test_runs=test_runs,
            is_training=False
        )
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
        
        # Train model
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        # Evaluate
        eval_results = model.evaluate(val_dataset, verbose=0)
        accuracy = eval_results[1] if len(eval_results) > 1 else eval_results
        
        # Get predictions
        predictions = []
        true_labels = []
        for x_batch, y_batch in val_dataset:
            pred_batch = model.predict(x_batch, verbose=0)
            predictions.append(np.argmax(pred_batch, axis=1))
            true_labels.append(y_batch.numpy())
        
        predictions = np.concatenate(predictions)
        true_labels = np.concatenate(true_labels)
        
        return accuracy, predictions, true_labels
    
    def train_traditional_model_on_sample(self, model_type: str, dc_num: int,
                                         test_runs: list, 
                                         sample_size: int = 10000) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Train traditional ML models on a sample that fits in memory
        
        Args:
            model_type: 'xgboost' or 'lightgbm'
            dc_num: DC system number
            test_runs: Test run IDs
            sample_size: Number of samples to use
        
        Returns:
            Tuple of (accuracy, predictions, true_labels)
        """
        print(f"Collecting {sample_size} samples for {model_type}...")
        
        # Collect training samples
        X_train, y_train = [], []
        for features, label in self.generator.generate_windows_for_dc(
            dc_num, engineer_features=True, test_runs=test_runs, is_training=True
        ):
            X_train.append(features)
            y_train.append(label)
            if len(X_train) >= sample_size:
                break
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Collect test samples
        X_test, y_test = [], []
        for features, label in self.generator.generate_windows_for_dc(
            dc_num, engineer_features=True, test_runs=test_runs, is_training=False
        ):
            X_test.append(features)
            y_test.append(label)
            if len(X_test) >= sample_size // 3:  # Smaller test set
                break
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        
        # Train model
        if model_type == 'xgboost':
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                n_jobs=-1,
                random_state=42
            )
        else:  # lightgbm
            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                n_jobs=-1,
                verbose=-1,
                random_state=42
            )
        
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy, y_pred, y_test
    
    def save_classification_report(self, dc_num: int, model_name: str, 
                                  true_labels: np.ndarray, predictions: np.ndarray):
        """
        Save and print classification report for a model
        
        Args:
            dc_num: DC system number
            model_name: Name of the model
            true_labels: True labels
            predictions: Model predictions
        """
        label_encoder = self.generator.dc1_label_encoder if dc_num == 1 else self.generator.dc2_label_encoder
        
        report_str = classification_report(
            true_labels, predictions, 
            target_names=label_encoder.classes_,
            output_dict=False
        )
        
        report_dict = classification_report(
            true_labels, predictions, 
            target_names=label_encoder.classes_,
            output_dict=True
        )
        
        # Store report
        if f'DC{dc_num}' not in self.all_reports:
            self.all_reports[f'DC{dc_num}'] = {}
        self.all_reports[f'DC{dc_num}'][model_name] = {
            'report_str': report_str,
            'report_dict': report_dict,
            'accuracy': accuracy_score(true_labels, predictions)
        }
        
        # Print report
        print(f"\n{'='*60}")
        print(f"Classification Report - DC{dc_num} - {model_name}")
        print(f"{'='*60}")
        print(report_str)
        print(f"{'='*60}\n")

# =============================================================================
# 3. MODEL BUILDERS
# =============================================================================
def build_lstm_model(window_size: int, n_features: int, n_classes: int) -> tf.keras.Model:
    """Build LSTM model for sequences"""
    model = models.Sequential([
        layers.Input(shape=(window_size, n_features)),  # Explicit input layer
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_cnn_model(window_size: int, n_features: int, n_classes: int) -> tf.keras.Model:
    """Build 1D CNN model for sequences"""
    model = models.Sequential([
        layers.Input(shape=(window_size, n_features)),  # Explicit input layer
        layers.Conv1D(32, kernel_size=5, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# =============================================================================
# 4. MAIN EXECUTION
# =============================================================================
def main(filepath: str = 'power_data.parquet', 
         window_size: int = 100,
         stride: int = 20,
         sample_size: int = 10000,
         show_all_reports: bool = True):
    """
    Main execution function with memory-efficient processing
    
    Args:
        filepath: Path to parquet file
        window_size: Window size for sequences
        stride: Stride for sliding window
        sample_size: Number of samples for traditional ML models
        show_all_reports: If True, show classification reports for all models
    """
    print("=" * 80)
    print("MEMORY-EFFICIENT POWER CLASSIFICATION SYSTEM")
    print("=" * 80)
    
    # Initialize data generator
    print(f"\nInitializing data generator for {filepath}")
    generator = MemoryEfficientDataGenerator(
        filepath=filepath,
        window_size=window_size,
        stride=stride
    )
    print(f"Parquet file has {generator.num_row_groups} row groups")
    
    # Get test runs for splitting (using first row group)
    first_batch = generator.parquet_file.read_row_group(0).to_pandas()
    if 'test_run' in first_batch.columns:
        unique_runs = first_batch['test_run'].unique()
        # Use 30% for testing
        n_test_runs = max(1, int(len(unique_runs) * 0.3))
        test_runs = np.random.choice(unique_runs, n_test_runs, replace=False).tolist()
        print(f"Using test runs {test_runs} for testing")
    else:
        print("No test_run column found, using sequential split")
        test_runs = None
    
    # Initialize trainer
    trainer = MemoryEfficientTrainer(generator)
    
    # Store results
    results = {}
    all_model_results = {}  # Store detailed results for all models
    
    # Process each DC system
    for dc_num in [1, 2]:
        print(f"\n{'='*80}")
        print(f"PROCESSING DC{dc_num}")
        print(f"{'='*80}")
        
        dc_results = {}
        dc_model_details = {}
        
        # Check if DC columns exist
        voltage_col = f'dc{dc_num}_voltage'
        current_col = f'dc{dc_num}_current'
        status_col = f'dc{dc_num}_status'
        
        if not all(col in first_batch.columns for col in [voltage_col, current_col, status_col]):
            print(f"DC{dc_num} columns not found, skipping")
            continue
        
        # Get number of classes
        statuses = []
        for i in range(min(3, generator.num_row_groups)):
            batch = generator.parquet_file.read_row_group(i).to_pandas()
            if status_col in batch.columns:
                statuses.extend(batch[status_col].unique())
        n_classes = len(set(statuses))
        print(f"Number of classes for DC{dc_num}: {n_classes}")
        print(f"Classes: {set(statuses)}")
        
        # Train traditional ML models on sample
        print(f"\n{'='*40}")
        print(f"Training Traditional ML Models")
        print(f"{'='*40}")
        
        # XGBoost
        print("\n📊 Training XGBoost...")
        xgb_acc, xgb_pred, xgb_true = trainer.train_traditional_model_on_sample(
            'xgboost', dc_num, test_runs, sample_size
        )
        dc_results['XGBoost (Sample)'] = xgb_acc
        print(f"✅ XGBoost Accuracy: {xgb_acc:.4f}")
        
        # Save and display classification report
        if show_all_reports:
            trainer.save_classification_report(dc_num, 'XGBoost', xgb_true, xgb_pred)
        
        dc_model_details['XGBoost'] = {
            'predictions': xgb_pred,
            'true_labels': xgb_true,
            'accuracy': xgb_acc
        }
        
        # LightGBM
        print("\n📊 Training LightGBM...")
        lgb_acc, lgb_pred, lgb_true = trainer.train_traditional_model_on_sample(
            'lightgbm', dc_num, test_runs, sample_size
        )
        dc_results['LightGBM (Sample)'] = lgb_acc
        print(f"✅ LightGBM Accuracy: {lgb_acc:.4f}")
        
        # Save and display classification report
        if show_all_reports:
            trainer.save_classification_report(dc_num, 'LightGBM', lgb_true, lgb_pred)
        
        dc_model_details['LightGBM'] = {
            'predictions': lgb_pred,
            'true_labels': lgb_true,
            'accuracy': lgb_acc
        }
        
        # Train deep learning models with generators
        print(f"\n{'='*40}")
        print(f"Training Deep Learning Models")
        print(f"{'='*40}")
        
        # LSTM
        print("\n🧠 Training LSTM...")
        lstm_model = build_lstm_model(window_size, 2, n_classes)
        lstm_acc, lstm_pred, lstm_true = trainer.train_deep_learning_model(
            lstm_model, dc_num, test_runs, epochs=20
        )
        dc_results['LSTM (Generator)'] = lstm_acc
        print(f"✅ LSTM Accuracy: {lstm_acc:.4f}")
        
        # Save and display classification report
        if show_all_reports:
            trainer.save_classification_report(dc_num, 'LSTM', lstm_true, lstm_pred)
        
        dc_model_details['LSTM'] = {
            'predictions': lstm_pred,
            'true_labels': lstm_true,
            'accuracy': lstm_acc
        }
        
        # CNN
        print("\n🧠 Training 1D-CNN...")
        cnn_model = build_cnn_model(window_size, 2, n_classes)
        cnn_acc, cnn_pred, cnn_true = trainer.train_deep_learning_model(
            cnn_model, dc_num, test_runs, epochs=20
        )
        dc_results['1D-CNN (Generator)'] = cnn_acc
        print(f"✅ 1D-CNN Accuracy: {cnn_acc:.4f}")
        
        # Save and display classification report
        if show_all_reports:
            trainer.save_classification_report(dc_num, '1D-CNN', cnn_true, cnn_pred)
        
        dc_model_details['1D-CNN'] = {
            'predictions': cnn_pred,
            'true_labels': cnn_true,
            'accuracy': cnn_acc
        }
        
        # Store results
        results[f'DC{dc_num}'] = dc_results
        all_model_results[f'DC{dc_num}'] = dc_model_details
        
        # Find best model
        best_model = max(dc_results, key=dc_results.get)
        print(f"\n🏆 Best model for DC{dc_num}: {best_model} ({dc_results[best_model]:.4f})")
        
        # Clear memory
        gc.collect()
        if tf.config.list_physical_devices('GPU'):
            tf.keras.backend.clear_session()
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    
    for dc_system, dc_results in results.items():
        print(f"\n{dc_system}:")
        print("-" * 40)
        
        # Sort models by accuracy (descending)
        sorted_models = sorted(dc_results.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (model, acc) in enumerate(sorted_models, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            print(f"{medal} {rank}. {model:25} {acc:.4f}")
        
        if dc_results:
            best = max(dc_results, key=dc_results.get)
            print(f"\n📈 Best Performance: {best} ({dc_results[best]:.4f})")
    
    # Print summary of all classification reports if requested
    if show_all_reports and trainer.all_reports:
        print("\n" + "=" * 80)
        print("SUMMARY: PER-CLASS PERFORMANCE COMPARISON")
        print("=" * 80)
        
        for dc_system in trainer.all_reports:
            print(f"\n{dc_system} - F1-Scores by Class:")
            print("-" * 60)
            
            # Get class names from first model
            first_model = list(trainer.all_reports[dc_system].keys())[0]
            class_names = list(trainer.all_reports[dc_system][first_model]['report_dict'].keys())
            class_names = [c for c in class_names if c not in ['accuracy', 'macro avg', 'weighted avg']]
            
            # Create comparison table
            print(f"{'Class':<20}", end="")
            for model in trainer.all_reports[dc_system]:
                print(f"{model[:10]:<12}", end="")
            print()
            print("-" * 60)
            
            for class_name in class_names:
                print(f"{class_name:<20}", end="")
                for model in trainer.all_reports[dc_system]:
                    f1_score = trainer.all_reports[dc_system][model]['report_dict'].get(
                        class_name, {}).get('f1-score', 0)
                    print(f"{f1_score:<12.4f}", end="")
                print()
    
    return results, trainer.all_reports

# =============================================================================
# 5. INCREMENTAL LEARNING FOR TRADITIONAL MODELS (ADVANCED)
# =============================================================================
class IncrementalLearner:
    """
    Incremental learning for traditional models that can't fit all data in memory
    Uses mini-batch gradient boosting
    """
    
    def __init__(self, model_type: str = 'lightgbm'):
        self.model_type = model_type
        self.model = None
        
    def train_incremental(self, generator: MemoryEfficientDataGenerator,
                         dc_num: int, test_runs: list,
                         batch_size: int = 5000,
                         n_estimators_per_batch: int = 10):
        """
        Train model incrementally on batches
        """
        print(f"Starting incremental training with {self.model_type}...")
        
        batch_num = 0
        
        # Process training data in batches
        X_batch, y_batch = [], []
        
        for features, label in generator.generate_windows_for_dc(
            dc_num, engineer_features=True, test_runs=test_runs, is_training=True
        ):
            X_batch.append(features)
            y_batch.append(label)
            
            if len(X_batch) >= batch_size:
                X_batch = np.array(X_batch)
                y_batch = np.array(y_batch)
                
                if self.model is None:
                    # Initialize model with first batch
                    if self.model_type == 'lightgbm':
                        self.model = lgb.LGBMClassifier(
                            n_estimators=n_estimators_per_batch,
                            max_depth=6,
                            learning_rate=0.1,
                            n_jobs=-1,
                            verbose=-1
                        )
                    self.model.fit(X_batch, y_batch)
                else:
                    # Continue training (for LightGBM)
                    self.model.n_estimators += n_estimators_per_batch
                    self.model.fit(
                        X_batch, y_batch,
                        init_model=self.model
                    )
                
                batch_num += 1
                print(f"  Processed batch {batch_num} ({len(X_batch)} samples)")
                
                # Reset batch
                X_batch, y_batch = [], []
                gc.collect()
        
        print(f"Incremental training complete ({batch_num} batches)")
        
        # Evaluate on test data
        X_test, y_test = [], []
        for features, label in generator.generate_windows_for_dc(
            dc_num, engineer_features=True, test_runs=test_runs, is_training=False
        ):
            X_test.append(features)
            y_test.append(label)
            if len(X_test) >= batch_size // 2:  # Smaller test batch
                break
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy, y_pred, y_test

if __name__ == "__main__":
    # Adjust these parameters based on your system's memory
    results, all_reports = main(
        filepath='your_power_data.parquet',
        window_size=100,
        stride=20,
        sample_size=10000,  # Adjust based on available memory
        show_all_reports=True  # Set to True to see all model reports
    )
