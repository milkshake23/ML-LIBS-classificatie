"""
LIBS Spectral Data Analysis Module

This module provides functions for loading, preprocessing, and analyzing LIBS spectral data
with Linear Discriminant Analysis (LDA) for tire classification.

"""

import os
import h5py # type: ignore[import]
import re # type: ignore[import]
import pandas as pd # type: ignore[import]
import numpy as np
import gc
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # type: ignore[import]
from sklearn.model_selection import train_test_split, cross_val_score # type: ignore[import]
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # type: ignore[import]
from sklearn.ensemble import RandomForestClassifier # type: ignore[import]
from sklearn.linear_model import LogisticRegression # type: ignore[import]
import matplotlib.pyplot as plt # type: ignore[import]
from sklearn.preprocessing import LabelEncoder # type: ignore[import]
import seaborn as sns # type: ignore[import]


class LibsDataLoader:
    """Handles loading and preprocessing of LIBS spectral data files."""
    
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.wavelength_reference = None
        
    @staticmethod
    def extract_tire_number(filename):
        """Extract tire number from filename"""
        match = re.search(r'_B(\d+)_', filename)
        if match:
            return int(match.group(1))
        else:
            match = re.search(r'nr-(\d+)', filename)
            if match:
                return int(match.group(1))
        return None
    
    def process_file_generator(self, max_files=None):
        """
        Generator that yields individual measurements one at a time
        More memory efficient than storing all in memory
        """
        all_files = [f for f in os.listdir(self.data_directory) if f.endswith('.h5')]
        if max_files:
            all_files = all_files[:max_files]
        
        total_measurements = 0
        
        for i, filename in enumerate(all_files):
            if i % 10 == 0:
                print(f"Processing file {i+1}/{len(all_files)}: {filename}")
            
            try:
                filepath = os.path.join(self.data_directory, filename)
                tire_number = self.extract_tire_number(filename)
                
                if tire_number is None:
                    continue
                
                # Determine origin from filename
                if "tread" in filename.lower():
                    origin = 'tread'
                elif "innerliner" in filename.lower():
                    origin = 'innerliner'
                elif "sidewall" in filename.lower():
                    origin = 'sidewall'
                else:
                    continue
                
                # Load HDF5 file efficiently
                with h5py.File(filepath, 'r') as h5:
                    intensity = h5['intensity'][:]
                    wavelength = h5['wavelength'][:]
                    
                    # Store wavelength reference only once
                    if self.wavelength_reference is None:
                        self.wavelength_reference = wavelength.copy()
                    
                    # Process measurements
                    if intensity.ndim == 2:
                        n_measurements, n_wavelengths = intensity.shape
                        
                        # Yield each measurement individually
                        for measurement_idx in range(n_measurements):
                            measurement_data = {
                                'tire_number': tire_number,
                                'origin': origin,
                                'measurement_id': measurement_idx,
                                'intensities': intensity[measurement_idx, :].astype(np.float32),
                            }
                            yield measurement_data
                            total_measurements += 1
                    else:
                        # Single measurement case
                        measurement_data = {
                            'tire_number': tire_number,
                            'origin': origin,
                            'measurement_id': 0,
                            'intensities': intensity.astype(np.float32),
                        }
                        yield measurement_data
                        total_measurements += 1
                        
            except Exception as e:
                print(f"  Error processing {filename}: {e}")
                continue
        
        print(f"Total measurements processed: {total_measurements:,}")
    
    def load_all_measurements(self, max_files=None):
        """
        Load all measurements into a DataFrame with memory-efficient approach
        """
        print("="*70)
        print("MEMORY-EFFICIENT LOADING OF ALL INDIVIDUAL MEASUREMENTS")
        print("="*70)
        
        # First pass: count total measurements and get wavelength info
        measurement_count = 0
        wavelength_info = None
        
        print("First pass: Counting measurements...")
        for measurement in self.process_file_generator(max_files):
            measurement_count += 1
            if wavelength_info is None:
                n_wavelengths = len(measurement['intensities'])
                wavelength_info = n_wavelengths
                # Get actual wavelengths from the reference if available
                if self.wavelength_reference is not None:
                    actual_wavelengths = self.wavelength_reference.copy()
            
            if measurement_count % 10000 == 0:
                print(f"  Counted {measurement_count:,} measurements...")
        
        print(f"Total measurements found: {measurement_count:,}")
        print(f"Wavelengths per spectrum: {wavelength_info}")
        
        # Pre-allocate numpy arrays for maximum memory efficiency
        print("Pre-allocating memory-efficient arrays...")
        X_data = np.empty((measurement_count, wavelength_info), dtype=np.float32)
        y_data = np.empty(measurement_count, dtype='U10')
        metadata = np.empty((measurement_count, 2), dtype=np.int32)
        
        # Second pass: fill the arrays
        print("Second pass: Loading data into arrays...")
        idx = 0
        for measurement in self.process_file_generator(max_files):
            X_data[idx] = measurement['intensities']
            y_data[idx] = measurement['origin']
            metadata[idx, 0] = measurement['tire_number']
            metadata[idx, 1] = measurement['measurement_id']
            
            idx += 1
            
            if idx % 10000 == 0:
                print(f"  Loaded {idx:,}/{measurement_count:,} measurements ({idx/measurement_count*100:.1f}%)")
                gc.collect()
        
        # Create DataFrame efficiently
        print("Creating final DataFrame...")
        wavelength_columns = [f'{wl:.1f}' for wl in actual_wavelengths]
        
        df_individual = pd.DataFrame(X_data, columns=wavelength_columns, dtype=np.float32)
        df_individual['origin'] = y_data
        df_individual['tire_number'] = metadata[:, 0]  
        df_individual['measurement_id'] = metadata[:, 1]
        
        # Clean up intermediate arrays
        del X_data, y_data, metadata
        gc.collect()

        non_feature_cols = ['tire_number', 'origin', 'measurement_id']
        original_wavelength_cols = [col for col in df_individual.columns if col not in non_feature_cols]
    
        print(f"Original wavelength columns: {len(original_wavelength_cols)}")
    
        # Convert column names to floats, round, and find unique values
        original_wavelengths = [float(col) for col in original_wavelength_cols]
        rounded_wavelengths = np.round(original_wavelengths, 1)
        
        # Create mapping from original to rounded wavelengths
        wavelength_mapping = {}
        for orig_col, rounded_wl in zip(original_wavelength_cols, rounded_wavelengths):
            rounded_col_name = f'{rounded_wl:.1f}'
            if rounded_col_name not in wavelength_mapping:
                wavelength_mapping[rounded_col_name] = []
            wavelength_mapping[rounded_col_name].append(orig_col)
        
        print(f"After rounding: {len(wavelength_mapping)} unique wavelengths")
        
        # Create new DataFrame with rounded and deduplicated wavelength columns
        print("Creating DataFrame with deduplicated wavelength columns...")
        
        # Start with metadata columns
        new_df_data = {}
        for col in non_feature_cols:
            new_df_data[col] = df_individual[col].values
        
        # Process wavelength columns - combine duplicates by averaging
        for rounded_wl, orig_cols in wavelength_mapping.items():
            if len(orig_cols) == 1:
                # No duplicates, just rename
                new_df_data[rounded_wl] = df_individual[orig_cols[0]].values
            else:
                # Multiple columns map to same rounded wavelength - average them
                combined_values = df_individual[orig_cols].mean(axis=1).values
                new_df_data[rounded_wl] = combined_values
                print(f"  Averaged {len(orig_cols)} columns into {rounded_wl} nm")
        
        # Create final DataFrame
        df_final = pd.DataFrame(new_df_data)
        
        # Reorder columns: metadata first, then wavelengths in sorted order
        wavelength_cols_final = sorted([col for col in df_final.columns if col not in non_feature_cols], 
                                    key=lambda x: float(x))
        column_order = non_feature_cols + wavelength_cols_final
        df_final = df_final[column_order]
        
        # Final verification - check for any remaining duplicate columns
        duplicate_cols = df_final.columns[df_final.columns.duplicated()]
        if len(duplicate_cols) > 0:
            print(f"Warning: Found {len(duplicate_cols)} duplicate columns in final DataFrame!")
            print(f"Duplicate columns: {duplicate_cols.tolist()}")
            # Remove duplicates by keeping the first occurrence
            df_final = df_final.loc[:, ~df_final.columns.duplicated()]
            print(f"Removed duplicates, final shape: {df_final.shape}")
        
        print(f"\n" + "="*70)
        print("MEMORY-EFFICIENT DATASET CREATED")
        print("="*70)
        print(f"Total measurements: {len(df_final):,}")
        print(f"Dataset shape: {df_final.shape}")
        print(f"Unique wavelength columns: {df_final.shape[1] - 3}")  # Subtract metadata columns
        print(f"Memory usage: {df_final.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Get wavelength range from final columns
        final_wavelength_values = [float(col) for col in wavelength_cols_final]
        print(f"Wavelength range: {min(final_wavelength_values):.1f} - {max(final_wavelength_values):.1f} nm")
        
        if len(original_wavelength_cols) != len(wavelength_cols_final):
            print(f"Removed {len(original_wavelength_cols) - len(wavelength_cols_final)} duplicate wavelengths")
        
        # Store the final wavelengths for reference
        self.final_wavelengths = np.array(final_wavelength_values)
        
        return df_final
    
    def load_all_measurements_no_rounding(self, max_files=None):
        """
        Load all measurements into a DataFrame with memory-efficient approach
        """
        print("="*70)
        print("MEMORY-EFFICIENT LOADING OF ALL INDIVIDUAL MEASUREMENTS")
        print("="*70)
        
        # First pass: count total measurements and get wavelength info
        measurement_count = 0
        wavelength_info = None
        
        print("First pass: Counting measurements...")
        for measurement in self.process_file_generator(max_files):
            measurement_count += 1
            if wavelength_info is None:
                n_wavelengths = len(measurement['intensities'])
                wavelength_info = n_wavelengths
            
            if measurement_count % 10000 == 0:
                print(f"  Counted {measurement_count:,} measurements...")
        
        print(f"Total measurements found: {measurement_count:,}")
        print(f"Wavelengths per spectrum: {wavelength_info}")
        
        # Pre-allocate numpy arrays for maximum memory efficiency
        print("Pre-allocating memory-efficient arrays...")
        X_data = np.empty((measurement_count, wavelength_info), dtype=np.float32)
        y_data = np.empty(measurement_count, dtype='U10')
        metadata = np.empty((measurement_count, 2), dtype=np.int32)
        
        # Second pass: fill the arrays
        print("Second pass: Loading data into arrays...")
        idx = 0
        for measurement in self.process_file_generator(max_files):
            X_data[idx] = measurement['intensities']
            y_data[idx] = measurement['origin']
            metadata[idx, 0] = measurement['tire_number']
            metadata[idx, 1] = measurement['measurement_id']
            
            idx += 1
            
            if idx % 10000 == 0:
                print(f"  Loaded {idx:,}/{measurement_count:,} measurements ({idx/measurement_count*100:.1f}%)")
                gc.collect()
        
        # Create DataFrame efficiently
        print("Creating final DataFrame...")
        wavelength_columns = [f'{wl:.3f}' for wl in np.linspace(200, 1000, wavelength_info)]
        
        df_individual = pd.DataFrame(X_data, columns=wavelength_columns, dtype=np.float32)
        df_individual['origin'] = y_data
        df_individual['tire_number'] = metadata[:, 0]  
        df_individual['measurement_id'] = metadata[:, 1]
        
        # Clean up intermediate arrays
        del X_data, y_data, metadata
        gc.collect()
        
        print(f"\n" + "="*70)
        print("MEMORY-EFFICIENT DATASET CREATED")
        print("="*70)
        print(f"Total measurements: {len(df_individual):,}")
        print(f"Dataset shape: {df_individual.shape}")
        print(f"Memory usage: {df_individual.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        return df_individual

    def round_wavelengths(self, df):
        """
        Round wavelength columns to one decimal place and remove duplicates by averaging
        """
        print("Rounding wavelength columns and removing duplicates...")
        
        non_feature_cols = ['tire_number', 'origin', 'measurement_id']
        original_wavelength_cols = [col for col in df.columns if col not in non_feature_cols]
    
        print(f"Original wavelength columns: {len(original_wavelength_cols)}")
    
        # Convert column names to floats, round, and find unique values
        original_wavelengths = [float(col) for col in original_wavelength_cols]
        rounded_wavelengths = np.round(original_wavelengths, 1)
        
        # Create mapping from original to rounded wavelengths
        wavelength_mapping = {}
        for orig_col, rounded_wl in zip(original_wavelength_cols, rounded_wavelengths):
            rounded_col_name = f'{rounded_wl:.1f}'
            if rounded_col_name not in wavelength_mapping:
                wavelength_mapping[rounded_col_name] = []
            wavelength_mapping[rounded_col_name].append(orig_col)
        
        print(f"After rounding: {len(wavelength_mapping)} unique wavelengths")
        
        # Create new DataFrame with rounded and deduplicated wavelength columns
        print("Creating DataFrame with deduplicated wavelength columns...")
        
        # Start with metadata columns
        new_df_data = {}
        for col in non_feature_cols:
            new_df_data[col] = df[col].values
        
        # Process wavelength columns - combine duplicates by averaging
        for rounded_wl, orig_cols in wavelength_mapping.items():
            if len(orig_cols) == 1:
                # No duplicates, just rename
                new_df_data[rounded_wl] = df[orig_cols[0]].values
            else:
                # Multiple columns map to same rounded wavelength - average them
                combined_values = df[orig_cols].mean(axis=1).values
                new_df_data[rounded_wl] = combined_values
                print(f"  Averaged {len(orig_cols)} columns into {rounded_wl} nm")
        
        # Create final DataFrame
        df_final = pd.DataFrame(new_df_data)
        
        # Reorder columns: metadata first, then wavelengths in sorted order
        wavelength_cols_final = sorted([col for col in df_final.columns if col not in non_feature_cols], 
                                    key=lambda x: float(x))
        column_order = non_feature_cols + wavelength_cols_final
        df_final = df_final[column_order]
        # Final verification - check for any remaining duplicate columns
        duplicate_cols = df_final.columns[df_final.columns.duplicated()]
        if len(duplicate_cols) > 0:
            print(f"Warning: Found {len(duplicate_cols)} duplicate columns in final DataFrame!")
            print(f"Duplicate columns: {duplicate_cols.tolist()}")
            # Remove duplicates by keeping the first occurrence
            df_final = df_final.loc[:, ~df_final.columns.duplicated()]
            print(f"Removed duplicates, final shape: {df_final.shape}")
        print(f"\nFinal dataset shape: {df_final.shape}")
        return df_final

class LibsDataPreprocessor:
    """Handles preprocessing and preparation of LIBS data for machine learning."""
    
    @staticmethod
    def prepare_ml_data(df, target_col='origin', test_size=0.2, random_state=42):
        """
        Prepare DataFrame for machine learning by first splitting into train/test sets,
        then separating features and targets, and encoding labels
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame containing spectral data and labels
        target_col : str, default='origin'
            Name of the target column
        test_size : float, default=0.2
            Proportion of dataset to include in test split
        random_state : int, default=42
            Random state for reproducible splits
            
        Returns:
        --------
        tuple : (X_train, X_test, y_train, y_test, label_encoder, wavelength_columns)
            X_train : pandas.DataFrame - Training features
            X_test : pandas.DataFrame - Test features  
            y_train : numpy.ndarray - Encoded training labels
            y_test : numpy.ndarray - Encoded test labels
            label_encoder : LabelEncoder - Fitted label encoder for inverse transform
            wavelength_columns : list - List of wavelength column names
        """
        print("="*70)
        print("PREPARING DATA FOR MACHINE LEARNING")
        print("="*70)
        
        # Step 1: Split the complete DataFrame first
        print(f"Original dataset shape: {df.shape}")
        print(f"Target distribution:\n{df[target_col].value_counts()}")
        
        print(f"\nSplitting complete dataset (test_size={test_size}, random_state={random_state})...")
        df_train, df_test = train_test_split(
            df, test_size=test_size, random_state=random_state, 
            stratify=df[target_col]
        )
        
        print(f"Training dataset shape: {df_train.shape}")
        print(f"Test dataset shape: {df_test.shape}")
        print(f"Training target distribution:\n{df_train[target_col].value_counts()}")
        print(f"Test target distribution:\n{df_test[target_col].value_counts()}")
        
        # Step 2: Define non-feature columns
        non_feature_cols = [
            'tire_number', 'origin', 'measurement_id', 'file_name', 'timestamp',
            'sample_type', 'origin_innerliner', 'origin_sidewall', 'origin_tread'
        ]
        
        # Step 3: Identify wavelength columns
        # Get all numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter out non-feature columns
        wavelength_columns = [col for col in numeric_columns if col not in non_feature_cols]
        
        # Alternative approach if no wavelength columns found
        if len(wavelength_columns) == 0:
            potential_wavelength_cols = []
            for col in df.columns:
                if isinstance(col, (int, float)):
                    potential_wavelength_cols.append(col)
                elif isinstance(col, str):
                    try:
                        float(col)
                        potential_wavelength_cols.append(col)
                    except ValueError:
                        pass
            wavelength_columns = potential_wavelength_cols
        
        if len(wavelength_columns) == 0:
            raise ValueError("No wavelength columns found! Please check your dataset structure.")
        
        print(f"\nIdentified {len(wavelength_columns)} wavelength columns")
        
        # Step 4: Extract features and targets from training set
        print("Extracting features and targets from training set...")
        X_train = df_train[wavelength_columns].copy()
        y_train = df_train[target_col].copy()
        
        # Step 5: Extract features and targets from test set
        print("Extracting features and targets from test set...")
        X_test = df_test[wavelength_columns].copy()
        y_test = df_test[target_col].copy()
        
        # Step 6: Ensure all features are numeric
        print("Converting features to numeric...")
        for col in wavelength_columns:
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
        
        # Step 7: Handle missing and infinite values in training set
        missing_count_train = X_train.isnull().sum().sum()
        if missing_count_train > 0:
            print(f"Filling {missing_count_train} missing values in training set with mean...")
            X_train = X_train.fillna(X_train.mean())
        
        inf_count_train = np.isinf(X_train.values).sum()
        if inf_count_train > 0:
            print(f"Replacing {inf_count_train} infinite values in training set with mean...")
            X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.mean())
        
        # Step 8: Handle missing and infinite values in test set (using training set statistics)
        missing_count_test = X_test.isnull().sum().sum()
        if missing_count_test > 0:
            print(f"Filling {missing_count_test} missing values in test set with training mean...")
            # Use training set mean for test set imputation
            X_test = X_test.fillna(X_train.mean())
        
        inf_count_test = np.isinf(X_test.values).sum()
        if inf_count_test > 0:
            print(f"Replacing {inf_count_test} infinite values in test set with training mean...")
            # Use training set mean for test set imputation
            X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_train.mean())
        
        # Step 9: Encode labels
        print("Encoding labels...")
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        # Step 10: Display final results
        print(f"\nFinal results:")
        print(f"Training features shape: {X_train.shape}")
        print(f"Test features shape: {X_test.shape}")
        print(f"Label classes: {label_encoder.classes_}")
        print(f"Training label distribution: {np.bincount(y_train_encoded)}")
        print(f"Test label distribution: {np.bincount(y_test_encoded)}")
        
        return X_train, X_test, y_train_encoded, y_test_encoded, label_encoder, wavelength_columns


class LdaAnalyzer:
    """Performs Linear Discriminant Analysis on LIBS spectral data."""
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.lda = None
        self.explained_variance_ratio = None
        
    def fit_transform(self, X_train, y_train):
        """Fit LDA and transform training data"""
        if self.n_components is None:
            self.n_components = len(y_train.unique()) - 1
        
        self.lda = LinearDiscriminantAnalysis(n_components=self.n_components)
        X_train_lda = self.lda.fit_transform(X_train, y_train)
        self.explained_variance_ratio = self.lda.explained_variance_ratio_
        
        print(f"Original feature space: {X_train.shape[1]} wavelengths")
        print(f"Reduced feature space: {X_train_lda.shape[1]} LDA components")
        print(f"Dimensionality reduction: {(1-X_train_lda.shape[1]/X_train.shape[1])*100:.1f}%")
        print(f"Explained variance ratio: {self.explained_variance_ratio}")
        print(f"Total explained variance: {self.explained_variance_ratio.sum():.4f}")
        
        return X_train_lda
    
    def transform(self, X_test):
        """Transform test data using fitted LDA"""
        if self.lda is None:
            raise ValueError("LDA not fitted yet. Call fit_transform first.")
        return self.lda.transform(X_test)
    
    def visualize_components(self, X_lda, y, max_points=1000):
        """Visualize LDA components"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Sample points for visualization if too many
        if len(X_lda) > max_points:
            sample_indices = np.random.choice(len(X_lda), max_points, replace=False)
            X_vis = X_lda[sample_indices]
            y_vis = y.iloc[sample_indices] if hasattr(y, 'iloc') else y[sample_indices]
        else:
            X_vis = X_lda
            y_vis = y
        
        if self.n_components >= 2:
            # 2D scatter plot
            color_map = {'tread': 0, 'innerliner': 1, 'sidewall': 2}
            colors = [color_map[origin] for origin in y_vis]
            
            scatter = axes[0].scatter(X_vis[:, 0], X_vis[:, 1], c=colors, 
                                     cmap='viridis', alpha=0.6, s=10)
            axes[0].set_xlabel(f'LDA Component 1 ({self.explained_variance_ratio[0]:.3f})')
            axes[0].set_ylabel(f'LDA Component 2 ({self.explained_variance_ratio[1]:.3f})')
            axes[0].set_title(f'LDA Components Visualization\n({len(y_vis)} samples shown)')
            
            # Add legend
            for origin, color_idx in color_map.items():
                axes[0].scatter([], [], c=plt.cm.viridis(color_idx/2), label=origin.capitalize(), s=50)
            axes[0].legend()
        else:
            # 1D histogram
            for origin in ['tread', 'innerliner', 'sidewall']:
                mask = y_vis == origin
                axes[0].hist(X_vis[mask, 0], alpha=0.7, label=origin, bins=50)
            axes[0].set_xlabel(f'LDA Component 1 ({self.explained_variance_ratio[0]:.3f})')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title(f'LDA Component Distribution\n({len(y_vis)} samples)')
            axes[0].legend()
        
        # Explained variance plot
        component_labels = [f'Component {i+1}' for i in range(len(self.explained_variance_ratio))]
        axes[1].bar(component_labels, self.explained_variance_ratio)
        axes[1].set_xlabel('LDA Component')
        axes[1].set_ylabel('Explained Variance Ratio')
        axes[1].set_title('LDA Explained Variance')
        
        plt.tight_layout()
        plt.show()


class ModelComparator:
    """Compares different machine learning models on original and LDA-reduced features."""
    
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=10000),
            'LDA Classifier': LinearDiscriminantAnalysis()
        }
        self.results = {}
    
    def compare_models(self, X_train, X_test, y_train, y_test, 
                      X_train_lda=None, X_test_lda=None, cv_folds=5):
        """Compare models on original and LDA-reduced features"""
        
        # Test on LDA-reduced features if provided
        if X_train_lda is not None and X_test_lda is not None:
            print("--- LDA-Reduced Features ---")
            for name, model in self.models.items():
                if name == 'LDA Classifier':  # Skip redundant LDA classifier
                    continue
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_lda, y_train, cv=cv_folds, scoring='accuracy')
                
                # Test accuracy
                model.fit(X_train_lda, y_train)
                y_pred = model.predict(X_test_lda)
                test_accuracy = accuracy_score(y_test, y_pred)
                
                self.results[f"{name}_lda"] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'test_accuracy': test_accuracy,
                    'y_pred': y_pred
                }
                
                print(f"{name:20} | CV: {cv_scores.mean():.4f} (±{cv_scores.std():.4f}) | Test: {test_accuracy:.4f}")
        
        return self.results
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, feature_type=""):
        """Plot confusion matrix for a model"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['innerliner', 'sidewall', 'tread'],
                    yticklabels=['innerliner', 'sidewall', 'tread'])
        plt.title(f'Confusion Matrix - {model_name} {feature_type}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        print(f"\nClassification Report - {model_name} {feature_type}:")
        print(classification_report(y_true, y_pred))
    
    def plot_performance_comparison(self):
        """Plot performance comparison between different approaches"""
        comparison_data = []
        for key, metrics in self.results.items():
            model_name, feature_type = key.rsplit('_', 1)
            comparison_data.append({
                'Model': model_name,
                'Features': 'Original' if feature_type == 'original' else 'LDA-reduced',
                'CV_Accuracy': metrics['cv_mean'],
                'Test_Accuracy': metrics['test_accuracy']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # CV Accuracy comparison
        pivot_cv = comparison_df.pivot(index='Model', columns='Features', values='CV_Accuracy')
        pivot_cv.plot(kind='bar', ax=axes[0], rot=45)
        axes[0].set_title('Cross-Validation Accuracy Comparison')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend(title='Feature Type')
        
        # Test Accuracy comparison
        pivot_test = comparison_df.pivot(index='Model', columns='Features', values='Test_Accuracy')
        pivot_test.plot(kind='bar', ax=axes[1], rot=45)
        axes[1].set_title('Test Accuracy Comparison')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend(title='Feature Type')
        
        plt.tight_layout()
        plt.show()


def analyze_lda_interpretability(lda_analyzer, X_columns):
    """Analyze LDA interpretability by examining discriminant functions"""
    if lda_analyzer.lda.scalings_ is not None:
        print("=== LDA INTERPRETABILITY ===")
        print(f"LDA scalings shape: {lda_analyzer.lda.scalings_.shape}")
        
        # Find most important wavelengths for each LDA component
        for comp_idx in range(lda_analyzer.lda.scalings_.shape[1]):
            print(f"\nLDA Component {comp_idx + 1}:")
            component_weights = lda_analyzer.lda.scalings_[:, comp_idx]
            
            # Get indices of top contributing features
            top_indices = np.argsort(np.abs(component_weights))[-10:][::-1]
            
            print("Top 10 most discriminative wavelengths:")
            for i, idx in enumerate(top_indices):
                wavelength = X_columns[idx] if hasattr(X_columns, '__getitem__') else f'Feature_{idx}'
                weight = component_weights[idx]
                print(f"  {i+1}. {wavelength}: {weight:.4f}")
        
        # Plot discriminant functions
        plt.figure(figsize=(15, 8))
        
        for comp_idx in range(min(2, lda_analyzer.lda.scalings_.shape[1])):
            plt.subplot(2, 1, comp_idx + 1)
            plt.plot(lda_analyzer.lda.scalings_[:, comp_idx], linewidth=1)
            plt.title(f'LDA Component {comp_idx + 1} Weights (Discriminant Function)')
            plt.xlabel('Feature Index (Wavelength)')
            plt.ylabel('Weight')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def plot_single_spectrum(spectrum, wavelengths=None, title='LIBS Spectrum', 
                        xlabel='Wavelength (nm)', ylabel='Intensity (a.u.)',
                        figsize=(12, 6), color='blue', linewidth=1.5, 
                        grid=True, annotate_peaks=False, peak_threshold=None,
                        save_path=None):
    """
    Plot a single LIBS spectrum
    
    Parameters:
    -----------
    spectrum : array-like
        1D array of intensity values for the spectrum
    wavelengths : array-like, optional
        Wavelength values corresponding to the spectrum intensities.
        If None, uses feature indices.
    title : str, default='LIBS Spectrum'
        Title for the plot
    xlabel : str, default='Wavelength (nm)'
        Label for x-axis
    ylabel : str, default='Intensity (a.u.)'
        Label for y-axis
    figsize : tuple, default=(12, 6)
        Figure size (width, height) in inches
    color : str, default='blue'
        Color for the spectrum line
    linewidth : float, default=1.5
        Width of the spectrum line
    grid : bool, default=True
        Whether to show grid
    annotate_peaks : bool, default=False
        Whether to annotate the highest peaks
    peak_threshold : float, optional
        If annotate_peaks is True, only annotate peaks above this intensity value.
        If None, annotates top 5 peaks.
    save_path : str, optional
        If provided, saves the plot to this file path
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    
    Example:
    --------
    >>> # Plot with wavelengths
    >>> wavelengths = np.linspace(200, 1000, 8000)
    >>> spectrum = df.iloc[0, 3:].values  # Get first spectrum
    >>> plot_single_spectrum(spectrum, wavelengths=wavelengths, 
    ...                      title='Tread Sample #1')
    
    >>> # Plot without wavelengths (using indices)
    >>> plot_single_spectrum(spectrum, title='Sample Spectrum')
    
    >>> # Plot with peak annotations
    >>> plot_single_spectrum(spectrum, wavelengths=wavelengths,
    ...                      annotate_peaks=True, peak_threshold=5000)
    """
    # Convert to numpy array
    spectrum = np.array(spectrum)
    
    # Create x-axis values
    if wavelengths is not None:
        wavelengths = np.array(wavelengths)
        if len(wavelengths) != len(spectrum):
            raise ValueError(f"Wavelengths length ({len(wavelengths)}) must match spectrum length ({len(spectrum)})")
        x_values = wavelengths
    else:
        x_values = np.arange(len(spectrum))
        xlabel = 'Feature Index'
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot spectrum
    ax.plot(x_values, spectrum, color=color, linewidth=linewidth, alpha=0.8)
    
    # Add labels and title
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add grid
    if grid:
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # Annotate peaks if requested
    if annotate_peaks:
        if peak_threshold is not None:
            # Find peaks above threshold
            peak_indices = np.where(spectrum > peak_threshold)[0]
            # Further filter to local maxima
            peak_mask = np.zeros(len(spectrum), dtype=bool)
            for idx in peak_indices:
                if idx > 0 and idx < len(spectrum) - 1:
                    if spectrum[idx] > spectrum[idx-1] and spectrum[idx] > spectrum[idx+1]:
                        peak_mask[idx] = True
            peak_indices = np.where(peak_mask)[0]
        else:
            # Find top 5 peaks - simple approach without scipy
            # Look for local maxima
            peak_mask = np.zeros(len(spectrum), dtype=bool)
            for idx in range(1, len(spectrum) - 1):
                if spectrum[idx] > spectrum[idx-1] and spectrum[idx] > spectrum[idx+1]:
                    peak_mask[idx] = True
            peak_indices = np.where(peak_mask)[0]
            
            # If we found peaks, take top 5 by intensity
            if len(peak_indices) > 5:
                peak_intensities = spectrum[peak_indices]
                top_peaks_indices = np.argsort(peak_intensities)[-5:]
                peak_indices = peak_indices[top_peaks_indices]
            elif len(peak_indices) == 0:
                # Fallback: just take top 5 values
                peak_indices = np.argsort(spectrum)[-5:]
        
        # Annotate each peak
        for peak_idx in peak_indices:
            peak_x = x_values[peak_idx]
            peak_y = spectrum[peak_idx]
            if wavelengths is not None:
                label = f'{peak_x:.1f} nm'
            else:
                label = f'idx {peak_idx}'
            ax.annotate(label, xy=(peak_x, peak_y), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, color='red',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', color='red', lw=1))
    
    # Add statistics text box
    stats_text = f'Mean: {np.mean(spectrum):.2f}\n'
    stats_text += f'Max: {np.max(spectrum):.2f}\n'
    stats_text += f'Min: {np.min(spectrum):.2f}\n'
    stats_text += f'Std: {np.std(spectrum):.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    return fig, ax