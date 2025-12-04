import pandas as pd # type: ignore
import tensorflow as tf # type: ignore
import numpy as np # type: ignore
import h5py # type: ignore
import os # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
import gc  # Garbage collection for memory management

def create_libs_tensorflow_dataset_from_dataframe_normalized(df, label_column='origin', batch_size=32, 
                                                           prefetch_buffer=tf.data.AUTOTUNE, cache_dataset=True, 
                                                           test_size=0.2, val_size=0.2, 
                                                           random_state=42, normalize_data=True,
                                                           dimensionality_reduction='none',
                                                           pca_components=None, pca_variance_threshold=0.95,
                                                           lda_components=None, 
                                                           wavelength_ranges=[(650, 750), (950, 960)]):
    """
    Create a TensorFlow Dataset from a LIBS DataFrame with early normalization and dimensionality reduction.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing spectral data and labels
    label_column : str, default='origin'
        Name of the column containing class labels
    batch_size : int, default=32
        Batch size for training
    prefetch_buffer : tf.data.AUTOTUNE, default=tf.data.AUTOTUNE
        Prefetch buffer size
    cache_dataset : bool, default=True
        Whether to cache the dataset in memory
    test_size : float, default=0.2
        Proportion of data to use for testing (0.0-1.0)
    val_size : float, default=0.2
        Proportion of remaining data to use for validation (0.0-1.0)
    random_state : int, default=42
        Random state for reproducible splits
    normalize_data : bool, default=True
        Whether to apply SNV normalization
    dimensionality_reduction : str, default='none'
        Type of dimensionality reduction to apply:
        - 'none': No reduction
        - 'pca': Principal Component Analysis
        - 'lda': Linear Discriminant Analysis
        - 'wavelength': Select specific wavelength ranges
        - 'nist' : Wavelengths selected based on NIST emmision records corresponding to the elements present in car tires (C, H, Al, Fe, etc) 
    pca_components : int, optional
        Number of PCA components. If None, uses pca_variance_threshold
    pca_variance_threshold : float, default=0.95
        Variance threshold for automatic PCA component selection
    lda_components : int, optional
        Number of LDA components. If None, uses maximum possible (n_classes-1)
    wavelength_ranges : list, default=[(650, 750), (950, 960)]
        List of wavelength ranges (min, max) to select for 'wavelength' reduction
    
    Returns:
    --------
    tuple : (train_dataset, val_dataset, test_dataset, label_encoder, dataset_info)
    """
    import gc
    from utils.normalization_functions import Normalizer
    from utils.pca_analysis_functions import PCAAnalyzer
    from utils.lda_analysis_functions import LDAAnalyzer
    from sklearn.decomposition import PCA #type: ignore
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #type: ignore

    print(f"📊 Processing DataFrame with {len(df):,} samples...")
    print(f"🔧 Dimensionality reduction: {dimensionality_reduction.upper()}")
    
    # Validate input DataFrame
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in DataFrame. Available columns: {list(df.columns)}")
    
    # Identify wavelength columns (assuming they are numeric column names)
    wavelength_columns = [col for col in df.columns if col != label_column and str(col).replace('.', '').isdigit()]
    
    if not wavelength_columns:
        # If no numeric columns found, assume all columns except label_column are wavelengths
        wavelength_columns = [col for col in df.columns if col != label_column]
    
    print(f"Found {len(wavelength_columns)} wavelength features")
    print(f"Wavelength range: {min([float(col) for col in wavelength_columns]):.1f} - {max([float(col) for col in wavelength_columns]):.1f} nm")
    
    # Step 1: Extract X and y first
    print("🔄 Extracting spectral data and labels...")
    X = df[wavelength_columns].values.astype(np.float32)
    y_str = df[label_column].values
    
    # Clear DataFrame from memory if possible
    del df
    gc.collect()
    
    # Step 2: Encode labels early
    print("🔄 Encoding labels...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str).astype(np.int8)
    
    print(f"Label encoding: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    # Display class distribution
    unique_labels, counts = np.unique(y, return_counts=True)
    print("Class distribution:")
    for label_idx, count in zip(unique_labels, counts):
        class_name = label_encoder.classes_[label_idx]
        percentage = (count / len(y)) * 100
        print(f"  {class_name}: {count:,} samples ({percentage:.1f}%)")
    
    # Clear string labels
    del y_str
    gc.collect()
    
    # Step 3: Normalize X before dimensionality reduction (more memory efficient)
    if normalize_data:
        print("🔄 Normalizing spectral data with SNV...")
        normalizer = Normalizer()
        
        # Process in chunks to avoid memory issues
        chunk_size = 10000  # Process 10k samples at a time
        n_chunks = (len(X) + chunk_size - 1) // chunk_size
        
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(X))
            
            if i % 10 == 0:
                print(f"  Normalizing chunk {i+1}/{n_chunks} ({start_idx:,}-{end_idx:,})")
            
            # Apply SNV normalization to chunk
            X[start_idx:end_idx] = normalizer.apply_standard_normal_variate_on_dataset(
                X[start_idx:end_idx]
            )
            
            if i % 20 == 0:  # Periodic garbage collection
                gc.collect()
        
        print("✅ Normalization completed")
    
    # Step 4: Apply dimensionality reduction
    original_features = X.shape[1]
    reduction_info = {'method': dimensionality_reduction}
    
    if dimensionality_reduction.lower() == 'pca':
        print("🔄 Applying PCA dimensionality reduction...")
        
        if pca_components is None:
            # Determine components based on variance threshold
            pca_temp = PCA()
            pca_temp.fit(X)
            cumsum_variance = np.cumsum(pca_temp.explained_variance_ratio_)
            pca_components = np.argmax(cumsum_variance >= pca_variance_threshold) + 1
            print(f"  Auto-selected {pca_components} components for {pca_variance_threshold*100:.1f}% variance")
        
        pca = PCA(n_components=pca_components)
        X = pca.fit_transform(X)
        
        reduction_info.update({
            'n_components': pca_components,
            'variance_explained': np.sum(pca.explained_variance_ratio_),
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'pca_model': pca
        })
        
        print(f"  ✅ PCA: {original_features} → {pca_components} features")
        print(f"  📊 Variance explained: {reduction_info['variance_explained']:.3f}")
        
    elif dimensionality_reduction.lower() == 'lda':
        print("🔄 Applying LDA dimensionality reduction...")
        
        n_classes = len(np.unique(y))
        max_lda_components = n_classes - 1
        
        if lda_components is None:
            lda_components = max_lda_components
            print(f"  Auto-selected {lda_components} components (max for {n_classes} classes)")
        else:
            lda_components = min(lda_components, max_lda_components)
            print(f"  Using {lda_components} components (limited by {n_classes} classes)")
        
        lda = LinearDiscriminantAnalysis(n_components=lda_components)
        X = lda.fit_transform(X, y)
        
        reduction_info.update({
            'n_components': lda_components,
            'max_possible_components': max_lda_components,
            'variance_explained': np.sum(lda.explained_variance_ratio_),
            'explained_variance_ratio': lda.explained_variance_ratio_,
            'lda_model': lda
        })
        
        print(f"  ✅ LDA: {original_features} → {lda_components} features")
        print(f"  📊 Variance explained: {reduction_info['variance_explained']:.3f}")
        
    elif dimensionality_reduction.lower() == 'wavelength':
        print("🔄 Applying wavelength range selection...")
        
        # Convert wavelength column names to float for comparison
        wavelength_values = [float(col) for col in wavelength_columns]
        selected_indices = []
        selected_wavelengths = []
        
        # Select wavelengths within specified ranges
        for i, wl in enumerate(wavelength_values):
            for wl_min, wl_max in wavelength_ranges:
                if wl_min <= wl <= wl_max:
                    selected_indices.append(i)
                    selected_wavelengths.append(wl)
                    break
        
        if not selected_indices:
            raise ValueError(f"No wavelengths found in specified ranges: {wavelength_ranges}")
        
        # Apply selection
        X = X[:, selected_indices]
        
        reduction_info.update({
            'wavelength_ranges': wavelength_ranges,
            'selected_wavelengths': selected_wavelengths,
            'selected_indices': selected_indices,
            'n_wavelengths': len(selected_wavelengths)
        })
        
        # Show range statistics
        for wl_min, wl_max in wavelength_ranges:
            range_wavelengths = [wl for wl in selected_wavelengths if wl_min <= wl <= wl_max]
            print(f"  Range {wl_min}-{wl_max}nm: {len(range_wavelengths)} wavelengths")
        
        print(f"  ✅ Wavelength selection: {original_features} → {len(selected_wavelengths)} features")
        print(f"  📊 Feature reduction: {(1 - len(selected_wavelengths)/original_features)*100:.1f}%")

    elif dimensionality_reduction.lower() == 'nist':
        print("🔄 Applying NIST wavelength selection...")
    
        nist_df = pd.read_csv('../data/csv/nist_top_50_percent_wavelengths.csv')
        
        # Extract all unique wavelengths from nist_df
        nist_wavelengths_set = set()
        for column in nist_df.columns:
            # Get wavelengths from this element column, drop NaN values
            element_wavelengths = nist_df[column].dropna().tolist()
            nist_wavelengths_set.update([float(wl) for wl in element_wavelengths if pd.notna(wl)])
        
        print(f"  Found {len(nist_wavelengths_set)} unique NIST wavelengths")
        print(f"  NIST wavelength range: {min(nist_wavelengths_set):.1f} - {max(nist_wavelengths_set):.1f} nm")
        
        # Convert wavelength column names to float for comparison
        wavelength_values = [float(col) for col in wavelength_columns]
        selected_indices = []
        selected_wavelengths = []
        
        # Find wavelengths to keep (with tolerance for matching)
        tolerance = 0.1  # 0.1 nm tolerance for floating point comparison
        
        for i, wl in enumerate(wavelength_values):
            # Check if this wavelength is close to any NIST wavelength
            for nist_wl in nist_wavelengths_set:
                if abs(wl - nist_wl) <= tolerance:
                    selected_indices.append(i)
                    selected_wavelengths.append(wl)
                    break
        
        if not selected_indices:
            raise ValueError("No wavelengths found matching NIST database within tolerance")
        
        # Apply NIST wavelength selection
        X = X[:, selected_indices]
        
        # Count wavelengths per element for reporting
        element_counts = {}
        for column in nist_df.columns:
            element_name = column.replace('_data', '')
            element_wavelengths = nist_df[column].dropna().tolist()
            element_wavelengths_float = [float(wl) for wl in element_wavelengths if pd.notna(wl)]
            
            # Count how many of this element's wavelengths were selected
            element_selected = 0
            for selected_wl in selected_wavelengths:
                for element_wl in element_wavelengths_float:
                    if abs(selected_wl - element_wl) <= tolerance:
                        element_selected += 1
                        break
            element_counts[element_name] = element_selected
        
        reduction_info.update({
            'nist_wavelengths_available': len(nist_wavelengths_set),
            'nist_wavelengths_matched': len(selected_wavelengths),
            'selected_wavelengths': selected_wavelengths,
            'selected_indices': selected_indices,
            'wavelength_tolerance': tolerance,
            'element_wavelength_counts': element_counts,
            'nist_wavelength_range': (min(nist_wavelengths_set), max(nist_wavelengths_set))
        })
        
        # Display element-wise statistics
        print(f"  ✅ NIST selection: {original_features} → {len(selected_wavelengths)} features")
        print(f"  📊 Feature reduction: {(1 - len(selected_wavelengths)/original_features)*100:.1f}%")
        print(f"  🎯 Matched {len(selected_wavelengths)}/{len(nist_wavelengths_set)} NIST wavelengths")
        
        # Show wavelengths per element
        print("  📋 Wavelengths selected per element:")
        for element, count in element_counts.items():
            if count > 0:
                print(f"     {element}: {count} wavelengths")
        
        # Show some example selected wavelengths
        if len(selected_wavelengths) <= 10:
            print(f"  📍 Selected wavelengths: {[f'{wl:.1f}' for wl in sorted(selected_wavelengths)]}")
        else:
            sorted_wl = sorted(selected_wavelengths)
            print(f"  📍 Selected wavelengths (sample): {[f'{wl:.1f}' for wl in sorted_wl[:5]]} ... {[f'{wl:.1f}' for wl in sorted_wl[-5:]]}")
        


    elif dimensionality_reduction.lower() != 'none':
        raise ValueError(f"Unknown dimensionality reduction method: {dimensionality_reduction}. "
                        "Choose from: 'none', 'pca', 'lda', 'wavelength' or 'nist'")
    
    else:
        print("🔄 No dimensionality reduction applied")
        reduction_info['n_components'] = original_features
    
    # Step 5: Split reduced data into train/test/val
    print("🔄 Splitting data...")
    if test_size > 0:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_temp, y_temp = X, y
        X_test, y_test = np.array([]), np.array([])
    
    if val_size > 0 and len(X_temp) > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
        )
    else:
        X_train, y_train = X_temp, y_temp
        X_val, y_val = np.array([]), np.array([])
    
    # Clear temporary variables
    del X, y, X_temp, y_temp
    gc.collect()
    
    print(f"Train set: {X_train.shape[0]:,} samples")
    if len(X_val) > 0:
        print(f"Validation set: {X_val.shape[0]:,} samples")
    if len(X_test) > 0:
        print(f"Test set: {X_test.shape[0]:,} samples")
    
    # Step 6: Create TensorFlow datasets
    print("🔄 Creating TensorFlow datasets...")
    
    def create_tf_dataset(X, y, batch_size, is_training=True):
        """Create a TensorFlow dataset with optimizations"""
        if len(X) == 0:
            return None
            
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        if is_training:
            dataset = dataset.shuffle(buffer_size=min(10000, len(X)))
        
        dataset = dataset.batch(batch_size)
        
        if cache_dataset:
            dataset = dataset.cache()
        
        dataset = dataset.prefetch(prefetch_buffer)
        
        return dataset
    
    # Create datasets
    train_dataset = create_tf_dataset(X_train, y_train, batch_size, is_training=True)
    val_dataset = create_tf_dataset(X_val, y_val, batch_size, is_training=False) if len(X_val) > 0 else None
    test_dataset = create_tf_dataset(X_test, y_test, batch_size, is_training=False) if len(X_test) > 0 else None
    
    # Dataset information
    dataset_info = {
        'total_samples': len(X_train) + len(X_val) + len(X_test),
        'train_samples': len(X_train),
        'val_samples': len(X_val) if len(X_val) > 0 else 0,
        'test_samples': len(X_test) if len(X_test) > 0 else 0,
        'n_features': X_train.shape[1],
        'original_n_features': original_features,
        'n_classes': len(label_encoder.classes_),
        'class_names': label_encoder.classes_.tolist(),
        'feature_shape': X_train.shape[1:],
        'steps_per_epoch': len(X_train) // batch_size if len(X_train) > 0 else 0,
        'validation_steps': (len(X_val) // batch_size) if len(X_val) > 0 else 0,
        'wavelength_range': (min([float(col) for col in wavelength_columns]), 
                           max([float(col) for col in wavelength_columns])),
        'label_column': label_column,
        'normalized': normalize_data,
        'data_type': str(X_train.dtype),
        'dimensionality_reduction': reduction_info
    }
    
    print(f"\n🎯 TensorFlow Dataset created successfully!")
    print(f"   • Total samples: {dataset_info['total_samples']:,}")
    print(f"   • Features per sample: {dataset_info['n_features']:,} (reduced from {original_features:,})")
    if dataset_info['n_features'] != original_features:
        print(f"   • Feature reduction: {(1 - dataset_info['n_features']/original_features)*100:.1f}%")
    print(f"   • Classes: {dataset_info['n_classes']} ({', '.join(dataset_info['class_names'])})")
    print(f"   • Original wavelength range: {dataset_info['wavelength_range'][0]:.1f} - {dataset_info['wavelength_range'][1]:.1f} nm")
    print(f"   • Data normalized: {dataset_info['normalized']}")
    print(f"   • Dimensionality reduction: {dimensionality_reduction.upper()}")
    print(f"   • Data type: {dataset_info['data_type']}")
    print(f"   • Steps per epoch: {dataset_info['steps_per_epoch']}")
    if dataset_info['validation_steps'] > 0:
        print(f"   • Validation steps: {dataset_info['validation_steps']}")
    
    # Memory usage report
    memory_usage = (X_train.nbytes + X_val.nbytes + X_test.nbytes + y_train.nbytes + y_val.nbytes + y_test.nbytes) / 1024**2
    print(f"   • Memory usage: ~{memory_usage:.1f} MB")
    
    return train_dataset, val_dataset, test_dataset, label_encoder, dataset_info