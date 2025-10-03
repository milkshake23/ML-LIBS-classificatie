import tensorflow as tf # type: ignore
import numpy as np # type: ignore
import h5py # type: ignore
import os # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
import gc  # Garbage collection for memory management

# Import the baseline correction functions
import sys
sys.path.append('..')  # Add parent directory to path
try:
    from baseline_correction_functions import BaselineCorrector
    BASELINE_CORRECTION_AVAILABLE = True
except ImportError:
    print("Warning: baseline_correction_functions not found. Falling back to basic AsLS.")
    BASELINE_CORRECTION_AVAILABLE = False
    
    # Fallback basic AsLS implementation
    from scipy import sparse # type: ignore
    from scipy.sparse.linalg import spsolve # type: ignore
    
    def baseline_als_basic(y, lam=1e5, p=0.001, niter=10):
        """Basic AsLS baseline correction fallback"""
        L = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        w = np.ones(L)
        
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        
        return z

def create_libs_tensorflow_dataset(data_directory='../data', batch_size=32, prefetch_buffer=tf.data.AUTOTUNE, 
                                   cache_dataset=True, apply_baseline_correction=False, max_files=None,
                                   baseline_method='hybrid', baseline_params=None):
    """
    Create a TensorFlow Dataset from LIBS HDF5 files for efficient processing of large datasets.
    
    Parameters:
    - data_directory: Path to directory containing HDF5 files
    - batch_size: Batch size for training
    - prefetch_buffer: Prefetch buffer size (use tf.data.AUTOTUNE for automatic)
    - cache_dataset: Whether to cache the dataset in memory
    - apply_baseline_correction: Whether to apply baseline correction
    - max_files: Maximum number of files to process (None for all files)
    - baseline_method: Baseline correction method ('als', '4s', 'hybrid')
    - baseline_params: Dictionary of parameters for baseline correction
    
    Returns:
    - train_dataset: TensorFlow Dataset for training
    - val_dataset: TensorFlow Dataset for validation  
    - test_dataset: TensorFlow Dataset for testing
    - label_encoder: Fitted label encoder
    - dataset_info: Dictionary with dataset information
    """
    
    # Set default baseline correction parameters
    if baseline_params is None:
        if baseline_method == 'hybrid':
            baseline_params = {
                'window_length': 51,
                'polyorder': 3,
                'als_lam': 1e4,
                'als_p': 0.01
            }
        elif baseline_method == 'als':
            baseline_params = {
                'lam': 1e4,
                'p': 0.01,
                'niter': 10
            }
        elif baseline_method == '4s':
            baseline_params = {
                'window_length': 51,
                'polyorder': 3,
                'iterations': 4,
                'threshold_factor': 0.1,
                'fill_factor': 0.8
            }
        else:
            baseline_params = {}
    
    print("📂 Scanning HDF5 files...")
    all_files = [f for f in os.listdir(data_directory) if f.endswith('.h5')]
    if max_files:
        all_files = all_files[:max_files]
    
    print(f"Found {len(all_files)} HDF5 files to process")
    
    if apply_baseline_correction:
        if BASELINE_CORRECTION_AVAILABLE:
            print(f"🔧 Baseline correction enabled: {baseline_method.upper()} method")
            print(f"   Parameters: {baseline_params}")
            # Initialize baseline corrector
            corrector = BaselineCorrector()
        else:
            print("🔧 Baseline correction enabled: Basic AsLS method (fallback)")
            print(f"   Parameters: {baseline_params}")
    
    # Initialize data containers
    all_spectra = []
    all_labels = []
    all_metadata = []
    
    def extract_tire_number(filename):
        """Extract tire number from filename"""
        import re
        match = re.search(r'_B(\d+)_', filename)
        if match:
            return int(match.group(1))
        else:
            match = re.search(r'nr-(\d+)', filename)
            if match:
                return int(match.group(1))
        return None
    
    def apply_baseline_correction_to_spectrum(spectrum, method, params, corrector_instance=None):
        """Apply baseline correction to a single spectrum"""
        try:
            if BASELINE_CORRECTION_AVAILABLE and corrector_instance is not None:
                # Use the advanced baseline correction methods
                if method == 'hybrid':
                    baseline = corrector_instance.baseline_hybrid_4s_als(
                        spectrum, 
                        window_length=params.get('window_length', 51),
                        polyorder=params.get('polyorder', 3),
                        als_lam=params.get('als_lam', 1e4),
                        als_p=params.get('als_p', 0.01)
                    )
                elif method == 'als':
                    baseline = corrector_instance.baseline_als(
                        spectrum,
                        lam=params.get('lam', 1e4),
                        p=params.get('p', 0.01),
                        niter=params.get('niter', 10)
                    )
                elif method == '4s':
                    baseline = corrector_instance.baseline_4s_peak_filling(
                        spectrum,
                        window_length=params.get('window_length', 51),
                        polyorder=params.get('polyorder', 3),
                        iterations=params.get('iterations', 4),
                        threshold_factor=params.get('threshold_factor', 0.1),
                        fill_factor=params.get('fill_factor', 0.8)
                    )
                else:
                    raise ValueError(f"Unknown baseline method: {method}")
            else:
                # Fallback to basic AsLS
                baseline = baseline_als_basic(
                    spectrum,
                    lam=params.get('lam', 1e4),
                    p=params.get('p', 0.01),
                    niter=params.get('niter', 10)
                )
            
            return spectrum - baseline
            
        except Exception as e:
            print(f"    Warning: Baseline correction failed: {e}")
            return spectrum  # Return original spectrum on failure
    
    print("🔄 Processing HDF5 files...")
    
    for i, filename in enumerate(all_files):
        if i % 10 == 0:
            print(f"  Processing file {i+1}/{len(all_files)}: {filename}")
        
        try:
            filepath = os.path.join(data_directory, filename)
            tire_number = extract_tire_number(filename)
            
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
            
            # Load HDF5 file
            with h5py.File(filepath, 'r') as h5:
                intensity = h5['intensity'][:]
                wavelength = h5['wavelength'][:]
                
                # Handle different intensity shapes
                if intensity.ndim == 2:
                    # Multiple measurements (254, 8188)
                    n_measurements, n_wavelengths = intensity.shape
                    
                    for measurement_idx in range(n_measurements):
                        spectrum = intensity[measurement_idx, :].astype(np.float32)
                        
                        # Apply baseline correction if requested
                        if apply_baseline_correction:
                            corrector_to_use = corrector if BASELINE_CORRECTION_AVAILABLE else None
                            spectrum = apply_baseline_correction_to_spectrum(
                                spectrum, baseline_method, baseline_params, corrector_to_use
                            )
                        
                        all_spectra.append(spectrum)
                        all_labels.append(origin)
                        all_metadata.append({
                            'tire_number': tire_number,
                            'measurement_id': measurement_idx,
                            'filename': filename
                        })
                else:
                    # Single measurement
                    spectrum = intensity.astype(np.float32)
                    
                    if apply_baseline_correction:
                        corrector_to_use = corrector if BASELINE_CORRECTION_AVAILABLE else None
                        spectrum = apply_baseline_correction_to_spectrum(
                            spectrum, baseline_method, baseline_params, corrector_to_use
                        )
                    
                    all_spectra.append(spectrum)
                    all_labels.append(origin)
                    all_metadata.append({
                        'tire_number': tire_number,
                        'measurement_id': 0,
                        'filename': filename
                    })
        
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            continue
    
    print(f"\n✅ Data loading completed!")
    print(f"Total samples: {len(all_spectra):,}")
    print(f"Spectrum shape: {all_spectra[0].shape}")
    
    # Convert to numpy arrays
    print("🔄 Converting to numpy arrays...")
    X = np.array(all_spectra, dtype=np.float32)
    y_str = np.array(all_labels)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str).astype(np.int32)
    
    print(f"Label encoding: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    # Clear memory
    del all_spectra
    gc.collect()
    
    # Split data
    print("🔄 Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp  # 0.25 x 0.8 = 0.2 for validation
    )
    
    print(f"Train set: {X_train.shape[0]:,} samples")
    print(f"Validation set: {X_val.shape[0]:,} samples") 
    print(f"Test set: {X_test.shape[0]:,} samples")
    
    # Create TensorFlow datasets
    print("🔄 Creating TensorFlow datasets...")
    
    def create_tf_dataset(X, y, batch_size, is_training=True):
        """Create a TensorFlow dataset with optimizations"""
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        if is_training:
            dataset = dataset.shuffle(buffer_size=min(10000, len(X)))
        
        dataset = dataset.batch(batch_size)
        
        if cache_dataset:
            dataset = dataset.cache()
        
        if is_training:
            dataset = dataset.repeat()
        
        dataset = dataset.prefetch(prefetch_buffer)
        
        return dataset
    
    # Create datasets
    train_dataset = create_tf_dataset(X_train, y_train, batch_size, is_training=True)
    val_dataset = create_tf_dataset(X_val, y_val, batch_size, is_training=False)
    test_dataset = create_tf_dataset(X_test, y_test, batch_size, is_training=False)
    
    # Dataset information
    dataset_info = {
        'total_samples': len(X),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'n_features': X.shape[1],
        'n_classes': len(label_encoder.classes_),
        'class_names': label_encoder.classes_.tolist(),
        'feature_shape': X.shape[1:],
        'steps_per_epoch': len(X_train) // batch_size,
        'validation_steps': len(X_val) // batch_size,
        'baseline_corrected': apply_baseline_correction,
        'baseline_method': baseline_method if apply_baseline_correction else None,
        'baseline_params': baseline_params if apply_baseline_correction else None
    }
    
    print(f"\n🎯 TensorFlow Dataset created successfully!")
    print(f"   • Total samples: {dataset_info['total_samples']:,}")
    print(f"   • Features per sample: {dataset_info['n_features']:,}")
    print(f"   • Classes: {dataset_info['n_classes']} ({', '.join(dataset_info['class_names'])})")
    print(f"   • Baseline corrected: {dataset_info['baseline_corrected']}")
    if apply_baseline_correction:
        print(f"   • Baseline method: {dataset_info['baseline_method']}")
        print(f"   • Method parameters: {dataset_info['baseline_params']}")
    print(f"   • Steps per epoch: {dataset_info['steps_per_epoch']}")
    
    return train_dataset, val_dataset, test_dataset, label_encoder, dataset_info

try:
    # Create TensorFlow datasets with Hybrid 4S+AsLS baseline correction
    train_ds, val_ds, test_ds, label_enc, info = create_libs_tensorflow_dataset(
        data_directory='data',
        batch_size=32,
        max_files=None,  # Process all files
        apply_baseline_correction=True,  # Enable baseline correction
        baseline_method='hybrid',  # Use Hybrid 4S+AsLS method
        baseline_params={  # Custom parameters for hybrid method
            'window_length': 51,
            'polyorder': 3,
            'als_lam': 1e4,
            'als_p': 0.01
        },
        cache_dataset=True
    )
    
    print(f"\n📊 Dataset Information:")
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Demonstrate dataset usage
    print(f"\n🔍 Sample batch inspection:")
    for batch_x, batch_y in train_ds.take(1):
        print(f"   Batch shape: {batch_x.shape}")
        print(f"   Label shape: {batch_y.shape}")
        print(f"   Data type: {batch_x.dtype}")
        print(f"   Label range: {tf.reduce_min(batch_y)} to {tf.reduce_max(batch_y)}")
        print(f"   Intensity range: {tf.reduce_min(batch_x):.3f} to {tf.reduce_max(batch_x):.3f}")
        break
    
    print(f"\n✅ TensorFlow Dataset ready for training with Hybrid 4S+AsLS baseline correction!")
    
except Exception as e:
    print(f"❌ Error creating TensorFlow dataset: {e}")
    print("Make sure TensorFlow and scipy are installed:")
    print("  pip install tensorflow scipy scikit-learn")
    import traceback
    traceback.print_exc()