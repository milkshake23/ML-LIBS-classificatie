"""
Utility functions to measure and report model training details (Memory usage, Training time and Inference time).
"""

"""
Utility functions to measure and report model training details (Memory usage, Training time and Inference time).
"""

import psutil # type: ignore[import]
import time
import sys
import gc
import numpy as np
from sklearn.utils import check_array # type: ignore[import]
from contextlib import contextmanager
from functools import wraps

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def monitor_memory_usage(interval=1):
    """Monitor memory usage at regular intervals."""
    try:
        while True:
            mem_usage = get_memory_usage()
            print(f"Current memory usage: {mem_usage:.2f} MB")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Memory monitoring stopped.")

# ============================================================================
# TRAINING TIME MEASUREMENT FUNCTIONS
# ============================================================================

def measure_training_time(model, X_train, y_train, n_runs=3):
    """Measure training time and return trained model"""
    times = []
    memory_increases = []
    
    for _ in range(n_runs):
        # Memory before training
        memory_before = get_memory_usage()
        
        # Time the training
        start_time = time.perf_counter()
        trained_model = model.fit(X_train, y_train)  # Make sure to capture the trained model
        end_time = time.perf_counter()
        
        # Memory after training
        memory_after = get_memory_usage()
        
        times.append(end_time - start_time)
        memory_increases.append(memory_after - memory_before)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'memory_increase_mean': np.mean(memory_increases),
        'trained_model': trained_model  # Return the last trained model
    }

def measure_gridsearch_time(grid_search, X_train, y_train, fit_params=None):
    """
    Measure time for GridSearchCV training.
    
    Parameters:
    -----------
    grid_search : GridSearchCV
        The grid search object
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    fit_params : dict, optional
        Additional parameters to pass to fit method
        
    Returns:
    --------
    dict : GridSearch timing and results
    """
    if fit_params is None:
        fit_params = {}
    
    # Memory before grid search
    gc.collect()
    memory_before = get_memory_usage()
    
    # Time the grid search
    start_time = time.perf_counter()
    grid_search.fit(X_train, y_train, **fit_params)
    end_time = time.perf_counter()
    
    # Memory after grid search
    memory_after = get_memory_usage()
    
    total_time = end_time - start_time
    memory_increase = memory_after - memory_before
    
    # Calculate average time per CV fold and parameter combination
    n_params = len(grid_search.cv_results_['params'])
    cv_folds = grid_search.cv if hasattr(grid_search, 'cv') else 5
    if hasattr(cv_folds, 'get_n_splits'):
        try:
            n_folds = cv_folds.get_n_splits(X_train, y_train)
        except:
            n_folds = 5
    else:
        n_folds = cv_folds if isinstance(cv_folds, int) else 5
    
    total_fits = n_params * n_folds
    avg_time_per_fit = total_time / total_fits if total_fits > 0 else total_time
    
    return {
        'total_time': total_time,
        'memory_before': memory_before,
        'memory_after': memory_after,
        'memory_increase': memory_increase,
        'n_parameter_combinations': n_params,
        'n_cv_folds': n_folds,
        'total_fits': total_fits,
        'avg_time_per_fit': avg_time_per_fit,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_estimator': grid_search.best_estimator_
    }

@contextmanager
def training_timer(model_name="Model"):
    """
    Context manager to time training operations.
    
    Usage:
    ------
    with training_timer("Logistic Regression"):
        model.fit(X_train, y_train)
    """
    print(f"🚀 Starting training: {model_name}")
    
    memory_before = get_memory_usage()
    start_time = time.perf_counter()
    
    try:
        yield
    finally:
        end_time = time.perf_counter()
        memory_after = get_memory_usage()
        
        training_time = end_time - start_time
        memory_increase = memory_after - memory_before
        
        print(f"✅ Training completed: {model_name}")
        print(f"   Training time: {training_time:.4f} seconds")
        print(f"   Memory increase: {memory_increase:.2f} MB")

def training_time_decorator(model_name=None):
    """
    Decorator to automatically measure training time for model.fit() calls.
    
    Usage:
    ------
    @training_time_decorator("My Model")
    def train_model():
        model.fit(X_train, y_train)
        return model
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = model_name or func.__name__
            
            memory_before = get_memory_usage()
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                
                end_time = time.perf_counter()
                memory_after = get_memory_usage()
                
                training_time = end_time - start_time
                memory_increase = memory_after - memory_before
                
                print(f"\n⏱️  Training Time Report: {name}")
                print(f"   Duration: {training_time:.4f} seconds")
                print(f"   Memory increase: {memory_increase:.2f} MB")
                
                return result
                
            except Exception as e:
                end_time = time.perf_counter()
                training_time = end_time - start_time
                print(f"\n❌ Training failed: {name}")
                print(f"   Time before failure: {training_time:.4f} seconds")
                raise e
                
        return wrapper
    return decorator

def measure_incremental_training_time(model, X_train, y_train, batch_size=100, n_batches=None):
    """
    Measure training time for incremental/online learning models.
    
    Parameters:
    -----------
    model : sklearn estimator with partial_fit method
        The incremental learning model
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    batch_size : int, default=100
        Size of each training batch
    n_batches : int, optional
        Number of batches to process (if None, use all data)
        
    Returns:
    --------
    dict : Incremental training statistics
    """
    if not hasattr(model, 'partial_fit'):
        raise ValueError("Model must have partial_fit method for incremental training")
    
    n_samples = len(X_train)
    if n_batches is None:
        n_batches = (n_samples + batch_size - 1) // batch_size
    
    batch_times = []
    cumulative_times = []
    memory_usage = []
    
    # Get unique classes for partial_fit
    classes = np.unique(y_train)
    
    start_total = time.perf_counter()
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        
        X_batch = X_train[start_idx:end_idx]
        y_batch = y_train[start_idx:end_idx]
        
        # Memory before batch
        memory_before = get_memory_usage()
        
        # Time this batch
        batch_start = time.perf_counter()
        
        if batch_idx == 0:
            # First batch needs classes parameter
            model.partial_fit(X_batch, y_batch, classes=classes)
        else:
            model.partial_fit(X_batch, y_batch)
        
        batch_end = time.perf_counter()
        
        # Record metrics
        batch_time = batch_end - batch_start
        cumulative_time = batch_end - start_total
        memory_after = get_memory_usage()
        
        batch_times.append(batch_time)
        cumulative_times.append(cumulative_time)
        memory_usage.append(memory_after)
        
        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            print(f"Batch {batch_idx + 1}/{n_batches}: {batch_time:.4f}s "
                  f"(Total: {cumulative_time:.4f}s, Memory: {memory_after:.1f}MB)")
    
    total_time = time.perf_counter() - start_total
    
    return {
        'total_time': total_time,
        'n_batches': n_batches,
        'batch_size': batch_size,
        'batch_times': batch_times,
        'cumulative_times': cumulative_times,
        'memory_usage': memory_usage,
        'mean_batch_time': np.mean(batch_times),
        'std_batch_time': np.std(batch_times),
        'samples_per_second': n_samples / total_time,
        'final_memory': memory_usage[-1],
        'memory_growth': memory_usage[-1] - memory_usage[0] if memory_usage else 0
    }

def compare_training_times(models_dict, X_train, y_train, n_runs=1):
    """
    Compare training times across multiple models.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary with model_name: model_instance pairs
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    n_runs : int, default=1
        Number of training runs to average over
        
    Returns:
    --------
    dict : Training time comparison results
    """
    results = {}
    
    print(f"\n🏁 Training Time Comparison ({n_runs} runs)")
    print("=" * 60)
    
    for model_name, model in models_dict.items():
        print(f"\n🚀 Training {model_name}...")
        
        timing_results = measure_training_time(model, X_train, y_train, n_runs=n_runs)
        results[model_name] = timing_results
        
        print(f"   Mean time: {timing_results['mean_time']:.4f} ± {timing_results['std_time']:.4f}s")
        print(f"   Memory increase: {timing_results['memory_increase_mean']:.2f}MB")
    
    # Print comparison summary
    print(f"\n📊 Training Time Summary:")
    print("-" * 40)
    
    # Sort by mean training time
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_time'])
    
    for model_name, timing_results in sorted_results:
        mean_time = timing_results['mean_time']
        memory_increase = timing_results['memory_increase_mean']
        print(f"   {model_name:20}: {mean_time:8.4f}s ({memory_increase:6.2f}MB)")
    
    # Find fastest model
    fastest_model = sorted_results[0]
    print(f"\n🏆 Fastest model: {fastest_model[0]} ({fastest_model[1]['mean_time']:.4f}s)")
    
    return results

def print_training_report(timing_results, model_name="Model"):
    """Print formatted training time report."""
    print(f"\n📊 {model_name} Training Report:")
    print("-" * 45)
    
    if 'mean_time' in timing_results:
        mean_time = timing_results['mean_time']
        std_time = timing_results.get('std_time', 0)
        print(f"   Training Time (mean ± std): {mean_time:.4f} ± {std_time:.4f} seconds")
        
        if 'min_time' in timing_results and 'max_time' in timing_results:
            print(f"   Training Time (min/max): {timing_results['min_time']:.4f} / {timing_results['max_time']:.4f} seconds")
    
    if 'memory_increase_mean' in timing_results:
        mem_increase = timing_results['memory_increase_mean']
        mem_std = timing_results.get('memory_increase_std', 0)
        print(f"   Memory Increase: {mem_increase:.2f} ± {mem_std:.2f} MB")
    
    # Special handling for GridSearch results
    if 'total_fits' in timing_results:
        print(f"   Total Parameter Combinations: {timing_results['n_parameter_combinations']}")
        print(f"   CV Folds: {timing_results['n_cv_folds']}")
        print(f"   Total Model Fits: {timing_results['total_fits']}")
        print(f"   Average Time per Fit: {timing_results['avg_time_per_fit']:.4f} seconds")
        print(f"   Best Score: {timing_results['best_score']:.4f}")
    
    # Special handling for incremental learning
    if 'n_batches' in timing_results:
        print(f"   Number of Batches: {timing_results['n_batches']}")
        print(f"   Batch Size: {timing_results['batch_size']}")
        print(f"   Mean Batch Time: {timing_results['mean_batch_time']:.4f} seconds")
        print(f"   Samples per Second: {timing_results['samples_per_second']:.0f}")
        print(f"   Memory Growth: {timing_results['memory_growth']:.2f} MB")

def estimate_training_time(model, X_sample, y_sample, full_data_size, sample_size=1000):
    """
    Estimate training time for full dataset based on a sample.
    
    Parameters:
    -----------
    model : sklearn estimator
        The model to test
    X_sample : array-like
        Sample of training features
    y_sample : array-like
        Sample of training labels
    full_data_size : int
        Size of the full dataset
    sample_size : int, default=1000
        Size of sample to use for estimation
        
    Returns:
    --------
    dict : Training time estimation
    """
    if len(X_sample) > sample_size:
        # Use random sample
        indices = np.random.choice(len(X_sample), sample_size, replace=False)
        X_test = X_sample[indices]
        y_test = y_sample[indices]
    else:
        X_test = X_sample
        y_test = y_sample
        sample_size = len(X_sample)
    
    # Measure training time on sample
    timing_results = measure_training_time(model, X_test, y_test, n_runs=3)
    sample_time = timing_results['mean_time']
    
    # Estimate scaling (assume roughly linear to slightly superlinear scaling)
    scaling_factor = full_data_size / sample_size
    
    # Different scaling estimates
    linear_estimate = sample_time * scaling_factor
    superlinear_estimate = sample_time * (scaling_factor ** 1.2)  # Slightly worse than linear
    log_linear_estimate = sample_time * scaling_factor * np.log(scaling_factor)
    
    return {
        'sample_size': sample_size,
        'sample_time': sample_time,
        'full_data_size': full_data_size,
        'scaling_factor': scaling_factor,
        'linear_estimate': linear_estimate,
        'superlinear_estimate': superlinear_estimate,
        'log_linear_estimate': log_linear_estimate,
        'recommended_estimate': superlinear_estimate  # Conservative estimate
    }

# ============================================================================
# INFERENCE TIME MEASUREMENT FUNCTIONS
# ============================================================================

def measure_inference_time(model, X_test, n_runs=10):
    """Measure average inference time over n_runs."""
    times = []

    # Warm-up run
    _ = model.predict(X_test[:10])

    for _ in range(n_runs):
        start_time = time.perf_counter()
        predictions = model.predict(X_test)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'times': times
    }

def measure_prediction_probabilities_time(model, X_test, n_runs=10):
    """Measure inference time for prediction probabilities."""
    times = []
    
    # Check if model has predict_proba method
    if not hasattr(model, 'predict_proba'):
        return None
    
    # Warm up
    _ = model.predict_proba(X_test[:10])
    
    for _ in range(n_runs):
        start_time = time.perf_counter()
        probabilities = model.predict_proba(X_test)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'times': times
    }

# ============================================================================
# MEMORY USAGE CALCULATION FUNCTIONS
# ============================================================================

def calculate_model_memory_usage(model, X_sample=None):
    """Calculate memory usage of a general model."""
    memory_info = {}
    
    # Base model memory
    model_size = sys.getsizeof(model)
    memory_info['model_object'] = model_size / 1024 / 1024  # MB
    
    # PCA-specific memory calculations
    if hasattr(model, 'components_'):
        components_size = model.components_.nbytes
        memory_info['pca_components'] = components_size / 1024 / 1024  # MB
        
        if hasattr(model, 'explained_variance_'):
            variance_size = model.explained_variance_.nbytes
            memory_info['explained_variance'] = variance_size / 1024 / 1024  # MB
        
        if hasattr(model, 'mean_'):
            mean_size = model.mean_.nbytes
            memory_info['mean_vector'] = mean_size / 1024 / 1024  # MB
    
    # Estimate transformation memory if sample provided
    if X_sample is not None and hasattr(model, 'components_'):
        n_samples, n_features = X_sample.shape
        n_components = model.components_.shape[0]
        
        # Memory for transformed data (float64)
        transformed_size = n_samples * n_components * 8  # 8 bytes per float64
        memory_info['transformed_data_estimate'] = transformed_size / 1024 / 1024  # MB
        
        # Memory for transformation process (temporary arrays)
        temp_memory = n_samples * n_features * 8  # Original data copy
        memory_info['transformation_temp'] = temp_memory / 1024 / 1024  # MB
    
    return memory_info

def calculate_lr_memory_usage(model, X_sample=None):
    """Calculate memory usage of a Logistic Regression model."""
    memory_info = {}
    
    # Base model memory
    model_size = sys.getsizeof(model)
    memory_info['model_object'] = model_size / 1024 / 1024  # MB
    
    # Logistic Regression specific memory calculations
    if hasattr(model, 'coef_'):
        coef_size = model.coef_.nbytes
        memory_info['coefficients'] = coef_size / 1024 / 1024  # MB
    
    if hasattr(model, 'intercept_'):
        intercept_size = model.intercept_.nbytes
        memory_info['intercept'] = intercept_size / 1024 / 1024  # MB
    
    # Estimate prediction memory if sample provided
    if X_sample is not None:
        n_samples, n_features = X_sample.shape
        
        # Memory for probability predictions (n_samples × n_classes × 8 bytes)
        n_classes = len(model.classes_) if hasattr(model, 'classes_') else 1
        prob_size = n_samples * n_classes * 8  # 8 bytes per float64
        memory_info['prediction_probs'] = prob_size / 1024 / 1024  # MB
        
        # Memory for decision function (temporary arrays)
        decision_size = n_samples * n_classes * 8
        memory_info['decision_function'] = decision_size / 1024 / 1024  # MB
    
    return memory_info

def calculate_lda_memory_usage(model, X_sample=None):
    """Calculate memory usage of an LDA model."""
    memory_info = {}
    
    # Base model memory
    model_size = sys.getsizeof(model)
    memory_info['model_object'] = model_size / 1024 / 1024  # MB
    
    # LDA-specific memory calculations
    if hasattr(model, 'scalings_'):
        scalings_size = model.scalings_.nbytes if model.scalings_ is not None else 0
        memory_info['lda_scalings'] = scalings_size / 1024 / 1024  # MB
    
    if hasattr(model, 'coef_'):
        coef_size = model.coef_.nbytes if model.coef_ is not None else 0
        memory_info['lda_coefficients'] = coef_size / 1024 / 1024  # MB
    
    if hasattr(model, 'intercept_'):
        intercept_size = model.intercept_.nbytes if model.intercept_ is not None else 0
        memory_info['lda_intercept'] = intercept_size / 1024 / 1024  # MB
    
    if hasattr(model, 'means_'):
        means_size = model.means_.nbytes if model.means_ is not None else 0
        memory_info['class_means'] = means_size / 1024 / 1024  # MB
    
    if hasattr(model, 'priors_'):
        priors_size = model.priors_.nbytes if model.priors_ is not None else 0
        memory_info['class_priors'] = priors_size / 1024 / 1024  # MB
    
    if hasattr(model, 'xbar_'):
        xbar_size = model.xbar_.nbytes if model.xbar_ is not None else 0
        memory_info['overall_mean'] = xbar_size / 1024 / 1024  # MB
    
    # Estimate transformation memory if sample provided
    if X_sample is not None and hasattr(model, 'scalings_') and model.scalings_ is not None:
        n_samples, n_features = X_sample.shape
        n_components = model.scalings_.shape[1]  # Number of discriminant components
        
        # Memory for transformed data (float64)
        transformed_size = n_samples * n_components * 8  # 8 bytes per float64
        memory_info['transformed_data_estimate'] = transformed_size / 1024 / 1024  # MB
        
        # Memory for transformation process (temporary arrays)
        temp_memory = n_samples * n_features * 8  # Original data copy
        memory_info['transformation_temp'] = temp_memory / 1024 / 1024  # MB
    
    return memory_info

def calculate_tree_memory_usage(model, X_sample=None):
    """Calculate memory usage of tree-based models (Decision Tree, Random Forest)."""
    memory_info = {}
    
    # Base model memory
    model_size = sys.getsizeof(model)
    memory_info['model_object'] = model_size / 1024 / 1024  # MB
    
    # Tree-specific calculations
    if hasattr(model, 'tree_'):  # Single tree (Decision Tree)
        tree = model.tree_
        # Node arrays
        node_memory = (tree.node_count * 8 * 4) / 1024 / 1024  # Approximation
        memory_info['tree_nodes'] = node_memory
        
        # Feature and threshold arrays
        if hasattr(tree, 'feature'):
            feature_memory = tree.feature.nbytes / 1024 / 1024
            memory_info['tree_features'] = feature_memory
        
        if hasattr(tree, 'threshold'):
            threshold_memory = tree.threshold.nbytes / 1024 / 1024
            memory_info['tree_thresholds'] = threshold_memory
    
    elif hasattr(model, 'estimators_'):  # Ensemble (Random Forest)
        n_estimators = len(model.estimators_)
        memory_info['n_estimators'] = n_estimators
        
        # Estimate memory for all trees
        if n_estimators > 0:
            single_tree = model.estimators_[0]
            if hasattr(single_tree, 'tree_'):
                tree = single_tree.tree_
                single_tree_memory = (tree.node_count * 8 * 4) / 1024 / 1024
                memory_info['all_trees_estimate'] = single_tree_memory * n_estimators
    
    # Prediction memory estimate
    if X_sample is not None:
        n_samples, n_features = X_sample.shape
        n_classes = len(model.classes_) if hasattr(model, 'classes_') else 1
        
        # Memory for predictions
        pred_size = n_samples * n_classes * 8
        memory_info['prediction_memory'] = pred_size / 1024 / 1024  # MB
    
    return memory_info

def calculate_svm_memory_usage(model, X_sample=None):
    """Calculate memory usage of SVM models."""
    memory_info = {}
    
    # Base model memory
    model_size = sys.getsizeof(model)
    memory_info['model_object'] = model_size / 1024 / 1024  # MB
    
    # SVM-specific calculations
    if hasattr(model, 'support_vectors_'):
        sv_size = model.support_vectors_.nbytes
        memory_info['support_vectors'] = sv_size / 1024 / 1024  # MB
        memory_info['n_support_vectors'] = len(model.support_vectors_)
    
    if hasattr(model, 'dual_coef_'):
        dual_coef_size = model.dual_coef_.nbytes
        memory_info['dual_coefficients'] = dual_coef_size / 1024 / 1024  # MB
    
    if hasattr(model, 'intercept_'):
        intercept_size = model.intercept_.nbytes
        memory_info['intercept'] = intercept_size / 1024 / 1024  # MB
    
    # Prediction memory estimate
    if X_sample is not None:
        n_samples, n_features = X_sample.shape
        n_classes = len(model.classes_) if hasattr(model, 'classes_') else 1
        
        # Memory for kernel computations (depends on support vectors)
        if hasattr(model, 'support_vectors_'):
            n_sv = len(model.support_vectors_)
            kernel_memory = n_samples * n_sv * 8  # Distance calculations
            memory_info['kernel_computation'] = kernel_memory / 1024 / 1024  # MB
    
    return memory_info

def calculate_knn_memory_usage(model, X_sample=None):
    """Calculate memory usage of KNN models."""
    memory_info = {}
    
    # Base model memory
    model_size = sys.getsizeof(model)
    memory_info['model_object'] = model_size / 1024 / 1024  # MB
    
    # KNN stores the entire training set
    if hasattr(model, '_fit_X'):
        training_data_size = model._fit_X.nbytes
        memory_info['training_data'] = training_data_size / 1024 / 1024  # MB
        memory_info['n_training_samples'] = len(model._fit_X)
    
    if hasattr(model, '_y'):
        labels_size = model._y.nbytes
        memory_info['training_labels'] = labels_size / 1024 / 1024  # MB
    
    # Prediction memory estimate (distance calculations)
    if X_sample is not None and hasattr(model, '_fit_X'):
        n_samples, n_features = X_sample.shape
        n_training = len(model._fit_X)
        
        # Memory for distance matrix
        distance_memory = n_samples * n_training * 8  # float64
        memory_info['distance_computation'] = distance_memory / 1024 / 1024  # MB
    
    return memory_info

def print_memory_report(memory_info, model_name="Model"):
    """Print formatted memory usage report."""
    print(f"\n📊 {model_name} Memory Usage Report:")
    print("-" * 45)
    
    total_memory = 0
    for component, size_mb in memory_info.items():
        if isinstance(size_mb, (int, float)) and size_mb > 0:
            print(f"   {component.replace('_', ' ').title()}: {size_mb:.4f} MB")
            total_memory += size_mb
        elif isinstance(size_mb, int) and 'n_' in component:
            print(f"   {component.replace('_', ' ').title()}: {size_mb}")
    
    print(f"   {'Total Estimated'}: {total_memory:.4f} MB")
    
    # Memory recommendations
    if total_memory > 1000:  # > 1GB
        print(f"\n⚠️  High memory usage detected ({total_memory:.1f} MB)")
        print("   💡 Consider model compression or feature selection")
    elif total_memory > 100:  # > 100MB
        print(f"\n⚡ Moderate memory usage ({total_memory:.1f} MB)")
        print("   💡 Monitor memory if processing large datasets")
    else:
        print(f"\n✅ Efficient memory usage ({total_memory:.4f} MB)")

def print_lr_memory_report(memory_info, model_name="Logistic Regression"):
    """Print formatted LR memory usage report."""
    return print_memory_report(memory_info, model_name)

def print_lda_memory_report(memory_info, model_name="LDA Model"):
    """Print formatted LDA memory usage report with LDA-specific recommendations."""
    print_memory_report(memory_info, model_name)
    
    total_memory = sum(v for v in memory_info.values() if isinstance(v, (int, float)))
    
    # LDA-specific recommendations
    if total_memory > 500:  # > 500MB
        print("   💡 LDA memory scales with n_features × n_classes")
        print("   💡 Consider feature selection before LDA")
    elif total_memory > 100:  # > 100MB
        print("   💡 LDA is generally memory efficient")
    else:
        print("   💡 LDA optimal for this dataset size")

def analyze_lda_components(lda_model, class_names):
    """Analyze LDA discriminant components."""
    if hasattr(lda_model, 'scalings_') and lda_model.scalings_ is not None:
        n_components = lda_model.scalings_.shape[1]
        n_features = lda_model.scalings_.shape[0]
        
        print(f"\n🔍 LDA Component Analysis:")
        print(f"   Number of discriminant components: {n_components}")
        print(f"   Input features: {n_features}")
        print(f"   Dimensionality reduction: {(1 - n_components/n_features)*100:.1f}%")
        
        if hasattr(lda_model, 'explained_variance_ratio_'):
            print(f"   Explained variance ratios: {lda_model.explained_variance_ratio_}")
            print(f"   Total variance explained: {np.sum(lda_model.explained_variance_ratio_):.3f}")

def print_inference_performance(pred_timing, prob_timing=None, model_name="Model", n_samples=None):
    """Print formatted inference performance report."""
    print(f"\n📊 {model_name} Inference Performance:")
    print("-" * 45)
    
    if n_samples:
        print(f"   Test samples: {n_samples}")
    
    print(f"   Prediction Time (mean ± std): {pred_timing['mean_time']*1000:.2f} ± {pred_timing['std_time']*1000:.2f} ms")
    print(f"   Prediction Time (min/max): {pred_timing['min_time']*1000:.2f} / {pred_timing['max_time']*1000:.2f} ms")
    
    if n_samples:
        print(f"   Time per sample: {pred_timing['mean_time']*1000/n_samples:.4f} ms")
        print(f"   Samples per second: {n_samples/pred_timing['mean_time']:.0f}")
    
    if prob_timing:
        print("-" * 40)
        print(f"   Probability Prediction Time (mean ± std): {prob_timing['mean_time']*1000:.2f} ± {prob_timing['std_time']*1000:.2f} ms")
        print(f"   Probability Time (min/max): {prob_timing['min_time']*1000:.2f} / {prob_timing['max_time']*1000:.2f} ms")

def get_model_type_specific_memory(model, X_sample=None):
    """Get memory usage based on model type."""
    model_type = type(model).__name__
    
    if 'LogisticRegression' in model_type:
        return calculate_lr_memory_usage(model, X_sample)
    elif 'LinearDiscriminantAnalysis' in model_type:
        return calculate_lda_memory_usage(model, X_sample)
    elif 'PCA' in model_type:
        return calculate_model_memory_usage(model, X_sample)
    elif 'DecisionTree' in model_type or 'RandomForest' in model_type:
        return calculate_tree_memory_usage(model, X_sample)
    elif 'SVM' in model_type or 'SVC' in model_type:
        return calculate_svm_memory_usage(model, X_sample)
    elif 'KNeighbors' in model_type:
        return calculate_knn_memory_usage(model, X_sample)
    else:
        # Generic calculation for unknown models
        return calculate_model_memory_usage(model, X_sample)

def comprehensive_model_analysis(model, X_test, model_name=None, n_runs=10):
    """Perform comprehensive analysis of model performance and memory usage."""
    if model_name is None:
        model_name = type(model).__name__
    
    print(f"\n🔍 Comprehensive Analysis: {model_name}")
    print("=" * 60)
    
    # Memory analysis
    memory_info = get_model_type_specific_memory(model, X_test)
    print_memory_report(memory_info, model_name)
    
    # Performance analysis
    pred_timing = measure_inference_time(model, X_test, n_runs)
    prob_timing = measure_prediction_probabilities_time(model, X_test, n_runs)
    
    print_inference_performance(pred_timing, prob_timing, model_name, len(X_test))
    
    # Efficiency metrics
    total_memory = sum(v for v in memory_info.values() if isinstance(v, (int, float)))
    samples_per_second = len(X_test) / pred_timing['mean_time']
    memory_per_sample = total_memory / len(X_test) * 1024  # KB per sample
    
    print(f"\n⚡ Efficiency Metrics:")
    print("-" * 25)
    print(f"Throughput: {samples_per_second:.0f} samples/second")
    print(f"Memory per sample: {memory_per_sample:.4f} KB")
    print(f"Total model memory: {total_memory:.4f} MB")
    
    return {
        'memory_info': memory_info,
        'pred_timing': pred_timing,
        'prob_timing': prob_timing,
        'total_memory_mb': total_memory,
        'samples_per_second': samples_per_second,
        'memory_per_sample_kb': memory_per_sample
    }

def compare_models_efficiency(models_results, metric='samples_per_second'):
    """Compare efficiency metrics across multiple models."""
    print(f"\n📊 Model Efficiency Comparison ({metric}):")
    print("=" * 50)
    
    for name, results in models_results.items():
        if metric in results:
            value = results[metric]
            if 'memory' in metric:
                print(f"   {name:25}: {value:.4f} MB")
            elif 'second' in metric:
                print(f"   {name:25}: {value:.0f} samples/sec")
            else:
                print(f"   {name:25}: {value:.4f}")
    
    # Find best performing model
    if models_results:
        if 'memory' in metric:
            best_model = min(models_results.items(), key=lambda x: x[1].get(metric, float('inf')))
            print(f"\n🏆 Most memory efficient: {best_model[0]}")
        else:
            best_model = max(models_results.items(), key=lambda x: x[1].get(metric, 0))
            print(f"\n🏆 Best performance: {best_model[0]}")


def calculate_lda_knn_memory_usage(lda_model, knn_model, X_train_reduced=None):
    """Estimate total memory usage of LDA + KNN pipeline (in MB)."""
    memory_info = {}

    # --- LDA memory ---
    lda_mem = calculate_lda_memory_usage(lda_model)
    memory_info['LDA'] = lda_mem
    memory_info['LDA_total_MB'] = sum(lda_mem.values())

    # --- KNN memory ---
    knn_info = {}
    knn_size = sys.getsizeof(knn_model)
    knn_info['model_object'] = knn_size / 1024 / 1024  # MB

    # Stored training data (this dominates memory)
    if hasattr(knn_model, '_fit_X'):
        fit_x_size = knn_model._fit_X.nbytes
        knn_info['stored_training_data'] = fit_x_size / 1024 / 1024  # MB

    if hasattr(knn_model, '_y'):
        fit_y_size = knn_model._y.nbytes
        knn_info['stored_labels'] = fit_y_size / 1024 / 1024  # MB

    # Optionally estimate transformation if LDA output is provided
    if X_train_reduced is not None:
        n_samples, n_components = X_train_reduced.shape
        transformed_size = n_samples * n_components * 8
        knn_info['transformed_data_estimate'] = transformed_size / 1024 / 1024

    memory_info['KNN'] = knn_info
    memory_info['KNN_total_MB'] = sum(knn_info.values())

    # --- Total pipeline memory ---
    memory_info['Pipeline_total_MB'] = (
        memory_info['LDA_total_MB'] + memory_info['KNN_total_MB']
    )

    return memory_info
