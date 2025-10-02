"""
Baseline Correction Methods for Spectroscopic Data

This module provides various baseline correction algorithms for LIBS and other
spectroscopic data, including AsLS, 4S Peak Filling, and Hybrid methods.


"""

import numpy as np
import pandas as pd # type: ignore[import]
import matplotlib.pyplot as plt
from scipy import sparse # type: ignore[import]
from scipy.sparse.linalg import spsolve # type: ignore[import]
from scipy.signal import savgol_filter # type: ignore[import]
from scipy.interpolate import interp1d # type: ignore[import]
from scipy.ndimage import median_filter # type: ignore[import]
from typing import Optional, Tuple, Union, Callable

class BaselineCorrector:
    """
    Comprehensive baseline correction class for spectroscopic data.
    
    Provides multiple baseline correction algorithms:
    - AsLS (Asymmetric Least Squares)
    - 4S Peak Filling
    - Hybrid 4S+AsLS
    """
    
    def __init__(self):
        """Initialize the BaselineCorrector."""
        self.correction_methods = {
            'als': self.baseline_als,
            '4s': self.baseline_4s_peak_filling,
            'hybrid': self.baseline_hybrid_4s_als
        }
    
    @staticmethod
    def baseline_als(y: np.ndarray, lam: float = 1e4, p: float = 0.01, niter: int = 10) -> np.ndarray:
        """
        Asymmetric Least Squares baseline correction.
        
        Parameters:
        -----------
        y : np.ndarray
            Input signal
        lam : float, default=1e4
            Smoothness parameter (larger = smoother baseline)
        p : float, default=0.01
            Asymmetry parameter (0 < p < 1, smaller = more asymmetric)
        niter : int, default=10
            Number of iterations
            
        Returns:
        --------
        np.ndarray
            Estimated baseline
        """
        L = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        w = np.ones(L)
        
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        
        return z
    
    @staticmethod
    def baseline_4s_peak_filling(y: np.ndarray, 
                                window_length: int = 51, 
                                polyorder: int = 3, 
                                iterations: int = 4,
                                threshold_factor: float = 0.1, 
                                fill_factor: float = 0.8) -> np.ndarray:
        """
        4S Peak Filling baseline correction algorithm.
        
        This algorithm iteratively identifies and fills peaks to estimate the baseline:
        1. Smooth the spectrum
        2. Identify peaks (points above smoothed baseline)
        3. Fill peaks by interpolation or median filtering
        4. Repeat until convergence
        
        Parameters:
        -----------
        y : np.ndarray
            Input signal
        window_length : int, default=51
            Window length for Savitzky-Golay filter
        polyorder : int, default=3
            Polynomial order for Savitzky-Golay filter
        iterations : int, default=4
            Number of peak filling iterations
        threshold_factor : float, default=0.1
            Threshold factor for peak identification
        fill_factor : float, default=0.8
            Factor for peak filling interpolation
            
        Returns:
        --------
        np.ndarray
            Estimated baseline
        """
        # Ensure window_length is odd and valid
        if window_length % 2 == 0:
            window_length += 1
        window_length = min(window_length, len(y)//4*2 + 1)  # Ensure reasonable size
        
        # Start with a copy of the original signal
        baseline = y.copy().astype(float)
        
        for iteration in range(iterations):
            # Step 1: Apply Savitzky-Golay smoothing
            try:
                smoothed = savgol_filter(baseline, window_length, polyorder)
            except ValueError:
                # Fallback to median filter
                smoothed = median_filter(baseline, size=min(window_length//2, 5))
            
            # Step 2: Identify peaks (points significantly above smoothed baseline)
            threshold = np.std(baseline - smoothed) * threshold_factor
            peak_mask = (baseline - smoothed) > threshold
            
            if np.sum(peak_mask) == 0:
                break
            
            # Step 3: Fill peaks
            baseline[peak_mask] = (smoothed[peak_mask] * fill_factor + 
                                 baseline[peak_mask] * (1 - fill_factor))
            
            # Additional interpolation for isolated peaks
            peak_indices = np.where(peak_mask)[0]
            if len(peak_indices) > 0:
                # Group consecutive peak indices
                peak_groups = []
                current_group = [peak_indices[0]]
                
                for i in range(1, len(peak_indices)):
                    if peak_indices[i] - peak_indices[i-1] <= 2:  # Close peaks
                        current_group.append(peak_indices[i])
                    else:
                        peak_groups.append(current_group)
                        current_group = [peak_indices[i]]
                peak_groups.append(current_group)
                
                # Interpolate across peak groups
                for group in peak_groups:
                    if len(group) > 1:  # Only interpolate for groups of peaks
                        start_idx = max(0, group[0] - 2)
                        end_idx = min(len(baseline) - 1, group[-1] + 2)
                        
                        # Ensure we have valid indices
                        if start_idx >= end_idx or start_idx < 0 or end_idx >= len(baseline):
                            continue  # Skip invalid ranges
                        
                        # Create interpolation points
                        try:
                            x_interp = np.array([start_idx, end_idx])
                            y_interp = np.array([baseline[start_idx], baseline[end_idx]])
                            
                            if len(x_interp) >= 2 and start_idx < end_idx:
                                f_interp = interp1d(x_interp, y_interp, kind='linear', 
                                                  fill_value='extrapolate')
                                
                                # Ensure interpolation range is valid
                                interp_range = np.arange(group[0], min(group[-1]+1, len(baseline)))
                                if len(interp_range) > 0:
                                    baseline[interp_range] = f_interp(interp_range)
                                    
                        except (ValueError, IndexError):
                            # Skip this group if interpolation fails
                            continue
        
        return baseline
    
    def baseline_hybrid_4s_als(self, 
                              y: np.ndarray, 
                              window_length: int = 51, 
                              polyorder: int = 3, 
                              als_lam: float = 1e4, 
                              als_p: float = 0.01) -> np.ndarray:
        """
        Hybrid baseline correction: 4S Peak Filling followed by AsLS refinement.
        
        Parameters:
        -----------
        y : np.ndarray
            Input signal
        window_length : int, default=51
            Window length for 4S method
        polyorder : int, default=3
            Polynomial order for 4S method
        als_lam : float, default=1e4
            AsLS smoothness parameter
        als_p : float, default=0.01
            AsLS asymmetry parameter
            
        Returns:
        --------
        np.ndarray
            Estimated baseline
        """
        # Step 1: 4S Peak Filling for initial baseline estimation
        baseline_4s = self.baseline_4s_peak_filling(y, window_length, polyorder, iterations=2)
        
        # Step 2: Apply AsLS to the 4S result for fine-tuning
        baseline_final = self.baseline_als(baseline_4s, lam=als_lam, p=als_p)
        
        return baseline_final
    
    def apply_correction_batch(self, 
                              spectra_matrix: Union[np.ndarray, pd.DataFrame], 
                              method: str = 'hybrid',
                              batch_size: int = 100,
                              **kwargs) -> np.ndarray:
        """
        Apply baseline correction to multiple spectra efficiently in batches.
        
        Parameters:
        -----------
        spectra_matrix : np.ndarray or pd.DataFrame
            Matrix of spectra (n_spectra x n_wavelengths)
        method : str, default='hybrid'
            Correction method ('als', '4s', or 'hybrid')
        batch_size : int, default=100
            Number of spectra to process in each batch
        **kwargs
            Additional parameters for the correction method
            
        Returns:
        --------
        np.ndarray
            Matrix of corrected spectra
        """
        # Convert to numpy array if DataFrame
        if isinstance(spectra_matrix, pd.DataFrame):
            spectra_array = spectra_matrix.values
        else:
            spectra_array = spectra_matrix
        
        n_spectra, n_wavelengths = spectra_array.shape
        corrected_spectra = np.zeros_like(spectra_array, dtype=np.float32)
        
        # Get the correction method
        if method not in self.correction_methods:
            raise ValueError(f"Unknown method '{method}'. Available methods: {list(self.correction_methods.keys())}")
        
        method_func = self.correction_methods[method]
        
        print(f"Applying {method.upper()} baseline correction to {n_spectra} spectra in batches of {batch_size}...")
        
        for start_idx in range(0, n_spectra, batch_size):
            end_idx = min(start_idx + batch_size, n_spectra)
            
            print(f"  Processing batch {start_idx//batch_size + 1}/{(n_spectra-1)//batch_size + 1} "
                  f"(spectra {start_idx+1}-{end_idx})")
            
            for i in range(start_idx, end_idx):
                try:
                    spectrum = spectra_array[i].astype(float)
                    baseline = method_func(spectrum, **kwargs)
                    corrected_spectra[i] = spectrum - baseline
                    
                except Exception as e:
                    print(f"    Warning: Failed to correct spectrum {i+1}: {e}")
                    corrected_spectra[i] = spectra_array[i]  # Keep original on failure
        
        return corrected_spectra
    
    def correct_single_spectrum(self, 
                               spectrum: np.ndarray, 
                               method: str = 'hybrid',
                               **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply baseline correction to a single spectrum.
        
        Parameters:
        -----------
        spectrum : np.ndarray
            Input spectrum
        method : str, default='hybrid'
            Correction method ('als', '4s', or 'hybrid')
        **kwargs
            Additional parameters for the correction method
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (corrected_spectrum, baseline)
        """
        if method not in self.correction_methods:
            raise ValueError(f"Unknown method '{method}'. Available methods: {list(self.correction_methods.keys())}")
        
        method_func = self.correction_methods[method]
        baseline = method_func(spectrum.astype(float), **kwargs)
        corrected_spectrum = spectrum - baseline
        
        return corrected_spectrum, baseline
    
    def visualize_correction(self, 
                            spectrum: np.ndarray, 
                            wavelengths: Optional[np.ndarray] = None,
                            method: str = 'hybrid',
                            **kwargs) -> None:
        """
        Visualize the baseline correction effect on a single spectrum.
        
        Parameters:
        -----------
        spectrum : np.ndarray
            Input spectrum
        wavelengths : np.ndarray, optional
            Wavelength values for x-axis
        method : str, default='hybrid'
            Correction method to visualize
        **kwargs
            Additional parameters for the correction method
        """
        corrected_spectrum, baseline = self.correct_single_spectrum(spectrum, method, **kwargs)
        
        if wavelengths is None:
            wavelengths = np.arange(len(spectrum))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Original spectrum with baseline
        axes[0, 0].plot(wavelengths, spectrum, 'b-', label='Original Spectrum', linewidth=1)
        axes[0, 0].plot(wavelengths, baseline, 'r--', label=f'{method.upper()} Baseline', linewidth=2)
        axes[0, 0].set_title('Original Spectrum with Baseline Estimate')
        axes[0, 0].set_xlabel('Wavelength')
        axes[0, 0].set_ylabel('Intensity')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Baseline-corrected spectrum
        axes[0, 1].plot(wavelengths, spectrum, 'b-', alpha=0.3, label='Original')
        axes[0, 1].plot(wavelengths, corrected_spectrum, 'g-', 
                       label='Baseline Corrected', linewidth=1.5)
        axes[0, 1].set_title('Baseline-Corrected Spectrum')
        axes[0, 1].set_xlabel('Wavelength')
        axes[0, 1].set_ylabel('Intensity')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Intensity distributions
        axes[1, 0].hist(spectrum, bins=50, alpha=0.5, label='Original', density=True)
        axes[1, 0].hist(corrected_spectrum, bins=50, alpha=0.5, label='Corrected', density=True)
        axes[1, 0].set_title('Intensity Distribution Comparison')
        axes[1, 0].set_xlabel('Intensity')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Baseline plot alone
        axes[1, 1].plot(wavelengths, baseline, 'r-', linewidth=2)
        axes[1, 1].set_title(f'{method.upper()} Estimated Baseline')
        axes[1, 1].set_xlabel('Wavelength')
        axes[1, 1].set_ylabel('Baseline Intensity')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def compare_methods(self, 
                       spectrum: np.ndarray, 
                       wavelengths: Optional[np.ndarray] = None,
                       methods: Optional[list] = None) -> None:
        """
        Compare different baseline correction methods on the same spectrum.
        
        Parameters:
        -----------
        spectrum : np.ndarray
            Input spectrum
        wavelengths : np.ndarray, optional
            Wavelength values for x-axis
        methods : list, optional
            List of methods to compare. Default: ['als', '4s', 'hybrid']
        """
        if methods is None:
            methods = ['als', '4s', 'hybrid']
        
        if wavelengths is None:
            wavelengths = np.arange(len(spectrum))
        
        n_methods = len(methods)
        fig, axes = plt.subplots(n_methods, 2, figsize=(15, 4*n_methods))
        
        if n_methods == 1:
            axes = axes.reshape(1, -1)
        
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        
        for i, method in enumerate(methods):
            corrected_spectrum, baseline = self.correct_single_spectrum(spectrum, method)
            color = colors[i % len(colors)]
            
            # Original with baseline
            axes[i, 0].plot(wavelengths, spectrum, 'b-', alpha=0.7, label='Original')
            axes[i, 0].plot(wavelengths, baseline, f'{color}--', 
                           label=f'{method.upper()} Baseline', linewidth=2)
            axes[i, 0].set_title(f'{method.upper()} Method - Original + Baseline')
            axes[i, 0].set_xlabel('Wavelength')
            axes[i, 0].set_ylabel('Intensity')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
            
            # Corrected spectrum
            axes[i, 1].plot(wavelengths, spectrum, 'b-', alpha=0.3, label='Original')
            axes[i, 1].plot(wavelengths, corrected_spectrum, f'{color}-', 
                           label=f'{method.upper()} Corrected', linewidth=1.5)
            axes[i, 1].set_title(f'{method.upper()} Method - Corrected Spectrum')
            axes[i, 1].set_xlabel('Wavelength')
            axes[i, 1].set_ylabel('Intensity')
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def analyze_correction_quality(original_data: Union[np.ndarray, pd.DataFrame], 
                              corrected_data: np.ndarray,
                              class_labels: Optional[np.ndarray] = None) -> dict:
    """
    Analyze the quality of baseline correction by comparing statistics.
    
    Parameters:
    -----------
    original_data : np.ndarray or pd.DataFrame
        Original spectral data
    corrected_data : np.ndarray
        Baseline-corrected spectral data
    class_labels : np.ndarray, optional
        Class labels for class-wise analysis
        
    Returns:
    --------
    dict
        Dictionary containing quality metrics
    """
    if isinstance(original_data, pd.DataFrame):
        original_array = original_data.values
    else:
        original_array = original_data
    
    # Calculate basic statistics
    quality_metrics = {
        'original_stats': {
            'mean': np.mean(original_array),
            'std': np.std(original_array),
            'min': np.min(original_array),
            'max': np.max(original_array)
        },
        'corrected_stats': {
            'mean': np.mean(corrected_data),
            'std': np.std(corrected_data),
            'min': np.min(corrected_data),
            'max': np.max(corrected_data)
        }
    }
    
    # Calculate changes
    quality_metrics['changes'] = {
        'mean_change': quality_metrics['corrected_stats']['mean'] - quality_metrics['original_stats']['mean'],
        'std_change': quality_metrics['corrected_stats']['std'] - quality_metrics['original_stats']['std'],
        'range_reduction': ((quality_metrics['original_stats']['max'] - quality_metrics['original_stats']['min']) - 
                           (quality_metrics['corrected_stats']['max'] - quality_metrics['corrected_stats']['min']))
    }
    
    # Class-wise analysis if labels provided
    if class_labels is not None:
        quality_metrics['class_wise'] = {}
        unique_classes = np.unique(class_labels)
        
        for cls in unique_classes:
            cls_mask = class_labels == cls
            quality_metrics['class_wise'][cls] = {
                'original_mean': np.mean(original_array[cls_mask]),
                'corrected_mean': np.mean(corrected_data[cls_mask]),
                'original_std': np.std(original_array[cls_mask]),
                'corrected_std': np.std(corrected_data[cls_mask])
            }
    
    return quality_metrics


def print_quality_report(quality_metrics: dict) -> None:
    """
    Print a formatted quality report from analyze_correction_quality results.
    
    Parameters:
    -----------
    quality_metrics : dict
        Quality metrics from analyze_correction_quality
    """
    print("=== BASELINE CORRECTION QUALITY REPORT ===")
    print(f"{'Metric':<15} {'Original':<15} {'Corrected':<15} {'Change':<10}")
    print("-" * 60)
    
    orig = quality_metrics['original_stats']
    corr = quality_metrics['corrected_stats']
    changes = quality_metrics['changes']
    
    print(f"{'Mean':<15} {orig['mean']:<15.2f} {corr['mean']:<15.2f} {changes['mean_change']:<10.2f}")
    print(f"{'Std Dev':<15} {orig['std']:<15.2f} {corr['std']:<15.2f} {changes['std_change']:<10.2f}")
    print(f"{'Minimum':<15} {orig['min']:<15.2f} {corr['min']:<15.2f} {corr['min']-orig['min']:<10.2f}")
    print(f"{'Maximum':<15} {orig['max']:<15.2f} {corr['max']:<15.2f} {corr['max']-orig['max']:<10.2f}")
    print(f"{'Range Reduction':<15} {'':<15} {'':<15} {changes['range_reduction']:<10.2f}")
    
    if 'class_wise' in quality_metrics:
        print("\n=== CLASS-WISE ANALYSIS ===")
        for cls, stats in quality_metrics['class_wise'].items():
            print(f"\nClass {cls}:")
            print(f"  Original mean: {stats['original_mean']:.2f} ± {stats['original_std']:.2f}")
            print(f"  Corrected mean: {stats['corrected_mean']:.2f} ± {stats['corrected_std']:.2f}")
            print(f"  Mean change: {stats['corrected_mean']-stats['original_mean']:+.2f}")


# Convenience functions for backward compatibility
def baseline_als(y: np.ndarray, lam: float = 1e4, p: float = 0.01, niter: int = 10) -> np.ndarray:
    """Convenience function for AsLS baseline correction."""
    corrector = BaselineCorrector()
    return corrector.baseline_als(y, lam, p, niter)


def baseline_4s_peak_filling(y: np.ndarray, 
                            window_length: int = 51, 
                            polyorder: int = 3, 
                            iterations: int = 4,
                            threshold_factor: float = 0.1, 
                            fill_factor: float = 0.8) -> np.ndarray:
    """Convenience function for 4S peak filling baseline correction."""
    corrector = BaselineCorrector()
    return corrector.baseline_4s_peak_filling(y, window_length, polyorder, iterations, 
                                             threshold_factor, fill_factor)


def baseline_hybrid_4s_als(y: np.ndarray, 
                          window_length: int = 51, 
                          polyorder: int = 3, 
                          als_lam: float = 1e4, 
                          als_p: float = 0.01) -> np.ndarray:
    """Convenience function for hybrid 4S+AsLS baseline correction."""
    corrector = BaselineCorrector()
    return corrector.baseline_hybrid_4s_als(y, window_length, polyorder, als_lam, als_p)


def apply_baseline_correction_batch(spectra_matrix: Union[np.ndarray, pd.DataFrame], 
                                   method_func: Callable,
                                   batch_size: int = 100, 
                                   **kwargs) -> np.ndarray:
    """Convenience function for batch baseline correction."""
    corrector = BaselineCorrector()
    # Determine method name from function
    method_name = 'hybrid'  # Default
    if 'als' in method_func.__name__:
        if '4s' in method_func.__name__ or 'hybrid' in method_func.__name__:
            method_name = 'hybrid'
        else:
            method_name = 'als'
    elif '4s' in method_func.__name__:
        method_name = '4s'
    
    return corrector.apply_correction_batch(spectra_matrix, method_name, batch_size, **kwargs)