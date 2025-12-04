"""
Saturation analysis functions for LIBS spectroscopy data.
"""

import pandas as pd # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from typing import Dict, List, Tuple, Optional, Union
import numpy as np # type: ignore
from scipy import special # type: ignore
from scipy.optimize import curve_fit # type: ignore
import warnings

def get_wavelength_columns(df: pd.DataFrame) -> List[str]:
    """
    Identify wavelength columns in the DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing LIBS measurements
        
    Returns:
    --------
    list : List of wavelength column names
    """
    # Standard non-wavelength columns
    non_feature_cols = ['tire_number', 'origin', 'measurement_id', 'sample_type']
    
    # Method 1: Try to get all numeric columns that aren't metadata
    wavelength_columns = []
    for col in df.columns:
        if col not in non_feature_cols:
            try:
                # Try to convert to float - if successful, it's likely a wavelength
                float(col)
                wavelength_columns.append(col)
            except ValueError:
                continue
    
    # Method 2: Fallback to original method if needed
    if len(wavelength_columns) == 0:
        wavelength_columns = [col for col in df.columns 
                            if col.replace('.', '').replace('-', '').isdigit()]
    
    return wavelength_columns

def check_saturation(df: pd.DataFrame, 
                    saturation_threshold: int = 65535,
                    verbose: bool = True) -> Dict:
    """
    Check for saturation across all wavelengths and measurements.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing LIBS measurements
    saturation_threshold : int, default=65535
        Intensity threshold above which pixels are considered saturated
    verbose : bool, default=True
        Whether to print analysis results
        
    Returns:
    --------
    dict : Comprehensive saturation analysis results
    """
    
    # Get wavelength columns
    wavelength_columns = get_wavelength_columns(df)
    
    if verbose:
        print(f"🔍 Saturation Analysis (threshold: {saturation_threshold})")
        print("=" * 60)
        print(f"Total measurements: {len(df)}")
        print(f"Total wavelengths: {len(wavelength_columns)}")
    
    if len(wavelength_columns) == 0:
        if verbose:
            print("❌ No wavelength columns detected!")
            print(f"Available columns: {list(df.columns)}")
        return {}
    
    # Check for saturated values
    saturated_mask = df[wavelength_columns] >= saturation_threshold
    
    # Overall saturation statistics
    total_values = len(df) * len(wavelength_columns)
    saturated_values = saturated_mask.sum().sum()
    saturation_percentage = (saturated_values / total_values) * 100
    
    # Measurements with any saturation
    measurements_with_saturation = saturated_mask.any(axis=1)
    n_saturated_measurements = measurements_with_saturation.sum()
    measurement_saturation_percentage = (n_saturated_measurements / len(df)) * 100
    
    # Wavelength-level analysis
    wavelengths_with_saturation = saturated_mask.any(axis=0)
    n_saturated_wavelengths = wavelengths_with_saturation.sum()
    wavelength_saturation_percentage = (n_saturated_wavelengths / len(wavelength_columns)) * 100
    
    # Most problematic wavelengths (most frequently saturated)
    saturation_counts_per_wavelength = saturated_mask.sum(axis=0)
    most_saturated_wavelengths = saturation_counts_per_wavelength.nlargest(10)
    
    # Most problematic measurements (most wavelengths saturated)
    saturation_counts_per_measurement = saturated_mask.sum(axis=1)
    most_saturated_measurements = saturation_counts_per_measurement.nlargest(10)
    
    # Analysis by sample type (if available)
    sample_type_saturation = None
    sample_type_col = None
    
    # Check for both 'sample_type' and 'origin' columns
    if 'sample_type' in df.columns:
        sample_type_col = 'sample_type'
    elif 'origin' in df.columns:
        sample_type_col = 'origin'
    
    if sample_type_col:
        sample_type_saturation = {}
        for sample_type in df[sample_type_col].unique():
            mask = df[sample_type_col] == sample_type
            type_df = df[mask]
            type_saturated = saturated_mask[mask]
            
            type_measurements = len(type_df)
            type_saturated_measurements = type_saturated.any(axis=1).sum()
            type_percentage = (type_saturated_measurements / type_measurements) * 100 if type_measurements > 0 else 0
            
            sample_type_saturation[sample_type] = {
                'total_measurements': type_measurements,
                'saturated_measurements': type_saturated_measurements,
                'percentage': type_percentage
            }
    
    # Find maximum intensity values
    max_intensities = df[wavelength_columns].max()
    overall_max = max_intensities.max()
    wavelength_with_max = max_intensities.idxmax()
    
    if verbose:
        print(f"\n📊 Overall Saturation Statistics:")
        print(f"   Total intensity values: {total_values:,}")
        print(f"   Saturated values: {saturated_values:,}")
        print(f"   Saturation percentage: {saturation_percentage:.4f}%")
        
        print(f"\n🔬 Measurement-Level Analysis:")
        print(f"   Measurements with saturation: {n_saturated_measurements}/{len(df)}")
        print(f"   Percentage of affected measurements: {measurement_saturation_percentage:.2f}%")
        
        print(f"\n🌊 Wavelength-Level Analysis:")
        print(f"   Wavelengths with saturation: {n_saturated_wavelengths}/{len(wavelength_columns)}")
        print(f"   Percentage of affected wavelengths: {wavelength_saturation_percentage:.2f}%")
        
        print(f"\n🔥 Top 10 Most Saturated Wavelengths:")
        print("-" * 40)
        for wavelength, count in most_saturated_wavelengths.items():
            percentage = (count / len(df)) * 100
            print(f"   λ{wavelength}: {count} measurements ({percentage:.2f}%)")
        
        print(f"\n🚨 Top 10 Most Saturated Measurements:")
        print("-" * 40)
        for idx, count in most_saturated_measurements.items():
            percentage = (count / len(wavelength_columns)) * 100
            sample_type = df.loc[idx, sample_type_col] if sample_type_col else 'Unknown'
            print(f"   Measurement {idx} ({sample_type}): {count} wavelengths ({percentage:.2f}%)")
        
        # Analysis by sample type
        if sample_type_saturation:
            print(f"\n🏷️  Saturation by Sample Type:")
            print("-" * 30)
            for sample_type, stats in sample_type_saturation.items():
                print(f"   {sample_type:12}: {stats['saturated_measurements']:3}/{stats['total_measurements']:3} ({stats['percentage']:6.2f}%)")
        
        print(f"\n📈 Maximum Intensity Analysis:")
        print(f"   Highest intensity: {overall_max:,.0f} at λ{wavelength_with_max}")
        print(f"   Saturation threshold: {saturation_threshold:,}")
        print(f"   Saturation ratio: {overall_max/saturation_threshold:.2f}x threshold")
    
    return {
        'saturated_mask': saturated_mask,
        'total_saturated_values': saturated_values,
        'saturation_percentage': saturation_percentage,
        'measurements_with_saturation': measurements_with_saturation,
        'n_saturated_measurements': n_saturated_measurements,
        'measurement_saturation_percentage': measurement_saturation_percentage,
        'wavelengths_with_saturation': wavelengths_with_saturation,
        'n_saturated_wavelengths': n_saturated_wavelengths,
        'most_saturated_wavelengths': most_saturated_wavelengths,
        'most_saturated_measurements': most_saturated_measurements,
        'sample_type_saturation': sample_type_saturation,
        'sample_type_column': sample_type_col,
        'max_intensity': overall_max,
        'wavelength_with_max': wavelength_with_max,
        'wavelength_columns': wavelength_columns,
        'saturation_threshold': saturation_threshold
    }

def plot_saturation_analysis(df: pd.DataFrame, 
                           saturation_results: Dict,
                           figsize: Tuple[int, int] = (15, 12),
                           save_path: Optional[str] = None) -> None:
    """
    Create comprehensive plots for saturation analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Original DataFrame
    saturation_results : dict
        Results from check_saturation function
    figsize : tuple, default=(15, 12)
        Figure size for plots
    save_path : str, optional
        Path to save the plot
    """
    
    if not saturation_results:
        print("❌ No saturation results to plot!")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('LIBS Data Saturation Analysis', fontsize=16, fontweight='bold')
    
    wavelength_columns = saturation_results['wavelength_columns']
    saturated_mask = saturation_results['saturated_mask']
    
    # 1. Saturation count per wavelength
    ax1 = axes[0, 0]
    saturation_counts = saturated_mask.sum(axis=0)
    wavelengths_numeric = [float(col) for col in wavelength_columns]
    
    ax1.plot(wavelengths_numeric, saturation_counts, linewidth=1, alpha=0.7)
    ax1.fill_between(wavelengths_numeric, saturation_counts, alpha=0.3)
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Number of Saturated Measurements')
    ax1.set_title('Saturation Count per Wavelength')
    ax1.grid(True, alpha=0.3)
    
    # 2. Saturation count per measurement
    ax2 = axes[0, 1]
    measurement_saturation_counts = saturated_mask.sum(axis=1)
    ax2.hist(measurement_saturation_counts, bins=50, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Number of Saturated Wavelengths')
    ax2.set_ylabel('Number of Measurements')
    ax2.set_title('Distribution of Saturated Wavelengths per Measurement')
    ax2.grid(True, alpha=0.3)
    
    # 3. Saturation heatmap (sample of data)
    ax3 = axes[0, 2]
    # Sample data for visualization (first 100 measurements, every 10th wavelength)
    sample_size = min(100, len(df))
    step_size = max(1, len(wavelength_columns) // 50)  # Show max 50 wavelengths
    
    if len(wavelength_columns) > 0:
        sample_mask = saturated_mask.iloc[:sample_size, ::step_size]
        sample_wavelengths = wavelength_columns[::step_size]
        
        sns.heatmap(sample_mask.T, cmap='Reds', cbar_kws={'label': 'Saturated'}, 
                    ax=ax3, xticklabels=False, 
                    yticklabels=[f'{float(w):.0f}' for w in sample_wavelengths[::max(1, len(sample_wavelengths)//10)]])
        ax3.set_title(f'Saturation Heatmap\n(First {sample_size} measurements)')
        ax3.set_xlabel('Measurement Index')
        ax3.set_ylabel('Wavelength (nm)')
    else:
        ax3.text(0.5, 0.5, 'No wavelength data\navailable', ha='center', va='center', transform=ax3.transAxes)
    
    # 4. Maximum intensity distribution
    ax4 = axes[1, 0]
    max_intensities_per_measurement = df[wavelength_columns].max(axis=1)
    ax4.hist(max_intensities_per_measurement, bins=50, alpha=0.7, edgecolor='black')
    ax4.axvline(x=saturation_results['saturation_threshold'], color='red', linestyle='--', linewidth=2, label='Saturation Threshold')
    ax4.set_xlabel('Maximum Intensity per Measurement')
    ax4.set_ylabel('Number of Measurements')
    ax4.set_title('Distribution of Maximum Intensities')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Sample type saturation (if available)
    ax5 = axes[1, 1]
    if saturation_results['sample_type_saturation']:
        sample_types = list(saturation_results['sample_type_saturation'].keys())
        percentages = [saturation_results['sample_type_saturation'][st]['percentage'] 
                      for st in sample_types]
        
        bars = ax5.bar(sample_types, percentages, alpha=0.7)
        ax5.set_xlabel('Sample Type')
        ax5.set_ylabel('Saturation Percentage (%)')
        ax5.set_title('Saturation by Sample Type')
        ax5.tick_params(axis='x', rotation=45)
        
        # Add percentage labels on bars
        for bar, pct in zip(bars, percentages):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{pct:.1f}%', ha='center', va='bottom')
    else:
        ax5.text(0.5, 0.5, 'No sample type\ninformation available', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Sample Type Analysis')
    
    # 6. Wavelength ranges with highest saturation
    ax6 = axes[1, 2]
    top_wavelengths = saturation_results['most_saturated_wavelengths'].head(15)
    if len(top_wavelengths) > 0:
        wavelength_values = [float(w) for w in top_wavelengths.index]
        counts = top_wavelengths.values
        
        bars = ax6.barh(range(len(wavelength_values)), counts, alpha=0.7)
        ax6.set_yticks(range(len(wavelength_values)))
        ax6.set_yticklabels([f'{w:.1f}' for w in wavelength_values])
        ax6.set_xlabel('Number of Saturated Measurements')
        ax6.set_ylabel('Wavelength (nm)')
        ax6.set_title('Top 15 Most Saturated Wavelengths')
        ax6.grid(True, alpha=0.3, axis='x')
    else:
        ax6.text(0.5, 0.5, 'No saturated\nwavelengths found', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Most Saturated Wavelengths')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def identify_problematic_data(df: pd.DataFrame, 
                            saturation_results: Dict,
                            severity_threshold: float = 0.1,
                            verbose: bool = True) -> Dict:
    """
    Identify measurements and wavelengths that should be flagged or removed.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Original DataFrame
    saturation_results : dict
        Results from check_saturation function
    severity_threshold : float, default=0.1
        Threshold for flagging problematic data (10% saturation)
    verbose : bool, default=True
        Whether to print analysis results
        
    Returns:
    --------
    dict : Recommendations for data handling
    """
    
    if not saturation_results:
        if verbose:
            print("❌ No saturation results provided!")
        return {}
    
    wavelength_columns = saturation_results['wavelength_columns']
    saturated_mask = saturation_results['saturated_mask']
    sample_type_col = saturation_results['sample_type_column']
    
    if verbose:
        print(f"\n🚨 Data Quality Assessment (threshold: {severity_threshold*100:.1f}% saturation)")
        print("=" * 70)
    
    # Problematic measurements
    measurement_saturation_rates = saturated_mask.sum(axis=1) / len(wavelength_columns)
    problematic_measurements = measurement_saturation_rates > severity_threshold
    n_problematic_measurements = problematic_measurements.sum()
    
    # Problematic wavelengths
    wavelength_saturation_rates = saturated_mask.sum(axis=0) / len(df)
    problematic_wavelengths = wavelength_saturation_rates > severity_threshold
    n_problematic_wavelengths = problematic_wavelengths.sum()
    
    if verbose:
        print(f"📊 Problematic Measurements:")
        print(f"   Measurements with >{severity_threshold*100:.1f}% wavelength saturation: {n_problematic_measurements}/{len(df)}")
        print(f"   Percentage of dataset: {(n_problematic_measurements/len(df))*100:.2f}%")
        
        print(f"\n🌊 Problematic Wavelengths:")
        print(f"   Wavelengths with >{severity_threshold*100:.1f}% measurement saturation: {n_problematic_wavelengths}/{len(wavelength_columns)}")
        print(f"   Percentage of spectrum: {(n_problematic_wavelengths/len(wavelength_columns))*100:.2f}%")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        
        if n_problematic_measurements == 0 and n_problematic_wavelengths == 0:
            print("   ✅ No significant saturation issues detected")
            print("   ✅ Data quality appears good for analysis")
        
        elif n_problematic_measurements > len(df) * 0.5:
            print("   ⚠️  More than 50% of measurements have significant saturation")
            print("   💡 Consider adjusting acquisition parameters (exposure time, laser power)")
            print("   💡 Review data collection methodology")
        
        elif n_problematic_wavelengths > len(wavelength_columns) * 0.1:
            print("   ⚠️  More than 10% of wavelengths show frequent saturation")
            print("   💡 Consider removing consistently saturated wavelengths")
            print("   💡 Check for systematic issues with specific spectral regions")
        
        else:
            print("   ⚡ Moderate saturation detected - manageable with preprocessing")
            if n_problematic_measurements > 0:
                print(f"   💡 Consider flagging {n_problematic_measurements} problematic measurements")
            if n_problematic_wavelengths > 0:
                print(f"   💡 Consider removing {n_problematic_wavelengths} problematic wavelengths")
        
        # Specific recommendations
        if problematic_measurements.any():
            worst_measurements = measurement_saturation_rates.nlargest(5)
            print(f"\n   🔍 Worst 5 measurements to review:")
            for idx, rate in worst_measurements.items():
                sample_type = df.loc[idx, sample_type_col] if sample_type_col else 'Unknown'
                print(f"      Measurement {idx} ({sample_type}): {rate*100:.1f}% saturated")
        
        if problematic_wavelengths.any():
            worst_wavelengths = wavelength_saturation_rates.nlargest(5)
            print(f"\n   🔍 Worst 5 wavelengths to consider removing:")
            for wavelength, rate in worst_wavelengths.items():
                print(f"      λ{wavelength}: {rate*100:.1f}% saturated")
    
    return {
        'problematic_measurements': problematic_measurements,
        'problematic_wavelengths': problematic_wavelengths,
        'measurement_saturation_rates': measurement_saturation_rates,
        'wavelength_saturation_rates': wavelength_saturation_rates,
        'severity_threshold': severity_threshold,
        'n_problematic_measurements': n_problematic_measurements,
        'n_problematic_wavelengths': n_problematic_wavelengths
    }

def generate_saturation_summary(saturation_results: Dict, verbose: bool = True) -> Dict:
    """
    Generate a summary of saturation analysis results.
    
    Parameters:
    -----------
    saturation_results : dict
        Results from check_saturation function
    verbose : bool, default=True
        Whether to print summary
        
    Returns:
    --------
    dict : Summary statistics
    """
    
    if not saturation_results:
        return {}
    
    saturation_pct = saturation_results['saturation_percentage']
    measurement_pct = saturation_results['measurement_saturation_percentage']
    wavelength_pct = (saturation_results['n_saturated_wavelengths'] / 
                     len(saturation_results['wavelength_columns'])) * 100
    
    # Determine quality level
    if saturation_pct < 0.1:
        quality = "Excellent"
        emoji = "✅"
        description = "minimal saturation detected"
    elif saturation_pct < 1.0:
        quality = "Good"
        emoji = "⚡"
        description = "minor saturation issues"
    elif saturation_pct < 5.0:
        quality = "Moderate"
        emoji = "⚠️ "
        description = "requires attention"
    else:
        quality = "Poor"
        emoji = "🚨"
        description = "major data quality concerns"
    
    summary = {
        'overall_saturation_percentage': saturation_pct,
        'affected_measurements_percentage': measurement_pct,
        'affected_wavelengths_percentage': wavelength_pct,
        'quality_level': quality,
        'quality_emoji': emoji,
        'quality_description': description,
        'max_intensity': saturation_results['max_intensity'],
        'max_intensity_wavelength': saturation_results['wavelength_with_max'],
        'total_measurements': len(saturation_results['saturated_mask']),
        'total_wavelengths': len(saturation_results['wavelength_columns'])
    }
    
    if verbose:
        print(f"\n📋 Summary:")
        print("=" * 40)
        print(f"{emoji} {quality} data quality - {description}")
        print(f"Overall saturation: {saturation_pct:.4f}%")
        print(f"Affected measurements: {measurement_pct:.2f}%")
        print(f"Affected wavelengths: {wavelength_pct:.2f}%")
    
    return summary

def run_complete_saturation_analysis(df: pd.DataFrame,
                                   saturation_threshold: int = 65535,
                                   severity_threshold: float = 0.05,
                                   plot: bool = True,
                                   figsize: Tuple[int, int] = (15, 12),
                                   save_plot: Optional[str] = None) -> Dict:
    """
    Run complete saturation analysis pipeline.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing LIBS measurements
    saturation_threshold : int, default=65535
        Intensity threshold for saturation detection
    severity_threshold : float, default=0.05
        Threshold for flagging problematic data
    plot : bool, default=True
        Whether to generate plots
    figsize : tuple, default=(15, 12)
        Figure size for plots
    save_plot : str, optional
        Path to save the plot
        
    Returns:
    --------
    dict : Complete analysis results
    """
    
    print("🔍 Starting comprehensive saturation analysis...")
    
    # Run saturation check
    saturation_results = check_saturation(df, saturation_threshold, verbose=True)
    
    if not saturation_results:
        return {}
    
    # Generate plots
    if plot:
        print("\n📊 Generating saturation plots...")
        plot_saturation_analysis(df, saturation_results, figsize=figsize, save_path=save_plot)
    
    # Identify problematic data
    print(f"\n🚨 Identifying problematic measurements and wavelengths...")
    quality_assessment = identify_problematic_data(df, saturation_results, 
                                                 severity_threshold=severity_threshold,
                                                 verbose=True)
    
    # Generate summary
    summary = generate_saturation_summary(saturation_results, verbose=True)
    
    return {
        'saturation_results': saturation_results,
        'quality_assessment': quality_assessment,
        'summary': summary
    }

def diagnose_data_structure(df: pd.DataFrame) -> None:
    """
    Diagnose DataFrame structure for saturation analysis debugging.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to diagnose
    """
    
    print("🔍 DIAGNOSTIC INFORMATION")
    print("=" * 50)
    
    print(f"DataFrame shape: {df.shape}")
    print(f"Column count: {len(df.columns)}")
    print(f"Column names sample: {list(df.columns[:10])}")
    print(f"Data types: {df.dtypes.value_counts().to_dict()}")
    
    # Check for wavelength columns
    wavelength_columns = get_wavelength_columns(df)
    print(f"\nWavelength columns detected: {len(wavelength_columns)}")
    
    if wavelength_columns:
        print(f"Sample wavelength columns: {wavelength_columns[:5]}")
        
        # Check intensity ranges
        intensity_data = df[wavelength_columns]
        print(f"\nIntensity statistics:")
        print(f"Min value: {intensity_data.min().min()}")
        print(f"Max value: {intensity_data.max().max()}")
        print(f"Mean value: {intensity_data.mean().mean():.2f}")
        
        # Check for values near saturation threshold
        thresholds = [50000, 60000, 65535]
        for threshold in thresholds:
            count = (intensity_data >= threshold).sum().sum()
            percentage = (count / (len(df) * len(wavelength_columns))) * 100
            print(f"Values >= {threshold}: {count} ({percentage:.4f}%)")
    else:
        print("❌ No wavelength columns detected!")
        print(f"All columns: {list(df.columns)}")

def voigt_profile(x: np.ndarray, amplitude: float, center: float, 
                 sigma: float, gamma: float, offset: float = 0) -> np.ndarray:
    """
    Voigt profile function combining Gaussian and Lorentzian profiles.
    
    Parameters:
    -----------
    x : np.ndarray
        Independent variable (wavelength)
    amplitude : float
        Peak amplitude
    center : float
        Peak center position
    sigma : float
        Gaussian width parameter
    gamma : float
        Lorentzian width parameter  
    offset : float, default=0
        Baseline offset
        
    Returns:
    --------
    np.ndarray : Voigt profile values
    """
    # Voigt profile using Faddeeva function
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    voigt = amplitude * np.real(special.wofz(z)) / (sigma * np.sqrt(2 * np.pi))
    return voigt + offset

def find_saturated_peaks(intensities: np.ndarray, wavelengths: np.ndarray, 
                        saturation_threshold: float = 65535,
                        min_peak_width: int = 3) -> List[Dict]:
    """
    Identify saturated spectral peaks and their boundaries.
    
    Parameters:
    -----------
    intensities : np.ndarray
        Intensity values for a single measurement
    wavelengths : np.ndarray
        Corresponding wavelength values
    saturation_threshold : float
        Saturation threshold value
    min_peak_width : int
        Minimum width for a peak to be considered
        
    Returns:
    --------
    List[Dict] : List of saturated peak information
    """
    saturated_mask = intensities >= saturation_threshold
    peaks: List[Dict[str, Union[int, float]]] = []
    
    if not np.any(saturated_mask):
        return peaks
    
    # Find continuous saturated regions
    diff = np.diff(np.concatenate(([False], saturated_mask, [False])).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    for start, end in zip(starts, ends):
        if end - start >= min_peak_width:
            # Extend region to include shoulders for fitting
            fit_start = max(0, start - 10)
            fit_end = min(len(intensities), end + 10)
            
            peak_info = {
                'saturated_start': start,
                'saturated_end': end,
                'fit_start': fit_start,
                'fit_end': fit_end,
                'peak_center_idx': start + (end - start) // 2,
                'peak_center_wavelength': wavelengths[start + (end - start) // 2],
                'saturated_width': end - start,
                'fit_width': fit_end - fit_start
            }
            peaks.append(peak_info)
    
    return peaks

def fit_voigt_to_peak(intensities: np.ndarray, wavelengths: np.ndarray,
                     peak_info: Dict, saturation_threshold: float = 65535) -> Dict:
    """
    Fit Voigt profile to a saturated peak and estimate true intensities.
    
    Parameters:
    -----------
    intensities : np.ndarray
        Intensity values for the measurement
    wavelengths : np.ndarray
        Corresponding wavelength values
    peak_info : Dict
        Peak information from find_saturated_peaks
    saturation_threshold : float
        Saturation threshold value
        
    Returns:
    --------
    Dict : Fitting results and corrected intensities
    """
    fit_start = peak_info['fit_start']
    fit_end = peak_info['fit_end']
    sat_start = peak_info['saturated_start']
    sat_end = peak_info['saturated_end']
    
    # Extract fitting region
    x_fit = wavelengths[fit_start:fit_end]
    y_fit = intensities[fit_start:fit_end].copy()
    
    # Create mask for non-saturated points to use in fitting
    non_saturated_mask = y_fit < saturation_threshold
    
    if np.sum(non_saturated_mask) < 5:
        # Not enough non-saturated points for reliable fitting
        return {
            'success': False,
            'reason': 'Insufficient non-saturated points',
            'corrected_intensities': None,
            'fit_params': None
        }
    
    # Use only non-saturated points for fitting
    x_fit_clean = x_fit[non_saturated_mask]
    y_fit_clean = y_fit[non_saturated_mask]
    
    # Initial parameter estimates
    center_estimate = peak_info['peak_center_wavelength']
    amplitude_estimate = np.max(y_fit_clean) * 2  # Estimate higher than observed max
    sigma_estimate = (x_fit[-1] - x_fit[0]) / 6   # Rough width estimate
    gamma_estimate = sigma_estimate / 2
    offset_estimate = np.min(y_fit_clean)
    
    initial_params = [amplitude_estimate, center_estimate, sigma_estimate, 
                     gamma_estimate, offset_estimate]
    
    # Parameter bounds
    bounds = (
        [0, x_fit[0], 0, 0, 0],  # Lower bounds
        [amplitude_estimate * 5, x_fit[-1], (x_fit[-1] - x_fit[0]) / 2, 
         (x_fit[-1] - x_fit[0]) / 2, np.max(y_fit_clean)]  # Upper bounds
    )
    
    try:
        # Fit Voigt profile
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(voigt_profile, x_fit_clean, y_fit_clean,
                                 p0=initial_params, bounds=bounds, maxfev=2000)
        
        # Calculate corrected intensities for the entire fitting region
        y_corrected = voigt_profile(x_fit, *popt)
        
        # Calculate fitting quality metrics
        y_pred_clean = voigt_profile(x_fit_clean, *popt)
        r_squared = 1 - np.sum((y_fit_clean - y_pred_clean)**2) / np.sum((y_fit_clean - np.mean(y_fit_clean))**2)
        rmse = np.sqrt(np.mean((y_fit_clean - y_pred_clean)**2))
        
        # Check if fit is reasonable
        peak_amplitude = popt[0]
        if peak_amplitude > saturation_threshold * 10 or r_squared < 0.5:
            return {
                'success': False,
                'reason': f'Unrealistic fit: amplitude={peak_amplitude:.0f}, R²={r_squared:.3f}',
                'corrected_intensities': None,
                'fit_params': None
            }
        
        return {
            'success': True,
            'corrected_intensities': y_corrected,
            'fit_params': {
                'amplitude': popt[0],
                'center': popt[1], 
                'sigma': popt[2],
                'gamma': popt[3],
                'offset': popt[4]
            },
            'fit_quality': {
                'r_squared': r_squared,
                'rmse': rmse,
                'param_errors': np.sqrt(np.diag(pcov)) if pcov is not None else None
            },
            'fit_region': {
                'wavelengths': x_fit,
                'original_intensities': y_fit,
                'fit_points_mask': non_saturated_mask
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'reason': f'Fitting failed: {str(e)}',
            'corrected_intensities': None,
            'fit_params': None
        }

def apply_voigt_correction_to_measurement(intensities: np.ndarray, 
                                        wavelengths: np.ndarray,
                                        saturation_threshold: float = 65535,
                                        min_peak_width: int = 3,
                                        verbose: bool = False) -> Dict:
    """
    Apply Voigt profile correction to all saturated peaks in a single measurement.
    
    Parameters:
    -----------
    intensities : np.ndarray
        Intensity values for the measurement
    wavelengths : np.ndarray
        Corresponding wavelength values
    saturation_threshold : float
        Saturation threshold
    min_peak_width : int
        Minimum peak width to consider
    verbose : bool
        Print detailed information
        
    Returns:
    --------
    Dict : Correction results
    """
    # Find saturated peaks
    saturated_peaks = find_saturated_peaks(intensities, wavelengths, 
                                         saturation_threshold, min_peak_width)
    
    if not saturated_peaks:
        return {
            'success': True,
            'corrected_intensities': intensities.copy(),
            'n_peaks_found': 0,
            'n_peaks_corrected': 0,
            'correction_applied': False,
            'peak_corrections': []
        }
    
    # Initialize corrected intensities
    corrected_intensities = intensities.copy()
    successful_corrections = 0
    peak_corrections = []
    
    for i, peak_info in enumerate(saturated_peaks):
        if verbose:
            print(f"   Processing peak {i+1}/{len(saturated_peaks)} at λ{peak_info['peak_center_wavelength']:.2f}")
        
        # Fit Voigt profile to this peak
        fit_result = fit_voigt_to_peak(intensities, wavelengths, peak_info, saturation_threshold)
        
        if fit_result['success']:
            # Apply correction to saturated region only
            fit_start = peak_info['fit_start']
            fit_end = peak_info['fit_end']
            sat_start = peak_info['saturated_start']
            sat_end = peak_info['saturated_end']
            
            # Replace saturated intensities with fitted values
            sat_start_in_fit = sat_start - fit_start
            sat_end_in_fit = sat_end - fit_start
            
            corrected_intensities[sat_start:sat_end] = fit_result['corrected_intensities'][sat_start_in_fit:sat_end_in_fit]
            successful_corrections += 1
            
            peak_corrections.append({
                'peak_index': i,
                'wavelength_range': (wavelengths[sat_start], wavelengths[sat_end-1]),
                'original_max': saturation_threshold,
                'corrected_max': np.max(fit_result['corrected_intensities'][sat_start_in_fit:sat_end_in_fit]),
                'fit_params': fit_result['fit_params'],
                'fit_quality': fit_result['fit_quality']
            })
            
            if verbose:
                corrected_max = np.max(fit_result['corrected_intensities'][sat_start_in_fit:sat_end_in_fit])
                print(f"     ✅ Corrected: {saturation_threshold:.0f} → {corrected_max:.0f} "
                      f"(R² = {fit_result['fit_quality']['r_squared']:.3f})")
        else:
            if verbose:
                print(f"     ❌ Failed: {fit_result['reason']}")
            
            peak_corrections.append({
                'peak_index': i,
                'wavelength_range': (wavelengths[peak_info['saturated_start']], 
                                   wavelengths[peak_info['saturated_end']-1]),
                'correction_failed': True,
                'failure_reason': fit_result['reason']
            })
    
    return {
        'success': True,
        'corrected_intensities': corrected_intensities,
        'n_peaks_found': len(saturated_peaks),
        'n_peaks_corrected': successful_corrections,
        'correction_applied': successful_corrections > 0,
        'peak_corrections': peak_corrections,
        'success_rate': successful_corrections / len(saturated_peaks) if saturated_peaks else 0
    }

def apply_voigt_saturation_correction(df: pd.DataFrame,
                                    saturation_threshold: float = 65535,
                                    min_peak_width: int = 3,
                                    max_measurements: Optional[int] = None,
                                    verbose: bool = True) -> pd.DataFrame:
    """
    Apply Voigt profile correction to all saturated measurements in the DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing LIBS measurements
    saturation_threshold : float
        Saturation threshold
    min_peak_width : int
        Minimum peak width to consider for correction
    max_measurements : int, optional
        Maximum number of measurements to process (for testing)
    verbose : bool
        Print progress information
        
    Returns:
    --------
    pd.DataFrame : DataFrame with corrected intensities
    """
    
    # Get wavelength columns
    wavelength_columns = get_wavelength_columns(df)
    wavelengths = np.array([float(col) for col in wavelength_columns])
    
    if len(wavelength_columns) == 0:
        print("❌ No wavelength columns found!")
        return df.copy()
    
    # Find saturated measurements
    saturated_mask = df[wavelength_columns] >= saturation_threshold
    measurements_with_saturation = saturated_mask.any(axis=1)
    saturated_measurement_indices = df.index[measurements_with_saturation].tolist()
    
    if max_measurements:
        saturated_measurement_indices = saturated_measurement_indices[:max_measurements]
    
    n_saturated = len(saturated_measurement_indices)
    
    if verbose:
        print(f"\n🔧 Voigt Profile Saturation Correction")
        print("=" * 50)
        print(f"Total measurements: {len(df)}")
        print(f"Saturated measurements: {n_saturated}")
        print(f"Processing measurements: {len(saturated_measurement_indices)}")
        print(f"Saturation threshold: {saturation_threshold}")
        print(f"Minimum peak width: {min_peak_width}")
    
    if n_saturated == 0:
        print("✅ No saturated measurements found - no correction needed")
        return df.copy()
    
    # Create copy of DataFrame for corrections
    df_corrected = df.copy()
    
    # Statistics tracking
    total_peaks_found = 0
    total_peaks_corrected = 0
    successful_measurements = 0
    
    # Process each saturated measurement
    for i, measurement_idx in enumerate(saturated_measurement_indices):
        if verbose and (i + 1) % max(1, len(saturated_measurement_indices) // 10) == 0:
            print(f"Progress: {i+1}/{len(saturated_measurement_indices)} ({(i+1)/len(saturated_measurement_indices)*100:.1f}%)")
        
        # Get intensities for this measurement
        intensities = df.loc[measurement_idx, wavelength_columns].values.astype(float)
        
        # Apply correction
        correction_result = apply_voigt_correction_to_measurement(
            intensities, wavelengths, saturation_threshold, min_peak_width, 
            verbose=False
        )
        
        if correction_result['success'] and correction_result['correction_applied']:
            # Update DataFrame with corrected values
            corrected_values = correction_result['corrected_intensities'].astype(df_corrected[wavelength_columns].dtypes.iloc[0])
            df_corrected.loc[measurement_idx, wavelength_columns] = corrected_values
            successful_measurements += 1
            
            total_peaks_found += correction_result['n_peaks_found']
            total_peaks_corrected += correction_result['n_peaks_corrected']
    
    # Print summary
    if verbose:
        print(f"\n📊 Correction Summary:")
        print("-" * 30)
        print(f"Measurements processed: {len(saturated_measurement_indices)}")
        print(f"Measurements corrected: {successful_measurements}")
        print(f"Success rate: {successful_measurements/len(saturated_measurement_indices)*100:.1f}%")
        print(f"Total peaks found: {total_peaks_found}")
        print(f"Total peaks corrected: {total_peaks_corrected}")
        print(f"Peak correction rate: {total_peaks_corrected/total_peaks_found*100:.1f}%" if total_peaks_found > 0 else "0.0%")
        
        # Compare before and after
        original_max = df[wavelength_columns].max().max()
        corrected_max = df_corrected[wavelength_columns].max().max()
        print(f"\nIntensity ranges:")
        print(f"Original max: {original_max:,.0f}")
        print(f"Corrected max: {corrected_max:,.0f}")
        print(f"Max increase: {corrected_max/original_max:.2f}x")
    
    return df_corrected

def plot_voigt_correction_examples(df_original: pd.DataFrame, 
                                 df_corrected: pd.DataFrame,
                                 n_examples: int = 3,
                                 saturation_threshold: float = 65535,
                                 figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Plot examples of Voigt profile corrections.
    
    Parameters:
    -----------
    df_original : pd.DataFrame
        Original DataFrame with saturated data
    df_corrected : pd.DataFrame
        DataFrame after Voigt correction
    n_examples : int
        Number of example measurements to plot
    saturation_threshold : float
        Saturation threshold for visualization
    figsize : tuple
        Figure size
    """
    
    wavelength_columns = get_wavelength_columns(df_original)
    wavelengths = np.array([float(col) for col in wavelength_columns])
    
    # Find measurements with saturated peaks
    saturated_mask = df_original[wavelength_columns] >= saturation_threshold
    measurements_with_saturation = saturated_mask.any(axis=1)
    saturated_indices = df_original.index[measurements_with_saturation].tolist()
    
    if len(saturated_indices) == 0:
        print("No saturated measurements found for plotting")
        return
    
    # Select examples
    example_indices = saturated_indices[:n_examples]
    
    fig, axes = plt.subplots(n_examples, 1, figsize=figsize, squeeze=False)
    fig.suptitle('Voigt Profile Saturation Correction Examples', fontsize=14, fontweight='bold')
    
    for i, measurement_idx in enumerate(example_indices):
        ax = axes[i, 0]
        
        # Get original and corrected intensities
        original_intensities = df_original.loc[measurement_idx, wavelength_columns].values.astype(float)
        corrected_intensities = df_corrected.loc[measurement_idx, wavelength_columns].values.astype(float)
        
        # Plot both spectra
        ax.plot(wavelengths, original_intensities, 'b-', alpha=0.7, linewidth=1, label='Original')
        ax.plot(wavelengths, corrected_intensities, 'r-', alpha=0.8, linewidth=1, label='Voigt Corrected')
        
        # Mark saturation threshold
        ax.axhline(y=saturation_threshold, color='orange', linestyle='--', alpha=0.7, label='Saturation Threshold')
        
        # Highlight corrected regions
        saturated_mask_single = original_intensities >= saturation_threshold
        if np.any(saturated_mask_single):
            corrected_regions = wavelengths[saturated_mask_single]
            for wavelength in corrected_regions:
                ax.axvline(x=wavelength, color='red', alpha=0.1, linewidth=0.5)
        
        # Get sample info
        origin = df_original.loc[measurement_idx, 'origin'] if 'origin' in df_original.columns else 'Unknown'
        tire_num = df_original.loc[measurement_idx, 'tire_number'] if 'tire_number' in df_original.columns else 'Unknown'
        
        ax.set_title(f'Measurement {measurement_idx} ({origin}, Tire {tire_num})')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Intensity')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set y-axis to show the correction clearly
        max_corrected = np.max(corrected_intensities)
        ax.set_ylim(0, max_corrected * 1.1)
    
    plt.tight_layout()
    plt.show()

def validate_voigt_corrections(df_original: pd.DataFrame, 
                             df_corrected: pd.DataFrame,
                             saturation_threshold: float = 65535,
                             verbose: bool = True) -> Dict:
    """
    Validate the quality of Voigt profile corrections.
    
    Parameters:
    -----------
    df_original : pd.DataFrame
        Original DataFrame
    df_corrected : pd.DataFrame  
        Corrected DataFrame
    saturation_threshold : float
        Saturation threshold
    verbose : bool
        Print validation results
        
    Returns:
    --------
    Dict : Validation metrics
    """
    
    wavelength_columns = get_wavelength_columns(df_original)
    
    # Find originally saturated values
    original_saturated_mask = df_original[wavelength_columns] >= saturation_threshold
    n_originally_saturated = original_saturated_mask.sum().sum()
    
    # Check corrected values
    corrected_values = df_corrected[wavelength_columns].values
    original_values = df_original[wavelength_columns].values
    
    # Find where corrections were applied
    corrections_applied = ~np.isclose(corrected_values, original_values, rtol=1e-6)
    n_values_changed = corrections_applied.sum()
    
    # Statistics on corrections - pure numpy vectorized approach
    correction_ratios = []

    # Create masks
    original_saturated = original_saturated_mask.values
    corrected_higher = corrected_values > original_values

    # Combined condition: originally saturated AND correction made it higher
    valid_corrections = original_saturated & corrected_higher

    # Calculate ratios only where valid corrections occurred
    if np.any(valid_corrections):
        correction_ratios = (corrected_values[valid_corrections] / 
                            original_values[valid_corrections]).tolist()
    
    validation_results = {
        'n_originally_saturated': n_originally_saturated,
        'n_values_changed': n_values_changed,
        'correction_coverage': n_values_changed / n_originally_saturated if n_originally_saturated > 0 else 0,
        'correction_ratios': correction_ratios,
        'mean_correction_ratio': np.mean(correction_ratios) if correction_ratios else 1.0,
        'max_correction_ratio': np.max(correction_ratios) if correction_ratios else 1.0,
        'min_correction_ratio': np.min(correction_ratios) if correction_ratios else 1.0,
        'original_max_intensity': original_values.max(),
        'corrected_max_intensity': corrected_values.max()
    }
    
    if verbose:
        print(f"\n✅ Voigt Correction Validation:")
        print("-" * 40)
        print(f"Originally saturated values: {n_originally_saturated:,}")
        print(f"Values changed by correction: {n_values_changed:,}")
        print(f"Correction coverage: {validation_results['correction_coverage']*100:.1f}%")
        print(f"Mean correction ratio: {validation_results['mean_correction_ratio']:.2f}x")
        print(f"Max correction ratio: {validation_results['max_correction_ratio']:.2f}x")
        print(f"Original max intensity: {validation_results['original_max_intensity']:,.0f}")
        print(f"Corrected max intensity: {validation_results['corrected_max_intensity']:,.0f}")
        
        if validation_results['correction_coverage'] > 0.8:
            print("🎉 Excellent correction coverage!")
        elif validation_results['correction_coverage'] > 0.5:
            print("⚡ Good correction coverage")
        else:
            print("⚠️  Limited correction coverage - consider adjusting parameters")
    
    return validation_results

