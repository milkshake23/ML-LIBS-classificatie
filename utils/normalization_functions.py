"""
    Normalization functions for spectral data
"""
import numpy as np # type: ignore[import]
import pandas as pd # type: ignore[import]

class Normalizer:
    """ Collection of standard normal variate (SNV) normalization methods for spectral data
    """
    def __init__(self):
        pass

    @staticmethod
    def apply_standard_normal_variate_on_multiple_spectra(spectra_matrix):
        """
        Apply Standard Normal Variate (SNV) normalization to spectra
        
        SNV removes multiplicative scatter effects by centering and scaling each spectrum 
        individually to unit variance.
        
        Formula: SNV(x) = (x - mean(x)) / std(x)
        
        Parameters:
        spectra_matrix: 2D array where each row is a spectrum
        
        Returns:
        normalized_spectra: SNV-normalized spectra
        """
        print(f"Applying SNV normalization to {len(spectra_matrix)} spectra...")
        
        # Calculate mean and std for each spectrum (row-wise)
        means = np.mean(spectra_matrix, axis=1, keepdims=True)
        stds = np.std(spectra_matrix, axis=1, keepdims=True, ddof=0)
        
        # Avoid division by zero
        stds = np.where(stds == 0, 1, stds)
        
        # Apply SNV normalization
        normalized_spectra = (spectra_matrix - means) / stds
        
        return normalized_spectra.astype(np.float32)

    @staticmethod
    def apply_standard_normal_variate_on_dataframe(df):
        """
        Apply Standard Normal Variate (SNV) normalization to all numeric columns in a DataFrame.
        Non-numeric columns are ignored.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(f"Applying SNV normalization to {len(numeric_cols)} numeric columns...")
        
        df_normalized = df.copy()
        
        for col in numeric_cols:
            col_mean = df[col].mean()
            col_std = df[col].std(ddof=0)
            if col_std == 0:
                col_std = 1  # Prevent division by zero
            df_normalized[col] = (df[col] - col_mean) / col_std
        
        return df_normalized.astype(np.float32)

    @staticmethod
    def apply_standard_normal_variate_on_dataset(X_train):
        """
        Apply Standard Normal Variate (SNV) normalization to baseline-corrected spectral dataset.
        
        This function normalizes each spectrum (row) individually by centering and scaling
        to unit variance. Specifically designed for LIBS spectral data with wavelength columns.
        
        Parameters:
        -----------
        X_train : pandas.DataFrame or numpy.ndarray
            Training dataset containing baseline-corrected spectral data
            - If DataFrame: columns should be wavelength values (as strings or numbers)
            - If numpy array: each row is a spectrum, each column is a wavelength
        
        Returns:
        --------
        X_train_normalized : same type as input
            SNV-normalized spectral data with same structure as input
            
        Notes:
        ------
        - SNV formula: SNV(x) = (x - mean(x)) / std(x) for each spectrum
        - Removes multiplicative scatter effects while preserving spectral shape
        - Each spectrum is normalized independently (row-wise operation)
        """
        
        if isinstance(X_train, pd.DataFrame):
            return Normalizer._normalize_dataframe_dataset(X_train)
        elif isinstance(X_train, np.ndarray):
            return Normalizer._normalize_array_dataset(X_train)
        else:
            raise TypeError("X_train must be either pandas DataFrame or numpy ndarray")

    @staticmethod
    def _normalize_dataframe_dataset(X_train):
        """Handle DataFrame input for SNV normalization"""
        
        print(f"Applying SNV normalization to dataset:")
        print(f"  - Samples: {len(X_train)}")
        
        # Create copy to avoid modifying original data
        X_normalized = X_train.copy()
        
        # Extract spectral data for normalization
        spectral_data = X_train.values.astype(np.float64)
        
        # Apply SNV normalization row-wise
        means = np.mean(spectral_data, axis=1, keepdims=True)
        stds = np.std(spectral_data, axis=1, keepdims=True, ddof=0)
        
        # Avoid division by zero (constant spectra)
        stds = np.where(stds == 0, 1, stds)
        
        # Apply SNV formula
        normalized_spectral_data = (spectral_data - means) / stds
        
        # Replace wavelength columns with normalized values
        X_normalized = pd.DataFrame(normalized_spectral_data.astype(np.float32), 
                                   columns=X_train.columns, 
                                   index=X_train.index)
        
        # Verify normalization
        sample_spectrum = normalized_spectral_data[0, :]
        print(f"  - Verification (first spectrum): mean={np.mean(sample_spectrum):.6f}, std={np.std(sample_spectrum, ddof=0):.6f}")
        print(f"  - SNV normalization completed successfully!")
        
        return X_normalized

    @staticmethod
    def _normalize_array_dataset(X_train):
        """Handle numpy array input for SNV normalization"""
        
        print(f"Applying SNV normalization to numpy array:")
        print(f"  - Shape: {X_train.shape}")
        print(f"  - Samples: {X_train.shape[0]}")
        print(f"  - Features (wavelengths): {X_train.shape[1]}")
        
        # Convert to float64 for precision during calculation
        spectral_data = X_train.astype(np.float64)
        
        # Apply SNV normalization row-wise
        means = np.mean(spectral_data, axis=1, keepdims=True)
        stds = np.std(spectral_data, axis=1, keepdims=True, ddof=0)
        
        # Avoid division by zero (constant spectra)
        stds = np.where(stds == 0, 1, stds)
        
        # Apply SNV formula
        normalized_data = (spectral_data - means) / stds
        
        # Verify normalization
        sample_spectrum = normalized_data[0, :]
        print(f"  - Verification (first spectrum): mean={np.mean(sample_spectrum):.6f}, std={np.std(sample_spectrum, ddof=0):.6f}")
        print(f"  - SNV normalization completed successfully!")
        
        return normalized_data.astype(np.float32)