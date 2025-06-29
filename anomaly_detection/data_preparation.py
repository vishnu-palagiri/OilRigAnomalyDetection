from scipy.signal import medfilt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def impute_column(series, strategy):
    """
    Imputes missing values in a pandas Series using the specified strategy.

    Parameters:
    - series (pd.Series): The input data column.
    - strategy (str): Imputation method — 'interpolate', 'ffill', or None.

    Returns:
    - pd.Series: The imputed series.
    """
    if strategy == 'interpolate':
        return series.interpolate(limit_direction='both')
    elif strategy == 'ffill':
        return series.ffill().bfill()
    return series


def denoise_column(series, window_size=5, threshold=3.5):
    """
    Applies median filtering and replaces values deviating beyond a MAD-based threshold.

    Parameters:
    - series (pd.Series): Numeric column to denoise.
    - window_size (int): Kernel size for the median filter (must be odd).
    - threshold (float): Threshold multiplier for outlier detection.

    Returns:
    - pd.Series: Denoised series.
    """
    median = medfilt(series, kernel_size=window_size)
    diff = np.abs(series - median)
    mad = np.median(diff)
    if mad == 0:
        return series
    mask = diff > threshold * mad
    series[mask] = median[mask]
    return series


def scale_column(series):
    """
    Standardizes a numeric series using z-score normalization.

    Parameters:
    - series (pd.Series): Input data to scale.

    Returns:
    - scaled (pd.Series): Scaled output.
    - scaler (StandardScaler): Fitted scaler object.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    return pd.Series(scaled, index=series.index), scaler


def apply_preprocessing(df, strategy_dict, threshold=3.5):
    """
    Applies preprocessing steps column-wise based on configuration:
    imputation, denoising, and scaling.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - strategy_dict (dict): Column-wise preprocessing options:
        {
            'column_name': {
                'impute': 'ffill' | 'interpolate' | None,
                'denoise': bool,
                'scale': bool
            },
            ...
        }
    - threshold (float): MAD threshold for denoising.

    Returns:
    - df_out (pd.DataFrame): Processed DataFrame.
    - scalers (dict): Dictionary of fitted scalers by column (if any).
    """
    df_out = pd.DataFrame(index=df.index)
    scalers = {}

    for col in df.columns:
        strat = strategy_dict.get(col, {})
        series = df[col]

        # Imputation
        if strat.get('impute'):
            series = impute_column(series, strat['impute'])

        # Denoising
        if strat.get('denoise', False) and pd.api.types.is_numeric_dtype(series):
            series = denoise_column(series, window_size=5, threshold=threshold)

        # Scaling
        if strat.get('scale', False) and pd.api.types.is_numeric_dtype(series):
            scaled_series, scaler = scale_column(series)
            scalers[col] = scaler
        else:
            scaled_series = series

        df_out[col] = scaled_series
        df[col] = series

    return df_out, df, scalers