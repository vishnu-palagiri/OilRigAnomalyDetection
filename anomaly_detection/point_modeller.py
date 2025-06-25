import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def zscore_detector(df, threshold=3.0):
    """
    Detects point anomalies using Z-score thresholding per column.

    Parameters:
    - df (pd.DataFrame): Input data with numeric columns.
    - threshold (float): Z-score threshold for anomaly detection.

    Returns:
    - pd.DataFrame: Boolean DataFrame indicating anomalies per column.
    """
    anomalies = {}
    for col in df.columns:
        series = df[col]
        z = (series - series.mean()) / series.std()
        anomalies[col] = np.abs(z) > threshold
    return pd.DataFrame(anomalies, index=df.index)


def isolation_forest_detector(df, contamination=0.02, random_state=32):
    """
    Uses Isolation Forest to detect point anomalies.

    Parameters:
    - df (pd.DataFrame): Input data with numeric features.
    - contamination (float): Expected proportion of anomalies.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - pd.Series: Boolean series indicating anomalous rows (True = anomaly).
    """
    clf = IsolationForest(contamination=contamination, random_state=random_state)
    preds = clf.fit_predict(df)
    return pd.Series(preds == -1, index=df.index)


def run_point_anomaly_detectors(df, methods=['zscore', 'isoforest'],
                                 threshold=3.0, contamination=0.02, random_state=32):
    """
    Runs selected point anomaly detection methods and aggregates results.

    Parameters:
    - df (pd.DataFrame): Preprocessed input data.
    - methods (list): Methods to use ('zscore', 'isoforest').
    - threshold (float): Z-score threshold for anomaly detection.
    - contamination (float): Expected anomaly ratio for Isolation Forest.
    - random_state (int): Seed for Isolation Forest reproducibility.

    Returns:
    - results (dict): Dictionary of individual method outputs.
    - combined (pd.Series): Boolean series combining all detectors.
    """
    results = {}

    if 'zscore' in methods:
        results['zscore'] = zscore_detector(df, threshold)

    if 'isoforest' in methods:
        results['isoforest'] = isolation_forest_detector(df, contamination, random_state)

    # Optional aggregation: combines using logical OR
    combined = None
    if 'zscore' in results and 'isoforest' in results:
        z = results['zscore'].any(axis=1)
        iso = results['isoforest']
        combined = z | iso

    return results, combined