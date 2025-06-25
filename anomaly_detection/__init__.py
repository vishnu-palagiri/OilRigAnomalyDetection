from utils import *
from anomaly_detection.continuous_modeller import *
from anomaly_detection.data_preparation import *
from anomaly_detection.embed import *
from anomaly_detection.point_modeller import *

import numpy as np
import pandas as pd

# Features to consider for modeling
attrs_to_use = ['WellHeadPressure', 'WellHeadTemperature', 'TubingPressure', 'FlowRate']


def load_sensor_data(file_path, timestamp_col='Timestamp'):
    """Load and index sensor time-series data."""
    df = pd.read_csv(file_path, parse_dates=[timestamp_col])
    df.set_index(timestamp_col, inplace=True)
    return df


def load_log_data(file_path, timestamp_cols=['StartTimestamp', 'EndTimestamp']):
    """Load maintenance log file with timestamps parsed."""
    logs = pd.read_csv(file_path, parse_dates=timestamp_cols)
    return logs


def run_model_inference(column_strategies, point_config, cont_config):
    """
    Complete inference pipeline:
    - Preprocess input data
    - Detect point and continuous anomalies
    - Embed anomalies
    - Compare with engineer logs
    - Save predictions and metadata
    """
    
    # --- Load Data ---
    sensor_df = load_sensor_data('data_simulator/SimulatedWellData.csv', 'Timestamp')
    engineer_logs_df = load_log_data('data_simulator/EngineerLogs.csv')

    print('Applying pre-processing')
    scaled_df, scaler = apply_preprocessing(sensor_df, column_strategies, threshold=3.5)

    print('Anomaly Detection')
    ops, scaled_df['PointAnomaly'] = run_point_anomaly_detectors(
        scaled_df[attrs_to_use],
        methods=point_config.get('methods'),
        threshold=point_config.get('z_thresh'),
        contamination=point_config.get('contamination') / 100,
        random_state=32
    )

    scaled_df['GradualAnomaly'], errors, threshold, lstm_ae_model = run_lstm_pipeline(
        scaled_df[attrs_to_use],
        window_size=cont_config.get('window_size') * 24,
        patience=cont_config.get('patience'),
        n_epochs=cont_config.get('epochs')
    )

    scaled_df['AnomalyDist'] = scaled_df.apply(
        lambda row: 'point' if row['PointAnomaly']
        else 'gradual' if row['GradualAnomaly']
        else None,
        axis=1
    )

    # --- Embed Detected Anomalies ---
    print('Get predicted anomaly embeddings')
    print(scaled_df.AnomalyDist.value_counts())

    scaled_df['AnomalyEmbedding'] = None
    for idx, row in scaled_df.iterrows():
        if row['AnomalyDist'] in ["point", "gradual"]:
            window = extract_window(idx, scaled_df[attrs_to_use], row['AnomalyDist'])
            if window is not None and len(window) > 0:
                embedding = get_lstm_embedding(window, lstm_ae_model)
                scaled_df.at[idx, 'AnomalyEmbedding'] = embedding.tolist()

    # --- Engineer Log Embeddings ---
    print('Get engineer logged anomaly embeddings')
    engineer_logs_df['DataEmbedding'] = None
    for idx, row in engineer_logs_df.iterrows():
        window_df = scaled_df[
            (scaled_df.index >= row['StartTimestamp']) & (scaled_df.index < row['EndTimestamp'])
        ][attrs_to_use]

        window = window_df.values.astype(np.float32)
        if window is not None and len(window) > 0:
            embedding = get_lstm_embedding(window, lstm_ae_model)
            engineer_logs_df.at[idx, 'DataEmbedding'] = embedding.tolist()

    # --- Similarity Matching ---
    print('Get top embeds between logged vs predicted anomalies')
    scaled_df = get_top_embeds(scaled_df, engineer_logs_df)

    # --- Save Results ---
    non_scaled_df = scaled_df[
        ['PointAnomaly', 'AnomalyDist', 'GradualAnomaly', 'AnomalyEmbedding',
         'SimilarMaintenanceNotes', 'SimilarObservations',
         'SimilarAnomalyTypes', 'SimilarityScores']
    ].join(sensor_df)

    non_scaled_df.to_csv('anomaly_detection/DetectedAnomalies.csv', index_label='Timestamp')
    print('Data saved back successfully')