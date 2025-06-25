import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity


def extract_window(ts, df, anomaly_type, hours_for_point=3, hours_for_cont=48):
    """
    Extracts a time window around an anomaly timestamp based on anomaly type.

    Parameters:
    - ts (pd.Timestamp): Center point of the anomaly.
    - df (pd.DataFrame): Original time-indexed data.
    - anomaly_type (str): Type of anomaly ('point' or 'gradual').
    - hours_for_point (int): Duration for point anomalies (centered around ts).
    - hours_for_cont (int): Duration for continuous anomalies.

    Returns:
    - np.ndarray: Extracted window as float32 array, or None if invalid.
    """
    if anomaly_type == "point":
        window_size = pd.Timedelta(hours=hours_for_point)
    elif anomaly_type == "gradual":
        window_size = pd.Timedelta(hours=hours_for_cont)
    else:
        return None
    
    start = ts - window_size / 2
    end = ts + window_size / 2
    window_df = df.loc[start:end]

    return window_df.values.astype(np.float32)


def get_lstm_embedding(seq, model):
    """
    Computes LSTM encoder embedding for a given time-series window.

    Parameters:
    - seq (np.ndarray): Input sequence of shape (T, F).
    - model (nn.Module): Trained LSTM Autoencoder model.

    Returns:
    - np.ndarray: Flattened hidden state (1D vector) from encoder.
    """
    with torch.no_grad():
        seq_tensor = torch.tensor(seq).unsqueeze(0)  # (1, T, F)
        _, (h, _) = model.encoder(seq_tensor)
    return h.squeeze().cpu().numpy()


def get_top_embeds(prod_df, logs_df, threshold=0.8, max_similarities=3):
    """
    For each predicted anomaly, finds most similar logged anomalies based on embeddings.

    Parameters:
    - prod_df (pd.DataFrame): Detected anomalies with 'AnomalyEmbedding'.
    - logs_df (pd.DataFrame): Engineer logs with 'DataEmbedding'.
    - threshold (float): Cosine similarity threshold to match events.
    - max_similarities (int): Max log entries to link per prediction.

    Returns:
    - pd.DataFrame: prod_df augmented with columns:
        ['SimilarMaintenanceNotes', 'SimilarObservations', 
         'SimilarAnomalyTypes', 'SimilarityScores']
    """
    downtime_embeddings = np.stack(logs_df["DataEmbedding"].values)
    anomaly_embeddings = np.stack(prod_df["AnomalyEmbedding"].dropna().values)
    anomaly_index = np.stack(prod_df['AnomalyEmbedding'].dropna().index)

    # Calculate cosine similarities
    similarities = cosine_similarity(anomaly_embeddings, downtime_embeddings)

    # Initialize empty match columns
    prod_df[['SimilarMaintenanceNotes', 'SimilarObservations',
             'SimilarAnomalyTypes', 'SimilarityScores']] = None, None, None, None

    for i in range(similarities.shape[0]):
        sim_row = similarities[i]
        above_thresh_idx = np.where(sim_row >= threshold)[0]

        if len(above_thresh_idx) > max_similarities:
            above_thresh_idx = above_thresh_idx[:max_similarities - 1]

        if len(above_thresh_idx) > 0:
            prod_df.at[anomaly_index[i], 'SimilarMaintenanceNotes'] = logs_df["MaintenanceNotes"].iloc[above_thresh_idx].tolist()
            prod_df.at[anomaly_index[i], 'SimilarObservations'] = logs_df["Observations"].iloc[above_thresh_idx].tolist()
            prod_df.at[anomaly_index[i], 'SimilarAnomalyTypes'] = logs_df["AnomalyType"].iloc[above_thresh_idx].tolist()
            prod_df.at[anomaly_index[i], 'SimilarityScores'] = sim_row[above_thresh_idx].round(4).tolist()

    return prod_df