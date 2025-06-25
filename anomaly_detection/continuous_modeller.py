import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for time-series reconstruction.

    Encodes input into a latent state and reconstructs it using a repeated hidden state.
    """
    def __init__(self, n_features, hidden_size=64):
        super().__init__()
        self.encoder = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=n_features, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h_repeated = h.repeat(x.size(1), 1, 1).transpose(0, 1)
        out, _ = self.decoder(h_repeated)
        return out


def create_sliding_windows(data, window_size):
    """
    Splits time-series data into overlapping sliding windows.

    Parameters:
    - data (ndarray): Time-series input.
    - window_size (int): Number of timesteps per window.

    Returns:
    - ndarray: Array of shape (samples, window_size, features)
    """
    X = []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
    return np.stack(X)


def train_lstm_autoencoder(model, dataloader, n_epochs=30, lr=1e-3, patience=3):
    """
    Trains LSTM autoencoder with early stopping.

    Parameters:
    - model (nn.Module): LSTM autoencoder.
    - dataloader (DataLoader): Training data.
    - n_epochs (int): Max number of training epochs.
    - lr (float): Learning rate.
    - patience (int): Early stopping patience.

    Returns:
    - model: Trained model.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    epoch_losses = []
    best_loss = float('inf')
    no_improve_count = 0

    for epoch in range(n_epochs):
        losses = []
        for batch in dataloader:
            x = batch[0]
            output = model(x)
            loss = criterion(output, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        epoch_loss = sum(losses) / len(losses)
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{n_epochs} - Loss: {epoch_loss:.6f}", end='\r')

        if epoch_loss < best_loss - 1e-6:
            best_loss = epoch_loss
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}: no loss improvement in {patience} epochs.")
                break

    return model


def compute_reconstruction_errors(model, dataloader):
    """
    Computes reconstruction error per sample.

    Parameters:
    - model: Trained LSTM autoencoder.
    - dataloader: Evaluation data.

    Returns:
    - errors (ndarray): Per-window MSE reconstruction error.
    """
    model.eval()
    errors = []

    with torch.no_grad():
        for x_batch in dataloader:
            x = x_batch[0]
            output = model(x)
            error = torch.mean((output - x) ** 2, dim=(1, 2))
            errors.extend(error.cpu().numpy())

    return np.array(errors)


def detect_anomalies(errors, threshold=None, quantile=0.99):
    """
    Flags anomalies based on reconstruction error threshold.

    Parameters:
    - errors (ndarray): Reconstruction errors.
    - threshold (float): Predefined threshold (optional).
    - quantile (float): Quantile to compute threshold if not given.

    Returns:
    - flags (ndarray): Boolean array of anomaly flags.
    - threshold (float): Applied anomaly threshold.
    """
    if threshold is None:
        threshold = np.quantile(errors, quantile)
    return errors > threshold, threshold


def run_lstm_pipeline(df, window_size=24, patience=3, n_epochs=30):
    """
    Runs full LSTM pipeline: training, scoring, and thresholding.

    Parameters:
    - df (pd.DataFrame): Input features.
    - window_size (int): Temporal context per sample.
    - patience (int): Early stopping patience.
    - n_epochs (int): Max epochs.

    Returns:
    - flags_series (pd.Series): Boolean anomaly flags aligned to input index.
    - error_series (pd.Series): MSE reconstruction errors.
    - threshold (float): Computed anomaly threshold.
    - model: Trained LSTM autoencoder.
    """
    data = df.values.astype(np.float32)
    X = create_sliding_windows(data, window_size)

    dataset = TensorDataset(torch.tensor(X))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = LSTMAutoencoder(n_features=data.shape[1])
    trained_model = train_lstm_autoencoder(model, dataloader, patience=patience, n_epochs=n_epochs)

    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    errors = compute_reconstruction_errors(trained_model, val_loader)
    flags, threshold = detect_anomalies(errors)

    # Align results back to original index
    anomaly_flags = np.zeros(len(df))
    anomaly_flags[window_size:] = flags.astype(int)

    error_series = pd.Series(np.nan, index=df.index)
    error_series.iloc[window_size:] = errors

    flags_series = pd.Series(anomaly_flags.astype(bool), index=df.index)

    return flags_series, error_series, threshold, model