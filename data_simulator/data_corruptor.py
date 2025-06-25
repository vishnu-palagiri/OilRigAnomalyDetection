import numpy as np
def corrupt_data_with_spikes(df, spike_config=None, missing_rate=0.01, random_seed=None):
    """
    Injects random outlier spikes/drops and missing values independently per column.

    Parameters:
        df: pd.DataFrame — input data
        spike_config: dict — e.g., {'FlowRate': {'count': 10, 'magnitude': 0.5}}
        missing_rate: float — fraction of values to drop (NaNs)
        random_seed: int — reproducibility
    Returns:
        pd.DataFrame
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    corrupted = df.copy()
    cols = [c for c in df.columns if c != 'Status']

    for col in cols:
        values = corrupted[col].copy()
        std_dev = df[col].std()
        spike_count = spike_config.get(col, {}).get('count', 5)
        spike_mag = spike_config.get(col, {}).get('magnitude', 3)  # multiple of std

        # Insert outliers
        spike_idx = np.random.choice(values.index, size=spike_count, replace=False)
        spike_sign = np.random.choice([-1, 1], size=spike_count)  # dip or surge
        spike_values = spike_sign * spike_mag * std_dev

        for idx, val in zip(spike_idx, spike_values):
            values.loc[idx] += val

        corrupted[col] = values

        # Insert missing values
        n_missing = int(len(df) * missing_rate)
        missing_idx = np.random.choice(df.index, size=n_missing, replace=False)
        corrupted.loc[missing_idx, col] = np.nan

    return corrupted