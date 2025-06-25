import numpy as np
import random
import pandas as pd
from data_simulator.generate_logs import *


def inject_tubing_blockage(df, start_idx, duration=48):
    """
    Simulates a tubing blockage by gradually increasing tubing pressure
    and decreasing flow rate over the anomaly duration.
    """
    df = df.copy()
    for i in range(duration):
        factor = 1 + (i / duration) * 0.25
        idx = start_idx + i
        if idx < len(df):
            df.iloc[idx, df.columns.get_loc('TubingPressure')] *= factor
            df.iloc[idx, df.columns.get_loc('FlowRate')] *= (1 - (i / (duration * 1.5)))
            df.iloc[idx, df.columns.get_loc('AnomalyType')] = 'TubingBlockage'
    return df


def inject_choke_erosion(df, start_idx, duration=72):
    """
    Simulates choke erosion by introducing gradual flow rate drift with noise.
    """
    df = df.copy()
    for i in range(duration):
        idx = start_idx + i
        if idx < len(df):
            noise = np.random.normal(0, 0.05 * df['FlowRate'].std())
            drift = 0.005 * i
            df.iloc[idx, df.columns.get_loc('FlowRate')] += noise + drift
            df.iloc[idx, df.columns.get_loc('AnomalyType')] = 'ChokeErosion'
    return df


def inject_liquid_loading(df, start_idx, duration=72):
    """
    Simulates liquid loading by increasing tubing pressure and applying
    exponential decay to flow rate.
    """
    df = df.copy()
    for i in range(duration):
        idx = start_idx + i
        if idx < len(df):
            df.iloc[idx, df.columns.get_loc('TubingPressure')] += i * 0.5
            df.iloc[idx, df.columns.get_loc('FlowRate')] *= (0.995 ** i)
            df.iloc[idx, df.columns.get_loc('AnomalyType')] = 'LiquidLoading'
    return df


def inject_tubing_collapse(df, start_idx, duration=12):
    """
    Simulates tubing collapse by setting flow rate to 0 and spiking tubing pressure.
    """
    df = df.copy()
    for i in range(duration):
        idx = start_idx + i
        if idx < len(df):
            df.iloc[idx, df.columns.get_loc('FlowRate')] = 0
            df.iloc[idx, df.columns.get_loc('TubingPressure')] += np.random.normal(50, 20)
            df.iloc[idx, df.columns.get_loc('AnomalyType')] = 'TubingCollapse'
    return df


def inject_overheating(df, start_idx, magnitude=20, duration=8):
    """
    Simulates overheating by linearly increasing temperature over time.
    """
    df = df.copy()
    rise = np.linspace(0, magnitude, duration)
    for i in range(duration):
        idx = start_idx + i
        if idx < len(df):
            df.iloc[idx, df.columns.get_loc('WellHeadTemperature')] += rise[i]
            df.iloc[idx, df.columns.get_loc('AnomalyType')] = 'OverHeating'
    return df


def inject_sand_production(df, start_idx, duration=24):
    """
    Simulates sand production via random spikes in flow rate and pressure.
    """
    df = df.copy()
    for i in range(duration):
        idx = start_idx + i
        if idx < len(df):
            df.iloc[idx, df.columns.get_loc('WellHeadPressure')] += np.random.normal(0, 30)
            df.iloc[idx, df.columns.get_loc('FlowRate')] += np.random.normal(0, 20)
            df.iloc[idx, df.columns.get_loc('AnomalyType')] = 'SandProduction'
    return df


def inject_anomalies_randomly(df, config, random_seed=None):
    """
    Randomly injects multiple types of anomalies into the time-series data,
    respecting downtime and overlap constraints.

    Parameters:
        df : pd.DataFrame
            The original well sensor data.
        config : dict
            Anomaly type → dict of injection parameters (duration, counts).
        random_seed : int or None
            Optional seed for reproducibility.

    Returns:
        anom_df : pd.DataFrame
            Modified data containing injected anomalies.
        logs_df : pd.DataFrame
            Metadata logs for logged anomaly injections.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    anom_df = df.copy()
    anom_df['AnomalyType'] = np.nan
    used_indices = set()
    logs_df = pd.DataFrame(columns=['StartTimestamp', 'EndTimestamp', 'AnomalyType', 'MaintenanceNotes', 'Observations'])

    anomaly_funcs = {
        'TubingBlockage': inject_tubing_blockage,
        'ChokeErosion': inject_choke_erosion,
        'LiquidLoading': inject_liquid_loading,
        'TubingCollapse': inject_tubing_collapse,
        'OverHeating': inject_overheating,
        'SandProduction': inject_sand_production
    }

    for anomaly_type, params in config.items():
        min_dur = params.get('min_duration', 12)
        max_dur = params.get('max_duration', 48)

        def generateAnomData(anom_df, logs_df, generate_logs=True):
            for _attempt in range(100):
                dur = random.randint(min_dur, max_dur)
                idx = random.randint(0, len(anom_df) - dur - 1)

                if anom_df['Status'].iloc[idx:idx + dur].sum() == dur:
                    if all(i not in used_indices for i in range(idx, idx + dur)):

                        if generate_logs:
                            logs_df = generate_operator_logs(
                                logs_df, anomaly_type,
                                anom_df.index[idx],
                                anom_df.index[idx + dur], 
                                random_seed=random_seed
                            )

                        used_indices.update(range(idx, idx + dur))
                        inject_func = anomaly_funcs[anomaly_type]

                        if 'Collapse' in anomaly_type or 'OverHeating' in anomaly_type:
                            anom_df = inject_func(anom_df, idx, duration=dur)
                        else:
                            anom_df = inject_func(anom_df, start_idx=idx, duration=dur)
                        break
            return anom_df, logs_df

        # Inject logged and unlogged anomalies
        for _ in range(params.get('logged_count', 1)):
            anom_df, logs_df = generateAnomData(anom_df, logs_df, generate_logs=True)

        for _ in range(params.get('unlogged_count', 1)):
            anom_df, logs_df = generateAnomData(anom_df, logs_df, generate_logs=False)

    return anom_df, logs_df