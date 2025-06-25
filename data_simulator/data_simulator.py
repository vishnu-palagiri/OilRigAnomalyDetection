import numpy as np
import random
import pandas as pd
from datetime import datetime, timedelta


def data_simulator(config):
    """
    Simulates synthetic sensor data for an oil and gas well over a specified time period.

    Parameters:
    - config (dict): Dictionary containing simulation parameters including:
        - 'periodOfData' (int): Duration of data in months.
        - 'eventCount' (int): Number of shut-in events to simulate.
        - 'initWHP' (float): Initial Well Head Pressure (psi).
        - 'initTP' (float): Initial Tubing Pressure (psi).
        - 'declineRate' (float): Pressure decline rate per hour.

    Returns:
    - df (pd.DataFrame): Time-indexed DataFrame containing the following columns:
        - 'Status': Well operational status (1 for online, 0 for shut-in).
        - 'WellHeadPressure': Simulated WHP with noise.
        - 'TubingPressure': Simulated TP with noise.
        - 'FlowRate': Simulated flow rate influenced by pressure and shut-in decay.
        - 'WellHeadTemperature': Simulated temperature derived from flow.
    """
    no_of_si_events = config.get('eventCount', 10)

    start_date = "2023-01-01"
    end_date = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=30 * config.get('periodOfData', 6))

    date_rng = pd.date_range(start=start_date, end=str(end_date), freq='H')
    n = len(date_rng)

    print(f"Generated Data from {start_date} to {end_date} for a period of {config.get('periodOfData', 6)} months")

    df = pd.DataFrame(index=date_rng)
    df['Status'] = 1

    # Inject shut-in (downtime) periods
    for _ in range(no_of_si_events):
        shut_start = random.randint(500, n - 200)
        event_len = random.randint(24, 240)
        df.iloc[shut_start:shut_start + event_len, df.columns.get_loc('Status')] = 0

    # Initialize pressure/flow arrays
    initial_whp = config.get('initWHP', 2500)
    initial_tp = config.get('initTP', 2000)
    decline_rate = config.get('declineRate', 0.012)
    shut_duration = 0

    whp, tp, flow, temp = [], [], [], []

    for i in range(n):
        t = i
        status = df['Status'].iloc[i]
        base_pressure = initial_whp - decline_rate * t
        base_tp = initial_tp - decline_rate * t * 0.8
        flow_val = max(0, 500 - (initial_whp - base_pressure) * 0.2)

        if status == 0:
            if i > 0 and df['Status'].iloc[i - 1] == 0:
                shut_duration += 1
            else:
                shut_duration = 0

            rise_factor = 1 - np.exp(-shut_duration / 12)
            extra_whp = 150 * rise_factor + np.random.normal(0, 3)
            extra_tp = 100 * rise_factor + np.random.normal(0, 2)
            base_pressure += extra_whp
            base_tp += extra_tp

            flow_decay = np.exp(-shut_duration / 6)
            flow_val = flow_val * flow_decay + np.random.normal(0, 2)
        else:
            shut_duration = 0

        whp.append(base_pressure + np.random.normal(0, 10))
        tp.append(base_tp + np.random.normal(0, 5))
        flow.append(flow_val + np.random.normal(0, 3))
        temp.append(60 + (0.2 * flow_val) + np.random.normal(0, 0.5))

    df['WellHeadPressure'] = whp
    df['TubingPressure'] = tp
    df['FlowRate'] = flow
    df['WellHeadTemperature'] = temp

    # Add missing values (~1% per column)
    for col in df.columns:
        missing_idx = np.random.choice(df.index, size=int(0.01 * n), replace=False)
        df.loc[missing_idx, col] = np.nan

    return df