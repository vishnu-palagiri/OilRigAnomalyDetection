import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots

def plot_with_anomalies(selected_anomalies, anomaly_colors):
    """
    Plot well data with highlighted anomaly periods using vertical rectangles.

    Parameters:
    - selected_anomalies: list of anomaly types to visualize.
    - anomaly_colors: dict mapping anomaly type to color code.

    Returns:
    - fig: Plotly Figure object.
    """
    df = pd.read_csv('data_simulator/SimulatedWellData.csv', index_col='Timestamp')
    fig = go.Figure()

    # --- Primary Y-Axis ---
    fig.add_trace(go.Scatter(x=df.index, y=df['FlowRate'],
                             mode='lines', name='Flow Rate',
                             line=dict(color='rgba(200,50,50,0.6)', width=2),
                             yaxis='y1', connectgaps=True))

    fig.add_trace(go.Scatter(x=df.index, y=df['WellHeadTemperature'],
                             mode='lines', name='WellHead Temp',
                             line=dict(color='rgba(255,215,0,0.9)', width=2),
                             yaxis='y1', connectgaps=True))

    # --- Secondary Y-Axis ---
    fig.add_trace(go.Scatter(x=df.index, y=df['WellHeadPressure'],
                             mode='lines', name='WellHead Pressure',
                             line=dict(color='rgba(0,100,255,0.7)', width=2),
                             yaxis='y2', connectgaps=True))

    fig.add_trace(go.Scatter(x=df.index, y=df['TubingPressure'],
                             mode='lines', name='Tubing Pressure',
                             line=dict(color='rgba(0,80,200,0.7)', width=2),
                             yaxis='y2', connectgaps=True))
    
    
    # --- Highlight Anomaly Periods ---
    for anomaly in selected_anomalies:
        mask = df['AnomalyType'] == anomaly
        if mask.any():
            anomaly_ranges = []
            is_anomaly = False

            for i in range(len(df)):
                if not is_anomaly and mask.iloc[i]:
                    is_anomaly = True
                    start = df.index[i]
                elif is_anomaly and not mask.iloc[i]:
                    is_anomaly = False
                    end = df.index[i]
                    anomaly_ranges.append((start, end))
            if is_anomaly:
                anomaly_ranges.append((start, df.index[-1]))

            for start, end in anomaly_ranges:
                fig.add_vrect(
                    x0=start, x1=end,
                    fillcolor=anomaly_colors.get(anomaly, 'lightgrey'),
                    opacity=0.3, layer='below', line_width=0,
                )


    # --- Layout ---
    fig.update_layout(
        title='Well Data & Anomalies (vs) Time',
        xaxis=dict(title='Time', showgrid=False),
        yaxis=dict(title='Flow Rate / Temperature', side='left', showgrid=False),
        yaxis2=dict(title='Pressure (psi)', overlaying='y', side='right', showgrid=False),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        template='plotly_white'
    )

    return fig

def get_model_summarizing_plot():
    """
    Generate interactive plot of detected anomalies and return associated DataFrame.

    Returns:
    - fig: Plotly Figure object.
    - df: DataFrame containing detected anomalies data.
    """
    fig = go.Figure()

    df = pd.read_csv('anomaly_detection/DetectedAnomalies.csv', parse_dates=['Timestamp'])
    df = df.set_index('Timestamp')

    # --- Primary Y-Axis ---
    fig.add_trace(go.Scatter(x=df.index, y=df['FlowRate'],
                             mode='lines', name='Flow Rate',
                             line=dict(color='rgba(200,50,50,0.6)', width=2),
                             yaxis='y1', connectgaps=True))

    fig.add_trace(go.Scatter(x=df.index, y=df['WellHeadTemperature'],
                             mode='lines', name='WellHead Temp',
                             line=dict(color='rgba(255,215,0,0.9)', width=2),
                             yaxis='y1', connectgaps=True))

    # --- Secondary Y-Axis ---
    fig.add_trace(go.Scatter(x=df.index, y=df['WellHeadPressure'],
                             mode='lines', name='WellHead Pressure',
                             line=dict(color='rgba(0,100,255,0.7)', width=2),
                             yaxis='y2', connectgaps=True))

    fig.add_trace(go.Scatter(x=df.index, y=df['TubingPressure'],
                             mode='lines', name='Tubing Pressure',
                             line=dict(color='rgba(0,80,200,0.7)', width=2),
                             yaxis='y2', connectgaps=True))

    # Point anomalies → "x" marker
    point_anom = df[df['PointAnomaly'] == 1]
    
    fig.add_trace(go.Scatter(
        x=point_anom.index, y=point_anom['FlowRate'],
        mode='markers', name='Point Anomaly',
        marker=dict(symbol='triangle-up', color='red', size=10)
    ))

    # Continuous anomalies → triangle marker
    cont_anom = df[df['GradualAnomaly'] == 1]
    fig.add_trace(go.Scatter(
        x=cont_anom.index, y=cont_anom['FlowRate'],
        mode='markers', name='Cont. Anomaly',
        marker=dict(symbol='x', color='orange', size=10)
    ))

    fig.update_layout(
        dragmode='lasso',
        height=400,
        margin=dict(t=20, b=20),
        xaxis = dict(showgrid=False),
        yaxis=dict(title='Flow Rate / Temperature', side='left', showgrid=False),
        yaxis2=dict(title='Pressure (psi)', overlaying='y', side='right', showgrid=False),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        template='plotly_white',
        showlegend=True
    )

    return fig, df


def plot_well_data_dual_axis(df, title="Oil Well Sensor Simulation (Dual Axis)"):
    """
    Visualize well sensor data using dual y-axes and highlight shut-in periods.

    Parameters:
    - df: DataFrame of well data (indexed by timestamp).
    - title: Title of the plot.
    """
    fig = go.Figure()

    # --- Primary Y-Axis ---
    fig.add_trace(go.Scatter(x=df.index, y=df['FlowRate'],
                             mode='lines', name='Flow Rate',
                             line=dict(color='rgba(200,50,50,0.6)', width=2),
                             yaxis='y1', connectgaps=True))

    fig.add_trace(go.Scatter(x=df.index, y=df['WellHeadTemperature'],
                             mode='lines', name='WellHead Temp',
                             line=dict(color='rgba(255,215,0,0.9)', width=2),
                             yaxis='y1', connectgaps=True))

    # --- Secondary Y-Axis ---
    fig.add_trace(go.Scatter(x=df.index, y=df['WellHeadPressure'],
                             mode='lines', name='WellHead Pressure',
                             line=dict(color='rgba(0,100,255,0.7)', width=2),
                             yaxis='y2', connectgaps=True))

    fig.add_trace(go.Scatter(x=df.index, y=df['TubingPressure'],
                             mode='lines', name='Tubing Pressure',
                             line=dict(color='rgba(0,80,200,0.7)', width=2),
                             yaxis='y2', connectgaps=True))

    # --- Highlight Shut-In Periods ---
    shut_ranges = []
    status = df['Status'].fillna(1)

    is_shut = False
    for i in range(len(status)):
        if not is_shut and status.iloc[i] == 0:
            is_shut = True
            start = df.index[i]
        elif is_shut and status.iloc[i] == 1:
            is_shut = False
            end = df.index[i]
            shut_ranges.append((start, end))
    if is_shut:
        shut_ranges.append((start, df.index[-1]))

    for start, end in shut_ranges:
        fig.add_vrect(x0=start, x1=end,
                      fillcolor='lightgrey', opacity=0.3,
                      layer='below', line_width=0)

    # --- Layout ---
    fig.update_layout(
        title=title,
        xaxis=dict(title='Time'),
        yaxis=dict(title='Flow Rate / Temperature', side='left'),
        yaxis2=dict(title='Pressure (psi)', overlaying='y', side='right'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        template='plotly_white'
    )

    fig.show()

def plot_anomalies_multifeature(df, point_anomalies, gradual_anomalies, title="Anomaly Detection"):
    """
    Create multi-subplot chart of well features with highlighted point and gradual anomalies.

    Parameters:
    - df: DataFrame with well features.
    - point_anomalies: Boolean index or mask for point anomalies.
    - gradual_anomalies: Boolean index or mask for gradual anomalies.
    - title: Title of the plot.
    """
    features = ['FlowRate', 'WellHeadTemperature', 'WellHeadPressure', 'TubingPressure']
    fig = make_subplots(rows=len(features), cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02)

    for i, col in enumerate(features, start=1):
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[col],
            mode='lines',
            name=f"{col}",
            line=dict(width=2),
            showlegend=True
        ), row=i, col=1)

        # Gradual Anomalies — red X
        fig.add_trace(go.Scatter(
            x=df.index[gradual_anomalies],
            y=df[col][gradual_anomalies],
            mode='markers',
            marker=dict(symbol='x', color='red', size=8),
            name='Gradual Anomaly',
            showlegend=(i == 1)
        ), row=i, col=1)

        # Point Anomalies — red triangle
        fig.add_trace(go.Scatter(
            x=df.index[point_anomalies],
            y=df[col][point_anomalies],
            mode='markers',
            marker=dict(symbol='triangle-up', color='firebrick', size=9),
            name='Point Anomaly',
            showlegend=(i == 1)
        ), row=i, col=1)
        
    fig.update_layout(
        height=140 * len(features),
        title_text=title,
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    fig.update_xaxes(title_text="Time", row=len(features), col=1)
    fig.show()
