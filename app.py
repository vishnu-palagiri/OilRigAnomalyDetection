import streamlit as st
from data_simulator import *
from anomaly_detection import *
from summarizer import *
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events


# Default config values
default_spike_config = {
    'FlowRate': {'count': 8, 'magnitude': 4},
    'WellHeadPressure': {'count': 5, 'magnitude': 2},
    'TubingPressure': {'count': 5, 'magnitude': 2},
    'WellHeadTemperature': {'count': 6, 'magnitude': 3}
}

default_config = {
    'periodOfData': 6,
    'eventCount': 10,
    'initWHP': 2500,
    'initTP': 2000,
    'declineRate': 0.012
}

default_anomaly_config = {
    'TubingBlockage': {'logged_count': 4, 'min_duration': 24, 'max_duration': 72, 'unlogged_count': 5},
    'ChokeErosion': {'logged_count': 3, 'min_duration': 48, 'max_duration': 96, 'unlogged_count': 5},
    'LiquidLoading': {'logged_count': 3, 'min_duration': 36, 'max_duration': 60, 'unlogged_count': 5},
    'TubingCollapse': {'logged_count': 5, 'min_duration': 6, 'max_duration': 12, 'unlogged_count': 5},
    'OverHeating': {'logged_count': 3, 'min_duration': 4, 'max_duration': 10, 'unlogged_count': 5},
    'SandProduction': {'logged_count': 3, 'min_duration': 12, 'max_duration': 24, 'unlogged_count': 5}
}

column_strategies = {
    'WellHeadPressure': {'impute': 'ffill', 'denoise': True, 'scale': True},
    'FlowRate':         {'impute': 'interpolate', 'denoise': True, 'scale': True},
    'TubingPressure':   {'impute': 'ffill', 'denoise': True, 'scale': True},
    'WellHeadTemperature': {'impute': 'interpolate', 'denoise': True, 'scale': True},
    'Status':           {'impute': 'ffill', 'denoise': False, 'scale': False},
    'AnomalyType':      {'impute': None, 'denoise': False, 'scale': False}
}

anomaly_colors = {
    'TubingBlockage': '#FF6384',      
    'ChokeErosion': '#E6B800',        
    'LiquidLoading': '#36A2EB',       
    'TubingCollapse': '#4BC0C0',      
    'OverHeating': '#FF9F40',         
    'SandProduction': '#9966FF'
}

st.set_page_config(page_title="Rig Anomaly Detection", layout="wide")
st.subheader("Anomaly Detection in Oil Rig Operations")

# Tabs as "Navbar"
summary_tab, model_inference, visualization_tab, data_generation = st.tabs(["📋 Summarize Predictions", "🧠 Model Inference", "📈 Data Visualization", "📊 Synthetic Data Generation"])

# === Tab 1: Data Generation ===
with data_generation:

    st.subheader("Base Parameters")
    col0, col1, col2, col3, col4 = st.columns(5)
    with col0:
        periodOfData = st.number_input("Period of Data (months)", value=default_config['periodOfData'], step=1)
    with col1:
        event_count = st.number_input("Shut In Periods (Non Downtime)", value=default_config['eventCount'], step=5)
    with col2:
        init_whp = st.number_input("Initial Well Head Pressure", value=default_config['initWHP'], step=30)
    with col3:
        init_tp = st.number_input("Initial Tubing Pressure", value=default_config['initTP'], step=30)
    with col4:
        decline_rate = st.number_input("Pressure Decline Rate", value=default_config['declineRate'], step=0.001, format="%.3f")

    config = {
        'periodOfData': periodOfData,
        'eventCount': event_count,
        'initWHP': init_whp,
        'initTP': init_tp,
        'declineRate': decline_rate
    }

    st.subheader("Data Quality Noise / Spike Configuration")

    sensor_keys = list(default_spike_config.keys())
    cols = st.columns(4)

    spike_config = {}
    for i, key in enumerate(sensor_keys):
        with cols[i % 4].expander(f"{key}", expanded=True):
            
            col1, col2 = st.columns(2)
            with col1:
                count = st.number_input(
                    f"Count", value=default_spike_config[key]['count'] - 1, key=f"{key}_count"
                )
            with col2:
                mag = st.number_input(
                    f"Magnitude", value=default_spike_config[key]['magnitude'] - 1, key=f"{key}_magnitude"
                )
            spike_config[key] = {'count': count, 'magnitude': mag}


    st.subheader("Unlabelled Anomaly Injection Configuration")

    anomaly_keys = list(default_anomaly_config.keys())
    anomaly_cols = st.columns(3)

    anomaly_config = {}
    for i, key in enumerate(anomaly_keys):
        with anomaly_cols[i % 3].expander(f"{key}", expanded=False):
            col0, col1 = st.columns(2)
            with col0:
                logged_count = st.number_input(f"Count of Logged events", value=default_anomaly_config[key]['logged_count'], key=f"{key}_logged_count")
            with col1:
                unlogged_count = st.number_input(f"Count of Non Logged events", value=default_anomaly_config[key]['unlogged_count'], key=f"{key}_unlogged_count")
            with col0:
                min_dur = st.number_input(f"Min Duration (hr)", value=default_anomaly_config[key]['min_duration'], key=f"{key}_min")
            with col1:
                max_dur = st.number_input(f"Max Duration (hr)", value=default_anomaly_config[key]['max_duration'], key=f"{key}_max")
            anomaly_config[key] = {
                'logged_count': logged_count,
                'unlogged_count': unlogged_count,
                'min_duration': min_dur,
                'max_duration': max_dur
            }

    if st.button("Run Data Generation"):
        run_data_generation(spike_config, config, anomaly_config)
        st.success("✅ Data generation completed successfully!")

with visualization_tab:
    st.subheader("📈 Visualize Sensor Data with Anomalies")
    
    st.markdown("### Toggle Anomalies")
    selected_anomalies = []
    cols = st.columns(len(anomaly_colors))

    for i, (anomaly, color) in enumerate(anomaly_colors.items()):
        with cols[i]:
            row = st.columns([0.1, 0.9])  # Adjust width ratio if needed
            with row[0]:
                selected = st.checkbox(" ", value=True, key=f"chk_{anomaly}")
            with row[1]:
                st.markdown(f"<div style='margin-top: 8px; color: {color}; font-weight: 600'>{anomaly}</div>", unsafe_allow_html=True)
                if selected:
                    selected_anomalies.append(anomaly)


    # Call the visualizer
    fig = plot_with_anomalies(selected_anomalies, anomaly_colors)
    st.plotly_chart(fig, use_container_width=True)

with model_inference:
    st.markdown("### Model Inference Configuration")
    # === Column Strategies ===
    st.markdown("#### Column Pre Processing Strategies")

    strategy_cols = list(column_strategies.keys())
    rows = [strategy_cols[i:i+3] for i in range(0, len(strategy_cols), 3)]

    for row in rows:
        cols = st.columns(len(row))
        for i, col_name in enumerate(row):
            with cols[i].expander(col_name, expanded=False):
                strategy = column_strategies[col_name]
                impute_col, denoise_col, scale_col = st.columns([2, 1, 1])

                with impute_col:
                    impute_method = st.selectbox("Impute", options=[None, 'ffill', 'interpolate'],
                                                index=[None, 'ffill', 'interpolate'].index(strategy['impute']),
                                                key=f"{col_name}_impute")

                with denoise_col:
                    denoise = st.checkbox("Denoise", value=strategy['denoise'], key=f"{col_name}_denoise")

                with scale_col:
                    scale = st.checkbox("Scale", value=strategy['scale'], key=f"{col_name}_scale")

                column_strategies[col_name] = {
                    'impute': impute_method,
                    'denoise': denoise,
                    'scale': scale
                }

    # === Anomaly Configuration ===
    st.markdown("#### Anomaly Detection Parameters")
    # Create 3 columns: Point Anomalies | Divider | Continuous Anomalies
    col1, divider, col2 = st.columns([0.48, 0.04, 0.48])

    with col1:
        st.markdown("##### Point Anomalies")
        methods = ['zscore', 'isoforest']
        selected_methods = []

        # 20% + 20% + 30% + 30% = 100%
        c1, c2, c3, c4 = st.columns([0.15, 0.25, 0.3, 0.3])

        with c1:
            if st.checkbox("zscore", value=True, key="point_zscore"):
                selected_methods.append("zscore")

        with c2:
            if st.checkbox("isoforest", value=True, key="point_isolation"):
                selected_methods.append("isoforest")

        with c3:
            z_thresh = st.number_input("Z-Score Threshold", value=3.0, step=0.1)

        with c4:
            contamination = st.number_input("Iso-For Contamination (%)", value=2, step=5)
        
        point_config = {
            'methods': selected_methods,
            'z_thresh': z_thresh,
            'contamination': contamination
        }

    with divider:
        st.markdown(
            """
            <div style="height: 100%; width: 100%; display: flex; align-items: stretch;">
                <div style="border-left: 2px solid #CCC; height: 150px; margin: auto;"></div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown("##### Continuous Anomalies")
        c1, c2, c3 = st.columns(3)
        with c1:
            window_size = st.number_input("Window Size (days)", value=2)
        with c2:
            epochs = st.number_input("Epochs", value=30, step=2)
        with c3:
            patience = st.number_input("Early Stopping Patience", value=3)

    cont_config = {
        'window_size': window_size,
        'epochs': epochs,
        'patience': patience
    }

    status = st.empty()
    if st.button("Run Inference"):
        run_model_inference(column_strategies, point_config, cont_config)
        status.info("🚀 Running inference with selected configurations...")
        status.success("✅ Model inference complete!")

with summary_tab:
    st.header("📋 Summarize Predictions")

    # Toggle for vrect trace visibility
    show_vrect = st.checkbox("Show Anomaly Labels", value=False)

    fig, df = get_model_summarizing_plot(anomaly_colors)

    # Update vrect trace visibility
    if 'shapes' in fig.layout:
        for shape in fig.layout.shapes:
            if shape.type == 'rect':
                shape.visible = show_vrect


    selected = plotly_events(fig, select_event=True)

    # Table view
    st.markdown("### Selected Data Points")
    filtered_df = df.loc[[pt['x'] for pt in selected]] if selected else df.iloc[0:0]
    st.dataframe(filtered_df[['FlowRate', 'WellHeadPressure', 'TubingPressure', 'WellHeadTemperature']])

    # Summarize button
    summarize_button = st.button("Summarize", disabled=len(filtered_df) == 0)

    if summarize_button:
        with st.spinner("Generating summary..."):
            html_output = summarize_predictions(filtered_df)
            st.markdown("### 📝 Summary", unsafe_allow_html=True)
            st.markdown(html_output, unsafe_allow_html=True)
