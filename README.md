# 🛢️ Oil and Gas Well Anomaly Detection System

This project offers a full-stack simulation and anomaly detection suite for oil and gas wells. It integrates **synthetic data generation**, **ML-based anomaly detection**, **embedding-based log matching**, and **LLM-powered summarization**—all accessible through a clean Streamlit UI.

---

## 🚀 Features

- 🧪 **Synthetic Data Generator**  
  Simulates realistic time-series data (flow rate, WHP, WHT, tubing pressure) with shut-in logic and operational decline curves.

- 🧨 **Noise & Fault Injector**  
  Adds missing values, random noise spikes, and six types of engineered anomalies (e.g., Tubing Blockage, Sand Production).

- 🧠 **Anomaly Detection**  
  - *Point Anomalies*: Z-Score and Isolation Forest
  - *Continuous Anomalies*: LSTM Autoencoder

- 🔁 **Embedding & Similarity Matching**  
  Matches detected anomalies against embedded engineer maintenance logs using **cosine similarity**.

- 📜 **LLM-Based Anomaly Summarization**  
  Text summaries of anomalous episodes using Google's `flan-t5-small` model with dynamically constructed prompts.

- 📊 **Visual Analytics Dashboard (Streamlit)**  
  Multi-tabbed layout for simulation, visualization, inference, and summary reporting.

---

## 📁 Folder Structure

```text
|- app.py                         # Streamlit UI logic & workflow
|- utils.py                      # Visualization helpers (Plotly plots & anomaly overlays)
|- data_simulator/
|  |- __init__.py                # Orchestrates the simulation → corruption → logging pipeline
|  |- data_simulator.py          # Simulates well behavior (downtime, pressure decay, etc.)
|  |- data_corruptor.py          # Adds missing values and synthetic spike noise
|  |- anomaly_injector.py        # Injects 6 anomaly types and logs based on templates
|  |- generate_logs.py           # Engineer-like logs (natural language templates + style randomization)
|- anomaly_detection/
|  |- __init__.py                # Inference controller (preprocessing → modeling → embedding)
|  |- data_preparation.py        # Per-feature imputing, denoising, scaling
|  |- point_modeller.py          # Z-score and Isolation Forest logic
|  |- continuous_modeller.py     # LSTM Autoencoder model and trainer
|  |- embed.py                   # Embedding extractor & cosine similarity matcher
|- summarizer/
|  |- __init__.py                # LLM-based summarizer using Google's flan-t5-small
|- requirements.txt              # Python dependencies
```

---

## 🧰 Setup Instructions

```bash
# Clone the repository
git clone https://github.com/vishnu-palagiri/OilRigAnomalyDetection

# Install required packages
pip install -r requirements.txt
```

Optional: For optimal performance, configure GPU usage or device_map="auto" for large models in the summarizer.

---

## 🧪 Running the App

```bash
streamlit run app.py
```
## 🧪 Application Tabs

### 📊 Synthetic Data Generation  
Used to generate synthetic data with a wide range of configuration parameters.

<details>
<summary>⚙️ <strong>Base Parameters</strong></summary>

| Parameter                   | Description                                         |
|----------------------------|-----------------------------------------------------|
| Period of Data             | Number of months of data to be generated            |
| Shut In Periods            | Total number of shut-in periods for the well        |
| Initial Well Head Pressure | The initial pressure at the well head               |
| Initial Tubing Pressure    | The initial pressure at the tubing                  |
| Pressure Decline Rate      | Rate at which pressure declines at the well head    |

</details>

<details>
<summary>🧪 <strong>Data Quality Config Parameters</strong></summary>

| Parameter | Description                                                                 |
|-----------|-----------------------------------------------------------------------------|
| Count     | Number of erroneous behaviors in the selected stream                       |
| Magnitude | Magnitude of erroneous behaviors (in standard deviations)                  |

</details>

<details>
<summary>🚨 <strong>Anomaly Injection Config Parameters</strong></summary>

| Parameter                  | Description                                                         |
|---------------------------|---------------------------------------------------------------------|
| Count of Logged Events    | Total number of anomaly events with associated engineer logs        |
| Count of Non Logged Events| Total number of anomaly events without associated engineer logs     |
| Min Duration              | Minimum duration of an anomaly event                                |
| Max Duration              | Maximum duration of an anomaly event                                |

</details>

<p align="center">
  <img src="img/syntheticdatageneration.png" alt="Synthetic Data Generation Tab" width="80%"/>
</p>


### 📈 Data Visualization

Visualize the **synthetic time series data** to observe trends, identify signal behaviors, and analyze induced anomalies across the simulation period.

Each anomaly type comes with its own toggle, allowing fine-grained control over the visualization:

- ✅ **Anomaly Toggles**: Enable or disable background highlights (rectangular overlays) for each anomaly type individually.
- 🕒 **Time Axis Exploration**: Inspect how different sensor signals change over the data generation window.
- 🎯 **Anomaly Context**: Highlighted periods correspond to time spans where anomalies were synthetically injected, aiding correlation and interpretability.

<p align="center">
  <img src="img/datavisualizer.png" alt="Data Visualization Tab" width="80%"/>
</p>

### 🧠 Model Inference
Model inference configuration allows you to fine-tune preprocessing steps and anomaly detection behavior for both point-wise and continuous anomaly detection models.

<details>
<summary>🧹 <strong>Column Pre-processing Strategies</strong></summary>

| Parameter | Description                                                                 |
|-----------|-----------------------------------------------------------------------------|
| Impute    | Imputation method – options include `ffill`, `interpolate`, or `none`      |
| Denoise   | Whether to apply noise filtering to the selected attribute                 |
| Scale     | Whether to scale the selected attribute before inference                   |

</details>

<details>
<summary>📍 <strong>Point Anomaly Detection</strong></summary>

| Parameter                | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| zscore                  | Enable/disable z-score anomaly detection                                     |
| isoforest               | Enable/disable Isolation Forest algorithm                                    |
| zscore threshold        | Z-score threshold for anomaly flagging (in standard deviations)              |
| IsoFor Contamination %  | Contamination parameter for Isolation Forest (i.e., expected anomaly fraction) |

</details>

<details>
<summary>📈 <strong>Continuous Anomaly Detection</strong></summary>

| Parameter                 | Description                                                              |
|--------------------------|--------------------------------------------------------------------------|
| window size (days)       | Size of the rolling window for LSTM AutoEncoder                          |
| epochs                   | Number of training epochs to run                                         |
| early stopping patience  | Number of epochs to wait without improvement before stopping training    |

</details>

<p align="center">
  <img src="img/modelinference.png" alt="Model Inference Tab" width="80%"/>
</p>

### 📋 Summarize Predictions 

Explore model predictions overlaid on sensor data in a time series view.

- 🖍️ **Lasso Selection**: Use lasso select to highlight regions of interest within the time series.
- 📄 **Dynamic Data Table**: Once a region is selected, a table populates with the corresponding anomalies detected in that window.
- 🧠 **Summarization Trigger**: Click **Summarize** to initiate a summary generation using the `google/flan-t5-small` language model.
- 🔎 **Semantic Log Extraction**: Relevant logs are semantically ranked based on anomaly context and summarized accordingly.
- 📌 **Summary Output**: Generated summaries are displayed at the bottom of the tab for easy review and analysis.

<p align="center">
  <img src="img/summarizer.png" alt="Model Summarizer Tab" width="80%"/>
  <img src="img/summarizer-1.png" alt="Model Summarizer Tab 1" width="80%"/>
</p>

---

## 🔍 Anomaly Types

| Type             | Description                               | Category     |
|------------------|-------------------------------------------|--------------|
| TubingBlockage   | Restriction causing flow drop & TP rise   | Gradual      |
| ChokeErosion     | Surging flow rate due to orifice wear     | Gradual      |
| LiquidLoading    | Hydrostatic loading, flow collapse        | Gradual      |
| TubingCollapse   | Sudden shutdown, TP spike                 | Point        |
| OverHeating      | Localized temp spikes near ESP            | Point        |
| SandProduction   | Abrupt pressure/flow noise from solids    | Point        |

All are injected with optional operator maintenance notes and observations generated from domain-aware templates.

---

## 🧠 LLM Summarization Example

Uses `google/flan-t5-small` to generate concise insights like:

> “Flow rate dropped gradually with rising tubing pressure. Signs suggest a tubing blockage potentially due to wax deposition.”

Prompt templates include anomaly types, observation phrases, and maintenance notes matched via cosine similarity.

---

## 📋 License

MIT License © 2025 Vishnu Palagiri

---
