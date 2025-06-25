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
  Text summaries of anomalous episodes using Microsoft's `phi-1_5` model with dynamically constructed prompts.

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
|  |- __init__.py                # LLM-based summarizer using Microsoft’s Phi-1_5
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

Tabs available:
- **📊 Synthetic Data Generation**
- **📈 Data Visualization**
- **🧠 Model Inference**
- **📋 Summarize Predictions**

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

Uses `microsoft/phi-1_5` to generate concise insights like:

> “Flow rate dropped gradually with rising tubing pressure. Signs suggest a tubing blockage potentially due to wax deposition.”

Prompt templates include anomaly types, observation phrases, and maintenance notes matched via cosine similarity.

---

## 📋 License

MIT License © 2025 Vishnu Palagiri

---
