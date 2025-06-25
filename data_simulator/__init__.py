from utils import *
from data_simulator.data_simulator import *
from data_simulator.data_corruptor import *
from data_simulator.anomaly_injector import *
from data_simulator.generate_logs import *

from sentence_transformers import SentenceTransformer
import pandas as pd

# Set reproducible seed
random_seed = 32

def run_data_generation(spike_config, config, anomaly_config):
    """
    Orchestrates the full data simulation pipeline:
    1. Generate synthetic well data
    2. Apply spike corruption and missingness
    3. Inject labeled and unlabeled anomalies
    4. Generate and embed maintenance log entries
    5. Save results to disk
    """
    print("Running data generation...")
    print("Spike Config:", spike_config)
    print("Config:", config)
    print("Anomaly Config:", anomaly_config)

    # --- Step 1: Simulate base well data ---
    df = data_simulator(config)
    print("✅ Synthetic Data Created")

    # --- Step 2: Corrupt data with spikes/missingness ---
    df_corrupted = corrupt_data_with_spikes(
        df,
        spike_config=spike_config,
        missing_rate=0.01,
        random_seed=random_seed
    )
    print("✅ Data Corrupted")

    # --- Step 3: Inject anomalies into corrupted data ---
    df_with_anomalies, logs_df = inject_anomalies_randomly(
        df_corrupted,
        anomaly_config,
        random_seed=random_seed
    )

    # --- Step 4: Embed maintenance text using a transformer model ---
    model = SentenceTransformer("all-MiniLM-L6-v2")

    notes_embeddings = model.encode(
        logs_df["MaintenanceNotes"].tolist(),
        convert_to_numpy=True
    )
    logs_df["MaintenanceNotesEmbedding"] = [vec for vec in notes_embeddings]

    obs_embeddings = model.encode(
        logs_df["Observations"].tolist(),
        convert_to_numpy=True
    )
    logs_df["ObservationsEmbedding"] = [vec for vec in obs_embeddings]

    # --- Step 5: Save outputs ---
    df_with_anomalies.to_csv("data_simulator/SimulatedWellData.csv", index_label="Timestamp")
    logs_df.to_csv("data_simulator/EngineerLogs.csv", index=False)
    print("📦 Data Saved: SimulatedWellData.csv and EngineerLogs.csv")
