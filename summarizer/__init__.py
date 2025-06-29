from ast import literal_eval
from itertools import chain
import requests
import os

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/generate"
MODEL_NAME = "llama3.1"

def summarize_predictions(df):
    """
    Summarizes anomaly details using LLM based on similar events matched via embeddings.

    Parameters:
    - df (pd.DataFrame): DataFrame with columns ['SimilarAnomalyTypes', 'SimilarMaintenanceNotes', 'SimilarObservations']

    Returns:
    - str: Textual summary generated from LLM
    """

    # Flatten and deduplicate list strings from dataframe columns
    types = '\n- ' + '\n- '.join(set(chain.from_iterable(
        df['SimilarAnomalyTypes'].dropna().apply(literal_eval)
    )))
    observations = '\n- ' + '\n- '.join(set(chain.from_iterable(
        df['SimilarMaintenanceNotes'].dropna().apply(literal_eval)
    )))
    notes = '\n- ' + '\n- '.join(set(chain.from_iterable(
        df['SimilarObservations'].dropna().apply(literal_eval)
    )))

    if types == '\n- ':
        types = "No Anomalies Identified"
        observations = "No Anomalies Identified"
        notes = "No Anomalies Identified"

    prompt = f"""
    You are an experienced Oil & Gas production engineer analyzing well anomalies. Based on historical data from similar anomalous events, provide actionable insights for the current anomaly.

    TASK:
    Analyze the provided historical anomaly data and generate:
    1. ONE-LINE SUMMARY: A concise description of the anomaly type
    2. SHORT EXPLANATION: Key characteristics, likely causes, and recommended actions (2-3 sentences)

    INSTRUCTIONS:
    - Use technical language appropriate for field operations
    - Focus on actionable insights for maintenance teams
    - If no similar anomalies are provided or data is insufficient, state: "No comparable historical anomalies identified - recommend manual inspection to verify if this represents a true anomaly or potential misclassification"

    HISTORICAL DATA:
    Similar Anomaly Types: {types}
    Similar Observations: {observations}  
    Similar Maintenance Notes: {notes}

    RESPONSE FORMAT:
    **One-line Summary:** [Your summary here]
    **Explanation:** [Your 2-3 sentence explanation here]

    TONE: Confident, concise, and written as a hands-on production engineer.
    """

    print(prompt)

    # Create payload
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,  # TODO - Set this to True for streaming Q&A by engineers
        "temperature": 0.1,
    }

    response = requests.post(OLLAMA_URL, json=payload)

    print(response.json())

    try:
        return response.json()['response']
    except Exception as e:
        return f"Errored out due to : {e}"