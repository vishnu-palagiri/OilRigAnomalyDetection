from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from ast import literal_eval
from itertools import chain

model_id = "microsoft/phi-1_5"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    offload_folder=f"summarizer/offload/{model_id}"
)

# Text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)


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

    # Prompt engineering
    prompt = f"""
    Summarize the anomaly using the following information about similar events.

    You would need to go through the anomaly types and the details to provide a summary of the current selected period's anomaly type.

    Similar Anomaly Types: {types}

    Similar Observations: {observations}

    Similar Maintenance Notes: {notes}

    Return a one-line summary and a short explanation.
    """

    print(prompt)

    # Generate response
    response = generator(
        prompt,
        max_length=min(len(prompt), 60),
        min_length=20,
        do_sample=False,
        temperature=0.3
    )

    print(response)

    return response[0]['generated_text']