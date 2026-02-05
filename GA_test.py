import os
os.environ['OLLAMA_NUM_PARALLEL']="2"
from genetic_algo import GASE, format_constraints
import pandas as pd
import json

METADATA = json.load(open("../archive/metadata.json"))
ORIGINALDATA = pd.read_csv("../archive/trainer.csv")
SYNDATA1 = pd.read_csv("../archive/synthetic_trainer.csv")
SYNDATA2 = pd.read_csv("../archive/synthetic_copilot.csv")
SYNDATA3 = pd.read_csv("../archive/synthetic_claude.csv")
BATCHSIZE = 10

def generate_synthetic_data_local(
    original_df: pd.DataFrame, 
    num_rows: int = BATCHSIZE,
    model: str = "phi3"
) -> pd.DataFrame:
    
    import ollama
    # import torch
    # import gc
    # 1. PRE-FLIGHT: Clear VRAM so Ollama can fit
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    #     gc.collect()

    columns = list(original_df.columns)
    # 1. Get a Better Sample (Randomize it)
    # Don't just take head(3). Take 3 random rows so the model sees variety.
    sample_csv = original_df.sample(3).to_string(index=False)
    # Get strict data types to allow model to understand constraints
    constraints = format_constraints(METADATA)
    prompt = f"""
    You are a synthetic data generator.
    Generate {num_rows} new rows of data that strictly follow the schema rules below.
    
    STRICT SCHEMA RULES (You MUST follow these):
    {constraints}
    
    Reference Data (For context only):
    {sample_csv}
    
    Output Requirements:
    1. Output valid JSON only: {{ "rows": [...] }}
    2. Do NOT invent new categories. Only use the "allowed values" listed above.
    3. Vary the data. Do not just copy the Reference Data.
    4. No NaNs or Nulls.
    """

    try:
        response = ollama.generate(
            model=model, 
            prompt=prompt, 
            format='json',
            options={
                'temperature': 0.6, # Lowered from 0.8 to reduce hallucinations
                'top_p': 0.9,
                'num_ctx': 2000
            }
        )
        
        return response['response']

    except Exception as e:
        print(f"Mutation Error: {e}")
        return pd.DataFrame(columns=columns)
    
if __name__ == "__main__":
    gase = GASE(original=ORIGINALDATA, models=(generate_synthetic_data_local,), data=(SYNDATA1, SYNDATA2, SYNDATA3), metadata=METADATA, max_generations=5)
    gase.run()