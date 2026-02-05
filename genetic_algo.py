import os
# os.environ['MPLBACKEND'] = 'Agg'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1" 
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

# import matplotlib
# matplotlib.use('Agg') 
# import matplotlib.pyplot as plt

import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from multiprocessing import cpu_count, freeze_support
from dataclasses import dataclass, field
from typing import Tuple, Callable
import json
from tqdm import tqdm
from contextlib import contextmanager
import random
import re
import json
import logging

METADATA = "../archive/metadata.json"
CORES = max(cpu_count() - 16, 1)
MAXPOP = CORES * 2
CROSSOVER = 10
MAXGEN = 10
STARTINGSIZE = 200
BATCHSIZE = 10
MAXITERATIONS = STARTINGSIZE / 2


def ratio(val1, val2, smooth = 0) -> float:
    return (min(val1, val2) + smooth) / (max(val1, val2) + smooth)

@contextmanager
def silence_all():
    """
    Context manager that suppresses stdout, stderr, and logging.
    """
    # 1. Save original streams
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    # 2. Open devnull
    devnull = open(os.devnull, 'w')
    
    # 3. Suppress Logging (Critical for ML libraries)
    previous_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    
    try:
        # Redirect streams
        sys.stdout = devnull
        sys.stderr = devnull
        yield
    finally:
        # Restore everything
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        logging.disable(previous_level)
        devnull.close()

"""Idea: using syneval as the fitness function, create a GA tool meant to be used with a generation model to optimize
data in accordance with the metrics.
1. Original idea in my head optmizes the solution, not the generation, so the model is only used for population instantiation
a. select subsets of data from original (high overlap), run the model to create synthetic subsets, then evaluate and cross
b. crossing will have to involve keeping track of original data subsets correlating to synthetic ones when combined,
fitness will be determined against these subsets
c. can either stop at full size, or continue until population fully evolves to one type (will need to consider specifics for this)
2. Maybe should be GAE (genetic algorithms + endosymbiosis) where mutation can also involve introducing new data, that can simply
augment itself into a chromasome (decide whether identical dataset size is needed)
a. May also try implementing a mutation that regenerates data if all possible initial data of an entry is low quality
3. End goal is to gurantee privacy (will remove all matching PII entries) while also consistently improving other metrics
4. Another possible functionality is if a user has multiple models, they can pass their own synthetic data in as the population
a. could provide a module that accepts as many models as the user provides, then uses each of them to create distinct individuals
b. will create a patchwork dataset that theoretically takes the best from each model

initial_placeholder: will be going with truncation selection + uniform mating to preserve diversity
"""

# todo: Main GA loop: population initialization ->  LOOP [SynEval fitness -> crossover -> mutation (ignore for now)]
# population initialization: original data -> sectioning function -> model -> population data array
# SynEval fitness: function for original data mapping -> multithreaded SynEval object calling per individual -> regularization of output -> storage of scores
# crossover: function for handling combination based on scores (tournament, sus, or custom [handle overlap])

def synEval_call(synthetic: pd.DataFrame, original: pd.DataFrame, metadata: dict) -> dict:
        """TODO
        """
        # NEED BOTH METADATA AND ORIGINAL DATA (currently class attributes)
        try:
            from run import SynEval
            metric = SynEval(synthetic, original, metadata)
            input_columns = metadata["utility"].get("input_columns", [])
            output_columns = metadata["utility"].get("output_columns", [])
            with silence_all():
                return {"fidelity": metric.evaluate_fidelity(), "utility": metric.evaluate_utility(input_columns, output_columns), "privacy": metric.evaluate_privacy()}
        
            # NEED TO DETERMINE ALGORITHM FOR COMBINING WEIGHTS (later on can perform advanced metric specifics)
            # Main idea is to implement diminishing returns where 0.8 is the inflection point? Want to penalize overspecialization
            """
            Fidelity:
            Already using diagnostic/quality scores, seem to already be improving all fidelity scores
            overall fidelity scores already combine numerical metrics
            Diversity: 
            coverage scores good as is, maybe take overall coverage (average)
            - going deeper, maybe use specific numerical/categorical coverage scores
            penalize duplication in synthetic data (can perform relative calculation manually if original duplication is 0)
            Entropy ratio/score difference (not sure how important but should look into)
            text scores will have to be taken individually and combined, weight will be proportional to #text columns/total columns


            """
        except Exception as e:
            print(f"!!! WORKER ERROR: {e}", flush=True)
            return {}


@dataclass
class GASE():
    # model: function
    original: pd.DataFrame
    metadata: dict
    models: Tuple[Callable, ...] = field(default_factory=tuple)
    data: Tuple[pd.DataFrame, ...] = field(default_factory=tuple)
    starting_pop: int = 20
    starting_size: int = STARTINGSIZE
    batch_size: int = BATCHSIZE
    max_generations: int = MAXGEN
    population: np.ndarray = field(init=False)

    def __post_init__(self):
        # placeholder code for now
        initial = []
        for i, func in enumerate(self.models):
            try:
                if callable(func):
                     initial.append(self.generate_member(func))
                else:
                    print(f"noncallable function {i} passed, skipping")
            except Exception as e:
                print(f"failure in function {i} execution, error: {e}, skipping")
        initial += self.data
        self.population = np.empty(len(initial), dtype=object)
        self.population[:] = initial

        # self.pop_init()
    
    def process_llm_output(self, output) -> pd.DataFrame:
        columns = list(self.original.columns)
        # ROBUST PARSING
        data = recover_json(output)
        
        # Extract list of rows
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            # Try 'rows', 'data', or find the first list value
            rows = data.get("rows") or data.get("data") or next((v for v in data.values() if isinstance(v, list)), [])
        else:
            rows = []

        print("check")

        if not rows:
            return pd.DataFrame(columns=columns)

        # 5. ALIGNMENT
        new_df = pd.DataFrame(rows)
        new_df = align_columns(new_df, columns)
        new_df = allign_categorical_vals(new_df, self.metadata)


        # 6. VALIDATION
        return validate_output(new_df, self.metadata)
    
    def generate_member(self, model: Callable) -> pd.DataFrame:
        """generates specified length dataset from provided model function

        Args:
            model (_type_): _description_

        Returns:
            pd.DataFrame: _description_
        """

        import ollama
    
        df_list = []
        total_rows = 0
        iterations = 0
        while (total_rows < self.starting_size and iterations < MAXITERATIONS):
            df = self.process_llm_output(model(self.original))
            total_rows += len(df)
            iterations += 1
            df_list.append(df)
            print("rows remaining:", self.starting_size - total_rows)
        df = pd.concat(df_list, ignore_index=True)
        
        # PLACEHOLDER CODE!! IDEALLY NEVER NEED THIS FOR CALLABLES, ONLY MODEL STR ARGUMENTS
        try:
            ollama.generate(model="phi3", prompt="", keep_alive=0)
            print("Ollama model unloaded from VRAM.")
        except:
            pass

        return df
        
    def pop_init(self):
        """Sets up the population and corresponding original data (currently just an even split)
        NEED AN EXPLICIT WAY FOR ROW # TO BE TRACKED FOR ORIGINAL DATA
        """

        split_data = np.array_split(self.data, self.starting_pop)
        # self.population = [self.model(original) for original in split_data]

        return
    
    def run(self):
        for generation in range(self.max_generations):
            self.crossover(CROSSOVER)
            score = self.parallel_fitness(max_pop=MAXPOP) # returns scores to be used in logging
            # self.mutation(score)

            print("Generation:", generation, "completed")
            print("Average population fitness:", np.mean(score), "Top population fitness:", np.max(score))

        print("Final Generation completed, returning highest fitness synthetic data")

        best = self.population[np.argsort(-score)[0]]
        best.to_csv("GASE_dataset3.csv", index=False)
        return best
    
    def parallel_fitness(self, max_pop):
        """multiprocessing for SynEval, needs to pass population, original, and metadata to the call
        followed by culling of population based on scores
        """

        from privacy import SequentialTextEvaluator

        if not hasattr(self, 'text_evaluator'):
            print("Loading Sequential Models (Main Process)...")
            self.text_evaluator = SequentialTextEvaluator(self.original, self.metadata)

        job_generator = Parallel(n_jobs=CORES, return_as="generator", backend="loky")(
            delayed(synEval_call)(
                synthetic,
                self.original,
                self.metadata
            ) for synthetic in self.population
        )

        scoreDicts = [result for result in tqdm(job_generator, total=len(self.population), desc="Syneval Processes")]

        print("Beginning Sequential step:")
        for (results, synthetic) in zip(scoreDicts, self.population):
            output = self.text_evaluator.evaluate_synthetic(synthetic)
            print(output)
            results.update(output)

        score = np.array([self.synEval_score(scores) for scores in scoreDicts], dtype=float)
        cull_idx = np.argsort(-score)[:max_pop]
        if max_pop < len(self.population):
            self.population = self.population[cull_idx]
            score = score[cull_idx]

        return score
    
    def synEval_score(self, scoreDict: dict):
        # should the text score to tabular score just be the actual column ratio? or value text more
        # start with ratio

        # Preliminary naive data conversions for proof-of-concept
                
        # fidelity
        fidelity_diagnostic = scoreDict["fidelity"].get("diagnostic", {}).get("Overall", {}).get("score", 0)
        fidelity_quality = scoreDict["fidelity"].get("quality", {}).get("Overall", {}).get("score", 0)

        # text metrics
        fidelity_text = scoreDict["fidelity"].get("text", {})

        fidelityText_len_mean = 0
        fidelityText_len_std = 0
        fidelityText_count_mean = 0
        fidelityText_count_std = 0

        fidelityText_sentiment_mean = 0
        fidelityText_sentiment_std = 0
        keys = fidelity_text.keys()

        text_ratio = len(keys) / len(self.metadata["columns"].keys())

        fidelity_text_mean = 0

        if text_ratio:
            for col in keys:
                dict = fidelity_text[col]
                fidelityText_len_mean += ratio(dict.get("length_stats", {}).get("original_mean", 0), dict.get("length_stats", {}).get("synthetic_mean", 0))
                fidelityText_len_std += ratio(dict.get("length_stats", {}).get("original_std", 0), dict.get("length_stats", {}).get("synthetic_std", 0))

                fidelityText_count_mean += ratio(dict.get("word_count_stats", {}).get("original_mean", 0), dict.get("word_count_stats", {}).get("synthetic_mean", 0))
                fidelityText_count_std += ratio(dict.get("word_count_stats", {}).get("original_std", 0), dict.get("word_count_stats", {}).get("synthetic_std", 0))

                fidelityText_sentiment_mean += ratio(dict.get("sentiment_analysis", {}).get("original_mean", 0) + 1, dict.get("sentiment_analysis", {}).get("synthetic_mean", 0) + 1)
                fidelityText_sentiment_std += ratio(dict.get("sentiment_analysis", {}).get("original_std", 0), dict.get("sentiment_analysis", {}).get("synthetic_std", 0))
            
            fidelity_text_mean = (fidelityText_len_mean + fidelityText_len_std + fidelityText_count_mean + fidelityText_count_std + fidelityText_sentiment_mean + fidelityText_sentiment_std) / (6 * len(keys))

        fidelity_final = text_ratio * fidelity_text_mean + (1 - text_ratio) * ((fidelity_diagnostic + fidelity_quality) / 2)

        # diversity
        # tabular_diversity = results_dict["diversity"].get("tabular_diversity", {})
        # diversity_coverage = np.array([tabular_diversity.get(key, 0) for key in tabular_diversity.get("coverage", {}).keys()]).mean()
        # diversity_syn_dupe_ratio = tabular_diversity.get("uniqueness", {}).get("synthetic_duplicate_ratio", 0)
        # diversity_orig_dupe_ratio = tabular_diversity.get("uniqueness", {}).get("original_duplicate_ratio", 0)

        # jeffrey_prior = 0.5 / len(synthetic) # jeffrey's smoothing ratio for 0 cases
        # num = min(diversity_syn_dupe_ratio, diversity_orig_dupe_ratio) + jeffrey_prior
        # denom = max(diversity_syn_dupe_ratio, diversity_orig_dupe_ratio) + jeffrey_prior
        # diversity_uniqueness = num / denom # score is based around distance from ratio match

        # diversity_entropy = tabular_diversity.get("entropy_metrics", {}).get("dataset_entropy", {}).get("entropy_ratio", 0)
        # if diversity_entropy > 1:
        #     diversity_entropy = 1 / diversity_entropy

        # diversity_final = (diversity_coverage + diversity_uniqueness + diversity_entropy) / 3
        
        # utility
        utility_real = scoreDict["utility"].get("real_data_model", {})
        utility_syn = scoreDict["utility"].get("synthetic_data_model", {})

        # SYNEVAL HAS NO REGRESSION UTILITY IMPLEMENTATION
        # if scoreDict["utility"].get("task_type", "") == "classification" or scoreDict["utility"].get("task_type", "") == "text_classification":
        utility_real_acc = utility_real.get("accuracy", 0)
        utility_syn_acc = utility_syn.get("accuracy", 0)
        utility_acc_ratio = min(utility_real_acc, utility_syn_acc) / max(utility_real_acc, utility_syn_acc)

        # using macro avg for now, but should compare results with micro
        utility_real_f1 = utility_real.get("macro avg", {}).get("f1-score", 0)
        utility_syn_f1 = utility_syn.get("macro avg", {}).get("f1-score", 0)
        utility_f1_ratio = min(utility_real_f1, utility_syn_f1) / max(utility_real_f1, utility_syn_f1)

        utility_final = utility_f1_ratio


        # privacy
        privacy = scoreDict.get("privacy", {})
        privacy_matches = privacy.get("exact_matches", {}).get("exact_matches_percentage", 100) / 100
        privacy_mia_auc = privacy.get("membership_inference", {}).get("mia_auc_score", 1)
        # privacy_anon_risk = privacy.get("anonymeter", {}).get("overall_risk", {}).get("risk_score", 1)
        privacy_anon_inf = 0
        for col in privacy.get("anonymeter", {}).get("inference", []):
            privacy_anon_inf += col.get("risk", 1)
        privacy_anon_inf /= len(self.metadata["columns"])


        # privacy text
        privacy_text_mean = 0
        if text_ratio:
            privacy_NER = privacy.get("named_entities", {}).get("overlap", {}).get("overlap_percentage", 100) / 100
            privacy_nom = privacy.get("nominal_mentions", {}).get("overlap", {}).get("overlap_percentage", 100) / 100
            privacy_style_orig = privacy.get("stylistic_outliers", {}).get("original", {}).get("outlier_percentage", 0)
            privacy_style_syn = privacy.get("stylistic_outliers", {}).get("synthetic", {}).get("outlier_percentage", 100)
            privacy_style = ratio(privacy_style_orig, privacy_style_syn)

            privacy_text_mean = (privacy_NER + privacy_nom + privacy_style) / 3

        privacy_final = text_ratio * privacy_text_mean + (1 - text_ratio) * (1 - privacy_matches) * (((1 - privacy_mia_auc) + (1 - privacy_anon_inf)) / 2)


        return (fidelity_final + utility_final + privacy_final) / 3

    
    def crossover(self, cross_amount=10):
        """crossover static amount of times
        """

        def weighted_crossover(parents, weights):
            """
            parents: list of DataFrames [df1, df2, ...]
            weights: list of probabilities [0.8, 0.2]
            """
            n_samples = [int(w * len(parent)) for w, parent in zip(weights, parents)]
            
            parts = []
            for i, df in enumerate(parents):
                sample = df.sample(n=n_samples[i], replace=False) 
                parts.append(sample)
            
            child = pd.concat(parts).sample(frac=1).reset_index(drop=True)
            return child

        def generate_crossover_plan(survivors, num_children=10):
            """
            Generates a list of crossover 'recipes' based on the surviving population.
            """
            plan = []
            num_survivors = len(survivors)
            indices = range(num_survivors)

            for i in range(num_children):
                strategy_roll = i % 4 
                
                if strategy_roll == 0:
                    parents = random.sample(indices, 2)
                    weights = [0.5, 0.5]
                    
                elif strategy_roll == 1:
                    parents = random.sample(indices, 2)
                    weights = [0.8, 0.2] 
                    
                elif strategy_roll == 2 and num_survivors >= 3:
                    parents = random.sample(indices, 3)
                    weights = [0.33, 0.33, 0.34]
                
                else:
                    n_parents = random.randint(2, min(4, num_survivors))
                    parents = random.sample(indices, n_parents)
                    
                    # Dirichlet distribution creates random weights that sum to 1.0
                    weights = np.random.dirichlet(np.ones(n_parents)).tolist()
                    weights = [round(w, 2) for w in weights]

                plan.append({
                    'parents': [survivors[p] for p in parents],
                    'weights': weights
                })
            
            return plan
        
        children = [weighted_crossover(recipe["parents"], recipe["weights"]) for recipe in generate_crossover_plan(self.population, cross_amount)]
        nd_children = np.empty(len(children), dtype=object)
        nd_children[:] = children
        self.population = np.concatenate((self.population, nd_children))

        return
    
    def mutation(self, scores):
        """TODO
        Plan is to retrigger the llms used for initial population with some novel version of original data,
        then randomly replace rows
        """

        if np.random.rand() < 0.2:
            print("triggering mutation event:")
            new_rows: pd.DataFrame = (self.models[np.random.randint(0, len(self.models))])(self.original)
            indices = np.random.randint(0, len(self.population), size=len(new_rows))
            for i in range(len(new_rows)):
                target_df = self.population[indices[i]]
                row_idx = np.random.randint(0, len(target_df))
                target_df.values[row_idx] = new_rows[i]

        return


def format_constraints(metadata_json):
    """
    Converts the metadata JSON into a strict text schema for the LLM.
    """
    constraints = []
    
    for col_name, info in metadata_json['columns'].items():
        # Handle Categorical Columns (The most important part!)
        if info['sdtype'] == 'categorical' and 'values' in info:
            # Join options with pipes or commas
            options = ", ".join([str(v) for v in info['values']])
            constraints.append(f"- Column '{col_name}' allowed values: [{options}]")
            
        # Handle Numerical Columns
        elif info['sdtype'] == 'numerical':
            constraints.append(f"- Column '{col_name}' must be type {info['ntype']}.")
            
    return "\n".join(constraints)


def align_columns(synthetic_df: pd.DataFrame, original_columns: list) -> pd.DataFrame:
    """
    Renames and aligns columns. 
    Coalesces split columns (e.g. 'hours-per-week' and 'hours_per_week').
    """
    def get_fingerprint(col_name):
        return re.sub(r'[^a-z0-9]', '', str(col_name).lower())

    # Create the result container
    aligned_df = pd.DataFrame(index=synthetic_df.index)
    
    for target_col in original_columns:
        target_fp = get_fingerprint(target_col)
        
        # Find all columns in the synthetic data that match this fingerprint
        matching_cols = [
            col for col in synthetic_df.columns 
            if get_fingerprint(col) == target_fp
        ]
        
        if not matching_cols:
            aligned_df[target_col] = None
        elif len(matching_cols) == 1:
            aligned_df[target_col] = synthetic_df[matching_cols[0]]
        else:
            # Merge split columns
            # 'bfill' ensures if col A is NaN but col B has data, we keep data.
            # We explicitly infer objects to prevent type errors during fill
            combined = synthetic_df[matching_cols].bfill(axis=1).iloc[:, 0]
            aligned_df[target_col] = combined

    return aligned_df


def recover_json(raw_content):
    """
    attempts to find useable json formatting in proper and improper llm output
    outputs dictionary if json found
    """
    try:
        return json.loads(raw_content)
    except json.JSONDecodeError:
        print("JSON ERROR")
        print(raw_content)
        matches = re.findall(r'\{[^{}]*\}', raw_content)
        if matches:
            fixed_json = f'{{"rows": [{",".join(matches)}]}}'
            try:
                return json.loads(fixed_json)
            except:
                print("JSON STILL BROKEN AFTER REPAIR")
                return {} # Failed repair
        else:
            print("NO JSON MATCHES")
            return {}


def allign_categorical_vals(df: pd.DataFrame, metadata_json: dict) -> pd.DataFrame:
    """
    Attempts to map fuzzy values back to the allowed metadata list.
    Example: Maps 'Puerto Rico' -> 'Puerto-Rico', 'Private Sector' -> 'Private'
    """
    import difflib
    try:
        df = df.copy()
        
        for col, info in metadata_json['columns'].items():
            if col not in df.columns or info['sdtype'] != 'categorical':
                continue
                
            allowed_values = info['values']
            # Create a lowercase map for exact case-insensitive matching first
            lower_map = {str(v).lower(): v for v in allowed_values}
            
            def fix_value(val):
                if pd.isna(val): return val
                s_val = str(val).strip()
                
                # 1. Exact Match
                if s_val in allowed_values:
                    return s_val
                    
                # 2. Case-insensitive Match
                if s_val.lower() in lower_map:
                    return lower_map[s_val.lower()]
                
                # 3. Fuzzy Match (e.g. 'Puerto Rico' -> 'Puerto-Rico')
                # get_close_matches returns a list, we take the top 1 if robust enough
                matches = difflib.get_close_matches(s_val, [str(v) for v in allowed_values], n=1, cutoff=0.8)
                if matches:
                    return matches[0]
                
                return val # Return original (will be dropped by validator later)

            df[col] = df[col].apply(fix_value)
            
        return df
    except Exception as e:
        print(f"error: {e} in categorical allignment")
        return df


def validate_output(df: pd.DataFrame, metadata_json: dict) -> pd.DataFrame:
    """
    Strictly validates rows against metadata constraints.
    Drops any row that contains an invalid categorical value or out-of-bounds number.
    """
    if df.empty:
        return df

    # We will build a boolean mask (True = Keep, False = Drop)
    # Start with all True
    keep_mask = pd.Series(True, index=df.index)
    print("original length:", len(df))

    for col, info in metadata_json['columns'].items():
        if col not in df.columns:
            continue
            
        # 1. Validate Categorical Columns
        if info['sdtype'] == 'categorical' and 'values' in info:
            allowed_values = set(info['values'])
            
            # Check if value is in allowed list
            # We use map instead of isin to handle potential type mismatches (str vs int) better
            is_valid = df[col].astype(str).isin([str(v) for v in allowed_values])
            
            # Debug: Print what we are dropping
            invalid_count = (~is_valid).sum()
            if invalid_count > 0:
                print(f"   -> Dropping {invalid_count} rows due to invalid '{col}'")
            
            keep_mask = keep_mask & is_valid

        # 2. Validate Numerical Columns (Basic sanity checks)
        elif info['sdtype'] == 'numerical':
            # Check if it can be numeric (not "ten" or "N/A")
            is_numeric = pd.to_numeric(df[col], errors='coerce').notna()
            
            # Optional: Add specific bounds if you know them
            # e.g., age must be > 0
            # if col == 'age':
            #     is_numeric = is_numeric & (pd.to_numeric(df[col], errors='coerce') > 0)
                
            keep_mask = keep_mask & is_numeric

    # Apply the mask
    clean_df = df[keep_mask].copy()
    
    # Fix Types: Enforce numerical types on numerical columns
    # (LLMs often output strings "40" instead of int 40)
    for col, info in metadata_json['columns'].items():
        if col in clean_df.columns and info['sdtype'] == 'numerical':
            clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
    
    print("final length:", len(clean_df))
    return clean_df


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
    print(sample_csv)
    # Get strict data types to allow model to understand constraints
    constraints = format_constraints()
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
                'num_ctx': 4096
            }
        )
        
        print(response['response'])
        return response['response']

    except Exception as e:
        print(f"Mutation Error: {e}")
        return pd.DataFrame(columns=columns)


if __name__ == "__main__":
    freeze_support()
    original_data = pd.read_csv("../archive/trainer.csv")
    syn_data1 = pd.read_csv("../archive/synthetic_trainer.csv")
    syn_data2 = pd.read_csv("../archive/synthetic_copilot.csv")
    syn_data3 = pd.read_csv("../archive/synthetic_claude.csv")
    with open(METADATA) as f:
        metadata = json.load(f)

    print("finished dataset loading")
    # gase = GASE(original=original_data, models=(generate_synthetic_data_local), data=(syn_data1, syn_data2, syn_data3), metadata=metadata)
    # gase.run()
    print(generate_synthetic_data_local(original_df=original_data, num_rows=10, constraints=format_constraints(metadata)))