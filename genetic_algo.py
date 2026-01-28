import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1" 
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from multiprocessing import cpu_count, freeze_support
from dataclasses import dataclass, field
from typing import Tuple
import json
from tqdm import tqdm
from contextlib import contextmanager
import random
import logging

METADATA = "../archive/metadata.json"
CORES = max(cpu_count() - 20, 1)
MAXPOP = CORES * 2


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

def synEval_score(scoreDict: dict):
    # Preliminary naive data conversions for proof-of-concept
            
            # fidelity
            fidelity_diagnostic = scoreDict["fidelity"].get("diagnostic", {}).get("Overall", {}).get("score", 0)
            fidelity_quality = scoreDict["fidelity"].get("quality", {}).get("Overall", {}).get("score", 0)

            fidelity_final = (fidelity_diagnostic + fidelity_quality) / 2

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
            
            # utility_score = abs(results_dict["utility"].get("real_data_model", -9999) - results_dict["utility"].get("synthetic_data_model", 9999))

            # privacy
            privacy = scoreDict.get("privacy", {})
            privacy_matches = privacy.get("exact_matches", {}).get("exact_matches_percentage", 100) / 100
            privacy_mia_auc = privacy.get("membership_inference", {}).get("mia_auc_score", 1)
            privacy_anon_risk = privacy.get("anonymeter", {}).get("overall_risk", {}).get("risk_score", 1)

            privacy_final = ((1 - privacy_matches) + (1 - privacy_mia_auc) + (1 - privacy_anon_risk)) / 3


            return (fidelity_final + privacy_final) / 2

def synEval_call(synthetic: pd.DataFrame, original: pd.DataFrame, metadata: dict):
        """TODO
        """
        # NEED BOTH METADATA AND ORIGINAL DATA (currently class attributes)
        try:
            from run import SynEval
            metric = SynEval(synthetic, original, metadata)
            with silence_all():
                return {"fidelity": metric.evaluate_fidelity(), "privacy": metric.evaluate_privacy()} # "utility": metric.evaluate_utility(), "diversity": metric.evaluate_diversity(), "privacy": metric.evaluate_privacy()}
        
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

            # Preliminary naive data conversions for proof-of-concept
            
            # fidelity
            fidelity_diagnostic = results_dict["fidelity"].get("diagnostic", {}).get("Overall", {}).get("score", 0)
            fidelity_quality = results_dict["fidelity"].get("quality", {}).get("Overall", {}).get("score", 0)

            fidelity_final = (fidelity_diagnostic + fidelity_quality) / 2

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
            
            # utility_score = abs(results_dict["utility"].get("real_data_model", -9999) - results_dict["utility"].get("synthetic_data_model", 9999))

            # privacy
            privacy = results_dict.get("privacy", {})
            privacy_matches = privacy.get("exact_matches", {}).get("exact_matches_percentage", 100) / 100
            privacy_mia_auc = privacy.get("membership_inference", {}).get("mia_auc_score", 1)
            privacy_anon_risk = privacy.get("anonymeter", {}).get("overall_risk", {}).get("risk_score", 1)

            privacy_final = ((1 - privacy_matches) + (1 - privacy_mia_auc) + (1 - privacy_anon_risk)) / 3


            return (fidelity_final + privacy_final) / 2
        except Exception as e:
            print(f"!!! WORKER ERROR: {e}", flush=True)
            return {}


@dataclass
class GASE():
    # model: function
    original: pd.DataFrame
    metadata: dict
    data: Tuple[pd.DataFrame, ...] = field(default_factory=tuple)
    starting_pop: int = 20
    max_generations: int = 50
    population: np.ndarray = field(init=False)

    def __post_init__(self, *args):
        # placeholder code for now
        self.population = np.empty(len(self.data), dtype=object)
        self.population[:] = self.data

        # self.pop_init()
    
    def pop_init(self):
        """Sets up the population and corresponding original data (currently just an even split)
        NEED AN EXPLICIT WAY FOR ROW # TO BE TRACKED FOR ORIGINAL DATA
        """

        split_data = np.array_split(self.data, self.starting_pop)
        # self.population = [self.model(original) for original in split_data]

        return
    
    def run(self):
        for generation in range(self.max_generations):
            self.crossover(13)
            self.mutation()
            score = self.parallel_fitness(max_pop=MAXPOP) # returns scores to be used in logging

            print("Generation:", generation, "completed")
            print("Average population fitness:", np.mean(score), "Top population fitness:", np.max(score))

        print("Final Generation completed, returning highest fitness synthetic data")

        best = self.population[np.argsort(-score)[0]]
        best.to_csv("GASE_dataset.csv", index=False)
        return best
    
    def parallel_fitness(self, max_pop):
        """multiprocessing for SynEval, needs to pass population, original, and metadata to the call
        followed by culling of population based on scores
        """

        from privacy import SequentialTextEvaluator

        if not hasattr(self, 'text_evaluator'):
            print("Loading Sequential Models (Main Process)...")
            self.text_evaluator = SequentialTextEvaluator(self.original, self.metadata)

        scoreDicts = Parallel(n_jobs=CORES, backend="loky")(
            delayed(synEval_call)(
                synthetic,
                self.original,
                self.metadata
            ) for synthetic in tqdm(self.population, desc="Syneval Processes")
        )

        for (results, synthetic) in zip(scoreDicts, self.population):
            results.update(self.text_evaluator.evaluate_synthetic(synthetic))

        score = np.array([synEval_score(scores) for scores in scoreDicts], dtype=float)
        cull_idx = np.argsort(-score)[:max_pop]
        if max_pop < len(self.population):
            self.population = self.population[cull_idx]

        return score
    
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
        
        print("num cores:", CORES)
        children = [weighted_crossover(recipe["parents"], recipe["weights"]) for recipe in generate_crossover_plan(self.population, cross_amount)]
        nd_children = np.empty(len(children), dtype=object)
        nd_children[:] = children
        self.population = np.concatenate((self.population, nd_children))

        return
    
    def mutation(self):
        """TODO
        Plan is to retrigger the llms used for initial population with some novel version of original data,
        then randomly replace rows
        """
        return

if __name__ == "__main__":
    freeze_support()
    original_data = pd.read_csv("../archive/trainer.csv")
    syn_data1 = pd.read_csv("../archive/synthetic_trainer.csv")
    syn_data2 = pd.read_csv("../archive/synthetic_copilot.csv")
    syn_data3 = pd.read_csv("../archive/synthetic_claude.csv")
    with open(METADATA) as f:
        metadata = json.load(f)

    print("finished dataset loading")
    gase = GASE(original=original_data, data=(syn_data1, syn_data2, syn_data3), metadata=metadata)
    gase.run()