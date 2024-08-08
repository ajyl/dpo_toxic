"""
Unalign experiment.
"""

import os
import torch
from toxicity.eval_interventions.eval_utils import (
    pretty_print_results,
)
from constants import (
    ROOT_DIR,
    PROFANITY,
    SEXUALLY_EXPLICIT,
    IDENTITY_ATTACK,
    THREAT,
    INSULT,
    SEVERE_TOXICITY,
    TOXICITY,
    PERSPECTIVE_API_ATTRIBUTES as ATTRIBUTES,
)
from utils import verbose_print, VERBOSE
from toxicity.eval_interventions.run_evaluations import run_eval

DATA_DIR = os.path.join(ROOT_DIR, "data/intervene_data")
CKPT_DIR = os.path.join(ROOT_DIR, "checkpoints")


def main():
    """Driver"""
    config = {
        "model": {
            "model_or_path": "gpt2-medium",
            "state_dict_path": os.path.join(CKPT_DIR, "dpo.pt"),
            "tokenizer": "gpt2-medium",
            "batch_size": 16,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "unalign": {
                "probe_path": os.path.join(CKPT_DIR, "probe.pt"),
                "num_value_vecs": 7,
                "scale": 10,
            },
        },
        "metrics": [
            {
                "datapath": (
                    os.path.join(DATA_DIR, "challenge_prompts_dev.jsonl")
                    if VERBOSE
                    else os.path.join(DATA_DIR, "challenge_prompts.jsonl")
                ),
                "metric": "perspective_api",
                "max_prompt_size": 32,
                "max_new_tokens": 20,
                "generate": True,
                "params": {"attributes": ATTRIBUTES, "include_prompt": False},
            },
            {
                "datapath": "wikitext",
                "dataname": "wikitext-2-raw-v1",
                "split": "test",
                "metric": "perplexity",
                "generate": False,
            },
            {
                "datapath": (
                    os.path.join(DATA_DIR, "wiki_samples_dev.jsonl")
                    if VERBOSE
                    else os.path.join(DATA_DIR, "wiki_samples.jsonl")
                ),
                "metric": "f1",
                "max_prompt_size": 32,
                "max_new_tokens": 20,
                "generate": True,
            },
        ],
        "interventions": [
            {"method": "noop", "params": {}},
        ],
    }
    results = run_eval(config)
    print("Final Results:")
    pretty_print_results(results)


if __name__ == "__main__":
    main()
