"""
Functions to run each metric.
"""

import math
from collections import Counter
from tqdm import tqdm
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from toxicity.PerspectiveAPI import (
    PerspectiveAPI,
    parse_response_payload as parse_toxic_scores,
)
from toxicity.eval_interventions.hook_utils import hook_subtract
from utils import VERBOSE


rate_limit = 20
if VERBOSE:
    rate_limit = 5
perspective_api_client = PerspectiveAPI(rate_limit=rate_limit)


def run_dummy(
    model,
    data,
    intervene_results,
    config,
):
    """
    Returns a constant.
    """
    return 42


def run_f1(
    model,
    data,
    intervene_results,
    config,
):
    """
    Calculate prec, recall, f1 for 1d tensors.
    """
    # LongTensor: [Batch, Seq]
    pred_tokens = intervene_results["pred_tokens"]
    gold_tokens = data["gold_input_ids"]
    assert pred_tokens.shape[0] == gold_tokens.shape[0]
    pad_token_id = model.tokenizer.pad_token_id
    f1s = []
    for idx in range(pred_tokens.shape[0]):
        _gold = [x for x in gold_tokens[idx].tolist() if x != pad_token_id]
        _pred = [x for x in pred_tokens[idx].tolist() if x != pad_token_id]
        common = Counter(_gold) & Counter(_pred)
        num_same = sum(common.values())
        if num_same == 0:
            f1s.append(0)
            continue

        prec = 1.0 * num_same / len(_pred)
        recall = 1.0 * num_same / len(_gold)
        f1 = (2 * prec * recall) / (prec + recall)
        f1s.append(f1)

    return np.mean(f1s)


def run_perplexity(
    model,
    data,
    intervene_results,
    config,
):
    """
    Calculate perplexity.

    :data:
    """
    encodings = data["prompt_input_ids"]

    max_length = 1024
    stride = max_length

    ending = encodings.shape[1]
    if VERBOSE:
        ending = max_length * 4

    lls = []
    for i in tqdm(range(0, ending, stride)):
        # iterates through all 287644 tokens in wikitext test in windows of stride (usually max_length)
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.shape[1])
        trg_len = end_loc - i  # may be different from stride on last loop

        input_ids = encodings[:, begin_loc:end_loc].to(config["device"])

        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()

            loss_fn = CrossEntropyLoss()
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
            )
            lls.append(loss * trg_len)
        ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    return ppl


def _parse_toxicity_scores(scores, config):
    """
    Parse toxicity scores
    """
    key_attribs = config["attributes"]
    parsed_scores = []
    for score_obj in scores:
        parsed = parse_toxic_scores(score_obj)
        parsed_scores.append(max([parsed[att] for att in key_attribs]))
    return parsed_scores


def run_perspective_api(
    model,
    data,
    intervene_results,
    config,
):
    generations = intervene_results["pred_text"]
    if not config["include_prompt"]:
        generations = [
            intervene_results["pred_text"][idx].replace(
                data["prompts"][idx], ""
            )
            for idx in range(len(intervene_results["pred_text"]))
        ]
    toxicity_scores = perspective_api_client.request_loop_with_delay(
        generations
    )
    parsed_scores = _parse_toxicity_scores(toxicity_scores, config)
    return np.mean(parsed_scores)
