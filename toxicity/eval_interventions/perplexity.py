# modified from: https://github.com/timoschick/self-debiasing/blob/main/perplexity.py

import argparse
import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer

from nlp import load_dataset
from toxic_suppression_wrapper import GPT2Wrapper
import pandas as pd


def compute_ppl(
    tokenizer,
    wrapper,
    values_per_layer=None,
    coef_value=3,
    use_cuda=False,
    max_length=-1,
    stride=-1,
):
    """
    Computes perplexity on the test set of WikiText2
    """

    device = "cuda:0" if torch.cuda.is_available() and use_cuda else "cpu"

    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    max_length = (
        max_length if max_length > 0 else wrapper._model.config.n_positions
    )

    if stride <= 0:
        stride = max_length

    lls_non_toxic, lls_regular = [], []
    ppl_non_toxic, ppl_regular = None, None

    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        # iterates through all 287644 tokens in wikitext test in windows of stride (usually max_length)
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop

        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)

        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # i have no idea what this line does

        with torch.no_grad():
            loss_regular = wrapper.compute_loss(input_ids, labels=target_ids)

            wrapper.set_value_activations(
                values_per_layer, coef_value=coef_value
            )

            loss_non_toxic = wrapper.compute_loss(input_ids, labels=target_ids)

            wrapper.remove_all_hooks()

            log_likelihood_non_toxic = loss_non_toxic * trg_len
            log_likelihood_regular = loss_regular * trg_len

        lls_non_toxic.append(log_likelihood_non_toxic)
        lls_regular.append(log_likelihood_regular)

        ppl_non_toxic = torch.exp(torch.stack(lls_non_toxic).sum() / end_loc)
        ppl_regular = torch.exp(torch.stack(lls_regular).sum() / end_loc)
        print(
            f"Perplexity after {i} tokens: {ppl_non_toxic} (non-toxic) vs {ppl_regular} (regular)"
        )

    print(
        f"Final perplexity: {ppl_non_toxic} (non-toxic) vs {ppl_regular} (regular)"
    )

    return ppl_non_toxic, ppl_regular


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="gpt2-medium",
        choices=["gpt2-xl", "gpt2", "gpt2-medium", "gpt2-large"],
    )
    parser.add_argument("--coef_value", default=3, type=float)
    parser.add_argument("--values_filepath", type=str)
    parser.add_argument("--use_cuda", action="store_true")

    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    wrapper = GPT2Wrapper(args.model_name, use_cuda=args.use_cuda)

    values = pd.read_pickle(args.values_filepath)

    ppl_non_toxic, ppl_regular = compute_ppl(
        tokenizer,
        wrapper,
        values_per_layer=values,
        coef_value=args.coef_value,
        use_cuda=args.use_cuda,
    )
