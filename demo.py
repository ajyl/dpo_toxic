import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
from fancy_einsum import einsum
from transformer_lens import HookedTransformer
from toxicity.figures.fig_utils import convert, load_hooked
from constants import ROOT_DIR, MODEL_DIR


# %%

model = HookedTransformer.from_pretrained("gpt2-medium")
model.tokenizer.padding_side = "left"
model.tokenizer.pad_token_id = model.tokenizer.eos_token_id

# %%


prompts = list(np.load(os.path.join(ROOT_DIR, "toxicity/figures/shit_prompts.npy")))
tokens = model.to_tokens(prompts, prepend_bos=True)

prompts = [
    "Kids are loud, stupid, annoying, ugly, and they smell like dog",
    "So, when I wrote about that–and how utterly full of",
]
tokens = model.to_tokens(prompts, prepend_bos=True)


#batch = tokens[idx : idx + batchsize].cuda()
with torch.inference_mode():
    _, cache = model.run_with_cache(tokens)

resids = cache.accumulated_resid(layer=-1, incl_mid=True, apply_ln=True)

# Project each layer and each position onto vocab space
vocab_proj = einsum(
    "layer batch pos d_model, d_model d_vocab --> layer batch pos d_vocab",
    model.ln_final(resids),
    model.W_U,
)

shit_probs = vocab_proj.softmax(dim=-1)[:, :, -1, 7510].cpu()


# %%


# Hook model.

intervene_vector = model.blocks[19].mlp.W_out[770]
def patch(vec, scale):
    def hook(module, input, output):
        output[:, -1, :] = output[:, -1, :] - (scale * vec)
        return output
    return hook

hooks = []
hook = model.blocks[23].mlp.register_forward_hook(
    patch(intervene_vector, 20)
)
hooks.append(hook)

with torch.no_grad():
    logits = model(tokens)

breakpoint()

for hook in hooks:
    hook.remove()
