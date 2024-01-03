# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import os
import json

import pandas as pd
import einops

import torch
import torch.nn.functional as F
from fancy_einsum import einsum
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from transformer_lens import (
    HookedTransformer,
)
from toxicity.figures.fig_utils import convert, load_hooked, get_svd
from constants import MODEL_DIR, DATA_DIR

torch.set_grad_enabled(False)

# %%

model = load_hooked(
    "gpt2-medium",
    os.path.join(MODEL_DIR, "dpo.pt"),
)
gpt2 = HookedTransformer.from_pretrained("gpt2-medium")
gpt2.tokenizer.padding_side = "left"
gpt2.tokenizer.pad_token_id = gpt2.tokenizer.eos_token_id

toxic_vector = torch.load(os.path.join(MODEL_DIR, "probe.pt"))

# %%

with open(
    os.path.join(DATA_DIR, "intervene_data/challenge_prompts.jsonl"), "r"
) as file_p:
    data = file_p.readlines()

prompts = [json.loads(x.strip())["prompt"] for x in data]
tokenized_prompts = model.to_tokens(prompts, prepend_bos=True).cuda()

# %%


_, scores_gpt2 = get_svd(gpt2, toxic_vector, 128)
vectors_of_interest = [
    (_score_obj[2], _score_obj[1], _score_obj[0])
    for _score_obj in scores_gpt2[:64]
]


# %%

gpt2_resid = []
dpo_resid = []
sample_size = 50
batch_size = 4
print("Grabbing mlp mids...")
_vec = vectors_of_interest[0]
_layer = _vec[0]
_idx = _vec[1]

for idx in tqdm(range(0, sample_size, batch_size)):
    batch = tokenized_prompts[idx : idx + batch_size, :]
    dpo_batch = batch.clone()

    with torch.inference_mode():
        _, cache = gpt2.run_with_cache(batch)
        resid = cache[f"blocks.{_layer}.hook_resid_mid"][:, -1, :]

    gpt2_resid.extend(resid.cpu().tolist())

    with torch.inference_mode():
        _, cache = model.run_with_cache(dpo_batch)
        resid = cache[f"blocks.{_layer}.hook_resid_mid"][:, -1, :]
    dpo_resid.extend(resid.cpu().tolist())


w_ins = [
    gpt2.blocks[_layer].mlp.W_in[:, _idx].cpu(),
    model.blocks[_layer].mlp.W_in[:, _idx].cpu(),
]

gpt2_stacked = torch.stack([torch.Tensor(x) for x in gpt2_resid], dim=0)
dpo_stacked = torch.stack([torch.Tensor(x) for x in dpo_resid], dim=0)
gpt2_dots = einsum("sample d_model, d_model", gpt2_stacked, w_ins[0])
dpo_dots = einsum("sample d_model, d_model", dpo_stacked, w_ins[1])
gpt_acts = model.blocks[0].mlp.act_fn(gpt2_dots)
dpo_acts = model.blocks[0].mlp.act_fn(dpo_dots)


# %%

all_data = torch.concat([gpt2_stacked, dpo_stacked], dim=0)
mean = all_data.mean(dim=0)
stddev = all_data.std(dim=0)
normalized = (all_data - mean) / stddev

U, S, V = torch.pca_lowrank(normalized)

diff = dpo_stacked - gpt2_stacked
diff_mean = diff.mean(dim=0)
print(diff_mean.shape)

comps = torch.concat([diff_mean.unsqueeze(-1), V], dim=1)
comps = comps[:, :2]
projected = torch.mm(normalized, comps)

pca_raw = []
num_samples = 30
for idx in range(num_samples):

    _activation = gpt_acts[idx].item()
    if _activation > 15:
        act = "High (> 15)"
    elif _activation > 0:
        act = "Low (> 0)"
    else:
        act = "None"

    print(_activation)
    pca_raw.append(
        {
            "Model": "GPT2",
            "x": projected[idx, 0].item(),
            "y": projected[idx, 1].item(),
            "Activated": act,
        }
    )

_offset = len(gpt2_resid)
print("____")
for idx in range(num_samples):
    _activation = dpo_acts[idx].item()
    if _activation > 15:
        act = "High (> 15)"
    elif _activation > 0:
        act = "Low (> 0)"
    else:
        act = "None"
    print(_activation)
    pca_raw.append(
        {
            "Model": "DPO",
            "x": projected[_offset + idx, 0].item(),
            "y": projected[_offset + idx, 1].item(),
            "Activated": act,
        }
    )

pca_data = pd.DataFrame(pca_raw)
sns.set_theme(context="paper", style="ticks", rc={"lines.linewidth": 1})

fig = sns.relplot(
    pca_data,
    x="x",
    y="y",
    hue="Activated",
    palette={"High (> 15)": "red", "Low (> 0)": "orange", "None": "green"},
    hue_order=["High (> 15)", "Low (> 0)", "None"],
    style="Model",
    markers={"GPT2": "o", "DPO": "^"},
    height=2.5,
    aspect=3.25 / 2.5,
    s=60,
    legend="full",
)

fig.ax.set_xticks([])
fig.ax.set_yticks([])
fig.ax.xaxis.label.set_text("Shift Component")
fig.ax.yaxis.label.set_text("Principle Component")
fig.ax.xaxis.label.set_visible(True)
fig.ax.yaxis.label.set_visible(True)

_offset = len(gpt2_resid)
for idx in range(num_samples):
    gpt2_x = projected[idx, 0].item()
    gpt2_y = projected[idx, 1].item()
    dpo_x = projected[_offset + idx, 0].item()
    dpo_y = projected[_offset + idx, 1].item()
    fig.ax.plot(
        [gpt2_x, dpo_x], [gpt2_y, dpo_y], color="black", ls=":", zorder=0
    )

plt.savefig(f"pca_layer{_layer}.pdf", bbox_inches="tight", dpi=1200)


# %%


fig = plt.figure(figsize=(6.75, 3.5))
gs = GridSpec(1, 3)
boundaries = [
    (0.1, 10),
    (0.1, 10),
    (0.1, 10),
]

for vec_idx in [1, 2, 3]:

    gpt2_resid = []
    dpo_resid = []
    sample_size = 50
    batch_size = 4
    print("Grabbing mlp mids...")
    _vec = vectors_of_interest[vec_idx]
    _layer = _vec[0]
    _idx = _vec[1]

    for idx in tqdm(range(0, sample_size, batch_size)):
        batch = tokenized_prompts[idx : idx + batch_size, :]
        dpo_batch = batch.clone()

        with torch.inference_mode():
            _, cache = gpt2.run_with_cache(batch)
            resid = cache[f"blocks.{_layer}.hook_resid_mid"][:, -1, :]

        gpt2_resid.extend(resid.cpu().tolist())

        with torch.inference_mode():
            _, cache = model.run_with_cache(dpo_batch)
            resid = cache[f"blocks.{_layer}.hook_resid_mid"][:, -1, :]
        dpo_resid.extend(resid.cpu().tolist())

    w_ins = [
        gpt2.blocks[_layer].mlp.W_in[:, _idx].cpu(),
        model.blocks[_layer].mlp.W_in[:, _idx].cpu(),
    ]

    gpt2_stacked = torch.stack([torch.Tensor(x) for x in gpt2_resid], dim=0)
    dpo_stacked = torch.stack([torch.Tensor(x) for x in dpo_resid], dim=0)
    gpt2_dots = einsum("sample d_model, d_model", gpt2_stacked, w_ins[0])
    dpo_dots = einsum("sample d_model, d_model", dpo_stacked, w_ins[1])
    gpt_acts = model.blocks[0].mlp.act_fn(gpt2_dots)
    dpo_acts = model.blocks[0].mlp.act_fn(dpo_dots)

    all_data = torch.concat([gpt2_stacked, dpo_stacked], dim=0)
    mean = all_data.mean(dim=0)
    stddev = all_data.std(dim=0)
    normalized = (all_data - mean) / stddev

    U, S, V = torch.pca_lowrank(normalized)

    diff = dpo_stacked - gpt2_stacked
    diff_mean = diff.mean(dim=0)

    comps = torch.concat([diff_mean.unsqueeze(-1), V], dim=1)
    comps = comps[:, :2]
    projected = torch.mm(normalized, comps)

    pca_raw = []
    num_samples = 30

    _boundary = boundaries[vec_idx - 1]
    for idx in range(num_samples):

        _activation = gpt_acts[idx].item()
        if _activation > _boundary[1]:
            act = f"High (> {_boundary[1]})"
        elif _activation > _boundary[0]:
            act = f"Low (> {_boundary[0]})"
        else:
            act = "None"

        print(_activation)
        pca_raw.append(
            {
                "Model": "GPT2",
                "x": projected[idx, 0].item(),
                "y": projected[idx, 1].item(),
                "Activated": act,
            }
        )

    _offset = len(gpt2_resid)
    print("____")
    for idx in range(num_samples):
        _activation = dpo_acts[idx].item()
        if _activation > _boundary[1]:
            act = f"High (> {_boundary[1]})"
        elif _activation > _boundary[0]:
            act = f"Low (> {_boundary[0]})"
        else:
            act = "None"
        print(_activation)
        pca_raw.append(
            {
                "Model": "DPO",
                "x": projected[_offset + idx, 0].item(),
                "y": projected[_offset + idx, 1].item(),
                "Activated": act,
            }
        )

    pca_data = pd.DataFrame(pca_raw)
    sns.set_theme(context="paper", style="ticks", rc={"lines.linewidth": 1})

    ax = fig.add_subplot(gs[0, vec_idx - 1])

    legend = None
    if vec_idx == 1:
        legend = "full"

    sns.scatterplot(
        pca_data,
        x="x",
        y="y",
        hue="Activated",
        palette={
            f"High (> {_boundary[1]})": "red",
            f"Low (> {_boundary[0]})": "orange",
            "None": "green",
        },
        hue_order=[
            f"High (> {_boundary[1]})",
            f"Low (> {_boundary[0]})",
            "None",
        ],
        style="Model",
        markers={"GPT2": "o", "DPO": "^"},
        s=60,
        legend=legend,
        ax=ax,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.xaxis.label.set_text("Shift Component")
    ax.yaxis.label.set_text("Principle Component")
    ax.xaxis.label.set_visible(True)
    ax.yaxis.label.set_visible(True)

    _offset = len(gpt2_resid)
    for idx in range(num_samples):
        gpt2_x = projected[idx, 0].item()
        gpt2_y = projected[idx, 1].item()
        dpo_x = projected[_offset + idx, 0].item()
        dpo_y = projected[_offset + idx, 1].item()
        ax.plot(
            [gpt2_x, dpo_x], [gpt2_y, dpo_y], color="black", ls=":", zorder=0
        )

plt.savefig(f"pca_layer_appx.pdf", bbox_inches="tight", dpi=1200)

