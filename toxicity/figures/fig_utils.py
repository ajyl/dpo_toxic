"""
Utility functions for figures.
"""

import torch
import torch.nn.functional as F
import einops
from transformer_lens import (
    HookedTransformer,
)


def convert(orig_state_dict, cfg):
    state_dict = {}

    state_dict["embed.W_E"] = orig_state_dict["transformer.wte.weight"]
    state_dict["pos_embed.W_pos"] = orig_state_dict["transformer.wpe.weight"]

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = orig_state_dict[
            f"transformer.h.{l}.ln_1.weight"
        ]
        state_dict[f"blocks.{l}.ln1.b"] = orig_state_dict[
            f"transformer.h.{l}.ln_1.bias"
        ]

        # In GPT-2, q,k,v are produced by one big linear map, whose output is
        # concat([q, k, v])
        W = orig_state_dict[f"transformer.h.{l}.attn.c_attn.weight"]
        W_Q, W_K, W_V = torch.tensor_split(W, 3, dim=1)
        W_Q = einops.rearrange(W_Q, "m (i h)->i m h", i=cfg.n_heads)
        W_K = einops.rearrange(W_K, "m (i h)->i m h", i=cfg.n_heads)
        W_V = einops.rearrange(W_V, "m (i h)->i m h", i=cfg.n_heads)

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        qkv_bias = orig_state_dict[f"transformer.h.{l}.attn.c_attn.bias"]
        qkv_bias = einops.rearrange(
            qkv_bias,
            "(qkv index head)->qkv index head",
            qkv=3,
            index=cfg.n_heads,
            head=cfg.d_head,
        )
        state_dict[f"blocks.{l}.attn.b_Q"] = qkv_bias[0]
        state_dict[f"blocks.{l}.attn.b_K"] = qkv_bias[1]
        state_dict[f"blocks.{l}.attn.b_V"] = qkv_bias[2]

        W_O = orig_state_dict[f"transformer.h.{l}.attn.c_proj.weight"]
        W_O = einops.rearrange(W_O, "(i h) m->i h m", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = orig_state_dict[
            f"transformer.h.{l}.attn.c_proj.bias"
        ]

        state_dict[f"blocks.{l}.ln2.w"] = orig_state_dict[
            f"transformer.h.{l}.ln_2.weight"
        ]
        state_dict[f"blocks.{l}.ln2.b"] = orig_state_dict[
            f"transformer.h.{l}.ln_2.bias"
        ]

        W_in = orig_state_dict[f"transformer.h.{l}.mlp.c_fc.weight"]
        state_dict[f"blocks.{l}.mlp.W_in"] = W_in
        state_dict[f"blocks.{l}.mlp.b_in"] = orig_state_dict[
            f"transformer.h.{l}.mlp.c_fc.bias"
        ]

        W_out = orig_state_dict[f"transformer.h.{l}.mlp.c_proj.weight"]
        state_dict[f"blocks.{l}.mlp.W_out"] = W_out
        state_dict[f"blocks.{l}.mlp.b_out"] = orig_state_dict[
            f"transformer.h.{l}.mlp.c_proj.bias"
        ]
    state_dict["unembed.W_U"] = orig_state_dict["lm_head.weight"].T

    state_dict["ln_final.w"] = orig_state_dict["transformer.ln_f.weight"]
    state_dict["ln_final.b"] = orig_state_dict["transformer.ln_f.bias"]
    return state_dict


def load_hooked(model_name, weights_path):
    _model = HookedTransformer.from_pretrained(model_name)
    cfg = _model.cfg

    _weights = torch.load(weights_path, map_location=torch.device("cuda"))[
        "state"
    ]
    weights = convert(_weights, cfg)
    model = HookedTransformer(cfg)
    model.load_and_process_state_dict(weights)
    model.tokenizer.padding_side = "left"
    model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
    return model


def get_svd(_model, toxic_vector, num_mlp_vecs):
    scores = []
    for layer in range(_model.cfg.n_layers):
        mlp_outs = _model.blocks[layer].mlp.W_out
        cos_sims = F.cosine_similarity(
            mlp_outs, toxic_vector.unsqueeze(0), dim=1
        )
        _topk = cos_sims.topk(k=300)
        _values = [x.item() for x in _topk.values]
        _idxs = [x.item() for x in _topk.indices]
        topk = list(zip(_values, _idxs, [layer] * _topk.indices.shape[0]))
        scores.extend(topk)

    sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
    top_vecs = [
        _model.blocks[x[2]].mlp.W_out[x[1]]
        for x in sorted_scores[:num_mlp_vecs]
    ]
    top_vecs = [x / x.norm() for x in top_vecs]
    _top_vecs = torch.stack(top_vecs)

    svd = torch.linalg.svd(_top_vecs.transpose(0, 1))
    return svd, sorted_scores
