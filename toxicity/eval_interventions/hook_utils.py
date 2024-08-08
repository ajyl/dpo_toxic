"""
Utility functions for hooking.
"""

from functools import partial
import torch
import torch.nn.functional as F


def rank_value_vecs(model, toxic_vector):
    """ 
    Rank all value vectors based on similarity vs. toxic_vector.
    toxic_vector: [d_model]
    """
    scores = []
    for layer in range(model.config.n_layer):
        # mlp_outs = model.blocks[layer].mlp.W_out
        # [d_mlp, d_model]
        mlp_outs = model.transformer.h[layer].mlp.c_proj.weight
        cos_sims = F.cosine_similarity(mlp_outs, toxic_vector.unsqueeze(0), dim=1)
        _topk = cos_sims.topk(k=100)
        _values = [x.item() for x in _topk.values]
        _idxs = [x.item() for x in _topk.indices]
        topk = list(zip(_values, _idxs, [layer] * _topk.indices.shape[0]))
        scores.extend(topk)

    sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
    return sorted_scores


def get_svd_u_vec(model, toxic_vector, topk_sorted_score, U_idx):
    """
    Get the svd U vector
    toxic_vector: toxic_vector [d_model]
    topk_sorted_score: (int) vectors we want to get
    U_idx: Index of u vector.
    """
    sorted_scores = rank_value_vecs(model, toxic_vector)
    top_vecs = [
        # model.blocks[x[2]].mlp.W_out[x[1]]
        model.transformer.h[x[2]].mlp.c_proj.weight[x[1]]
        for x in sorted_scores[:topk_sorted_score]
    ]
    top_vecs = [x / x.norm() for x in top_vecs]
    _top_vecs = torch.stack(top_vecs)

    svd = torch.linalg.svd(_top_vecs.transpose(0, 1))
    svd_U = svd.U.transpose(0, 1)
    return svd_U[U_idx]


def get_intervene_vector(model, config):
    """
    Get vector according to specifications in :config:
    """

    def _get_mlp_w_out(_config):
        layer = _config["layer"]
        idx = _config["idx"]
        return model.transformer.h[layer].mlp.c_proj.weight[idx]

    def _get_mlp_w_in(_config):
        w_in_idx = _config["w_ins"][0]
        layer = w_in_idx[0]
        idx = w_in_idx[1]
        return model.transformer.h[layer].mlp.c_fc.weight[:, idx]

    def _get_toxic_probe(_config):
        return torch.load(_config["datapath"])

    def _get_svd(_config):
        topk_sorted_score = _config["topk_sorted_score"]
        u_idx = _config["idx"]
        toxic_vector = torch.load(_config["datapath"])
        return get_svd_u_vec(model, toxic_vector, topk_sorted_score, u_idx)

    def _get_random(_config):
        shape = model.transformer.h[0].mlp.c_proj.weight[0].shape
        device = model.device
        return torch.rand(shape).to(device)

    return {
        "mlp_w_out": _get_mlp_w_out,
        "mlp_w_in": _get_mlp_w_in,
        "toxic_probe": _get_toxic_probe,
        "svd": _get_svd,
        "random": _get_random,
    }[config["type"]](config)


def hook_subtract(model, config):
    intervene_vector = get_intervene_vector(model, config)
    scale = config["scale"]
    subtract_from = config["subtract_from"]
    hook_timesteps = config["hook_timesteps"]

    def patch(vec, _scale):
        def hook(module, input, output):

            _vec = vec.unsqueeze(0).unsqueeze(0)
            if hook_timesteps == -1:
                _vec = _vec.repeat(output.shape[0], 1, 1)
            else:
                _vec = _vec.repeat(output.shape[0], output.shape[1], 1)
            output[:, hook_timesteps:, :] = output[:, hook_timesteps:, :] - (
                _scale * _vec
            )
            return output

        return hook

    hooks = []
    for layer in subtract_from:
        # hook = model.transformer.h[layer].mlp.c_proj.register_forward_hook(
        hook = model.transformer.h[layer].mlp.register_forward_hook(
            patch(intervene_vector, scale)
        )
        hooks.append(hook)
    return model, hooks


def dont_hook(model, config):
    return model, []
