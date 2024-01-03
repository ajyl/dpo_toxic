"""
Intervention functionalities.
"""
from tqdm import tqdm
import torch
from utils import verbose_print, VERBOSE


def get_prompts(model, data, config):
    """
    Dummy intervention.
    """
    return {
        "pred_tokens": data["prompt_input_ids"],
        "pred_text": data["prompts"],
    }


def get_gold(model, data, config):
    """
    Dummy intervention.
    """
    return {
        "pred_tokens": data["gold_input_ids"],
        "pred_text": data["gold"],
    }


def generate_default(model, data, config):
    """
    Do not intervene.
    """
    batch_size = config["batch_size"]
    pad_token_id = model.tokenizer.pad_token_id
    all_output = []
    all_output_text = []
    for idx in tqdm(range(0, data["prompt_input_ids"].shape[0], batch_size)):
        batch = data["prompt_input_ids"][idx : idx + batch_size]
        with torch.inference_mode():
            output = model.generate(
                batch.to("cuda"),
                max_new_tokens=config["max_new_tokens"],
                do_sample=False,
                pad_token_id=pad_token_id,
            )

            if VERBOSE:
                _output = model.forward(batch.to("cuda"))
                logits = _output.logits
                topk = logits.topk(k=5).indices
                verbose_print(model.tokenizer.batch_decode(topk[:, -1, :]))

        output_text = model.tokenizer.batch_decode(
            output, skip_special_tokens=True
        )
        all_output.extend(output)
        all_output_text.extend(output_text)
    return {
        "pred_tokens": torch.stack(all_output, dim=0),
        "pred_text": all_output_text,
    }
