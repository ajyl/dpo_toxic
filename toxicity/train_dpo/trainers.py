"""
Train loop for DPO.
"""
from typing import Optional, Dict, List, Union, Tuple

import random
import os
from collections import defaultdict
import time
import json
import functools
import contextlib
from collections import Counter

import numpy as np
import wandb
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import KLDivLoss
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.api import (
    FullStateDictConfig,
    FullOptimStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import transformers
from omegaconf import DictConfig

from toxicity.train_dpo.pplm_dataset import get_pplm_batch_iterator
from toxicity.train_dpo.dpo_utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
    get_local_dir,
)
from constants import GPT2_PAD_IDX

torch.backends.cuda.matmul.allow_tf32 = True


def generate(
    model,
    batch,
    max_new_tokens,
    pad_token_id,
    include_ngram_blocked=False,
    include_ref=False,
    fsdp=False,
    ref_model=None,
):
    """
    Return greedy and n-gram blocked generations.
    """
    prompt_shape = batch["prompt_input_ids"].shape[1]
    with torch.no_grad():
        # FSDP generation according to https://github.com/pytorch/pytorch/issues/100069
        ctx = lambda: (
            FSDP.summon_full_params(model, writeback=False, recurse=False)
            if fsdp
            else contextlib.nullcontext()
        )
        with ctx():
            greedy_resp = model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )

        greedy_resp_labels = greedy_resp.detach().clone()
        greedy_resp_labels[:, :prompt_shape] = -100
        output = {
            "policy_input_ids": greedy_resp,
            "policy_attention_mask": greedy_resp != GPT2_PAD_IDX,
            "policy_labels": greedy_resp_labels,
        }

    return output


def dpo_loss(
    policy_pos_logps: torch.FloatTensor,
    policy_neg_logps: torch.FloatTensor,
    ref_pos_logps: torch.FloatTensor,
    ref_neg_logps: torch.FloatTensor,
    beta: float,
    reference_free: bool = False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """
    Compute the DPO loss for a batch of policy and reference model log probabilities.

    :params:

    :policy_pos_logps: logprobs of positive responses from policy model: (batch_size,)
    :policy_neg_logps: logprobs of negative responses from policy model: (batch_size,)
    :ref_pos_logps: logprobs of positive responses from reference model: (batch_size,)
    :ref_neg_logps: logprobs of negative responses from reference model: (batch_size,)
    :beta: Temperature parameter for the DPO loss, typically something
        in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
    :reference_free: If True, we ignore the _provided_ reference model and
        implicitly use a reference model that assigns equal probability to all responses.

    :returns:

    A tuple of three tensors: (losses, pos_rewards, neg_rewards).
    The losses tensor contains the DPO loss for each example in the batch.
    The pos_rewards and neg_rewards tensors contain the rewards for the
        positive and neg responses, respectively.
    """
    pi_logratios = policy_pos_logps - policy_neg_logps
    ref_logratios = ref_pos_logps - ref_neg_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits)
    pos_rewards = beta * (policy_pos_logps - ref_pos_logps).detach()
    neg_rewards = beta * (policy_neg_logps - ref_neg_logps).detach()

    return losses, pos_rewards, neg_rewards


def get_kl_div(
    kl_criterion: KLDivLoss,
    pos_pi_logits: torch.FloatTensor,  # [batch, seq, vocab]
    neg_pi_logits: torch.FloatTensor,  # [batch, seq, vocab]
    pos_ref_logits: torch.FloatTensor,  # [batch, seq, vocab]
    neg_ref_logits: torch.FloatTensor,  # [batch, seq, vocab]
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """
    Return KL Loss.
    """
    # [batch, seq, vocab] --> [batch]
    pos_kl_div = (
        kl_criterion(
            F.log_softmax(pos_pi_logits, dim=-1),
            F.log_softmax(pos_ref_logits, dim=-1),
        )
        .sum(dim=-1)
        .mean(dim=-1)
    )
    neg_kl_div = (
        kl_criterion(
            F.log_softmax(neg_pi_logits, dim=-1),
            F.log_softmax(neg_ref_logits, dim=-1),
        )
        .sum(dim=-1)
        .mean(dim=-1)
    )
    return pos_kl_div, neg_kl_div


def get_batch_logps(
    logits: torch.FloatTensor,
    input_ids: torch.FloatTensor,
    average_log_prob: bool = False,
) -> torch.FloatTensor:
    """
    Compute the log probabilities of the given labels under the given logits.

    :params:

    :logits: Logits of the model (unnormalized). (batch, seq, vocab)
    :labels: Labels for which to compute the log probabilities.
        Label tokens with a value of -100 are ignored. (batch, seq)
    :average_log_prob: If True, return the average log probability per
        (non-masked) token. Otherwise, return the sum of the log probabilities
        of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log
        probabilities of the given labels under the given logits.
    """
    # [batch, seq]
    labels = input_ids[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != GPT2_PAD_IDX

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == GPT2_PAD_IDX] = 0

    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def concatenated_inputs(
    batch: Dict[str, Union[List, torch.LongTensor]]
) -> Dict[str, torch.LongTensor]:
    """
    Concatenate the positive and negative inputs into a single tensor.

    :params:

    :batch: A batch of data. Must contain the keys 'pos_input_ids' and
        'neg_input_ids', which are tensors of shape (batch, seq).

    :returns:
        A dictionary containing the concatenated inputs under the key
        'concatenated_input_ids'.
    """
    max_length = max(
        batch["pos_input_ids"].shape[1],
        batch["neg_input_ids"].shape[1],
    )
    concatenated_batch = {}
    for k in batch:
        if k.startswith("pos_") and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if "labels" in k else 0
            concatenated_key = k.replace("pos", "concatenated")
            concatenated_batch[concatenated_key] = pad_to_length(
                batch[k], max_length, pad_value=pad_value
            )
    for k in batch:
        if k.startswith("neg_") and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if "labels" in k else 0
            concatenated_key = k.replace("neg", "concatenated")
            concatenated_batch[concatenated_key] = torch.cat(
                (
                    concatenated_batch[concatenated_key],
                    pad_to_length(batch[k], max_length, pad_value=pad_value),
                ),
                dim=0,
            )
    return concatenated_batch


class BasicTrainer(object):
    def __init__(
        self,
        policy: nn.Module,
        config: DictConfig,
        seed: int,
        run_dir: str,
        reference_model: Optional[nn.Module] = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        """
        A trainer for a language model, supporting either SFT or DPO training.

        If multiple GPUs are present, naively splits the model across them, effectively
        offering N times available memory, but without any parallel computation.
        """
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir
        self.example_counter = 0
        self.batch_counter = 0
        self.last_log = None
        self.patience = 0
        self.val_metric_value = -1
        if config.validation_direction == "max":
            self.val_direction = 1
            self.best_val_metric = -1

        else:
            self.val_direction = -1
            self.best_val_metric = 1e10

        tokenizer_name_or_path = (
            config.model.tokenizer_name_or_path or config.model.name_or_path
        )
        rank0_print(f"Loading tokenizer {tokenizer_name_or_path}")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, cache_dir=get_local_dir(config.local_dirs)
        )
        if tokenizer_name_or_path.startswith("gpt2"):
            self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.policy = policy
        self.reference_model = reference_model
        self.kl_criterion = KLDivLoss(reduction="none", log_target=True)

        self.train_iterator = get_pplm_batch_iterator(
            self.tokenizer,
            self.config,
            split="train",
        )
        self.eval_iterator = get_pplm_batch_iterator(
            self.tokenizer,
            self.config,
            split="valid",
        )
        self.eval_batches = list(self.eval_iterator)
        rank0_print(
            f"Loaded {len(self.eval_batches)} eval batches of size {config.eval_batch_size}"
        )

    def get_batch_samples(
        self, batch: Dict[str, torch.LongTensor]
    ) -> Tuple[str, str]:
        """
        Generate samples from the policy (and reference model, if doing DPO training)
        for the given batch of inputs
        """

        # FSDP generation according to https://github.com/pytorch/pytorch/issues/100069
        ctx = lambda: (
            FSDP.summon_full_params(
                self.policy, writeback=False, recurse=False
            )
            if "FSDP" in self.config.trainer
            else contextlib.nullcontext()
        )
        with ctx():
            policy_output = self.policy.generate(
                batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.config.max_length,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        if self.config.loss.name == "dpo":
            ctx = lambda: (
                FSDP.summon_full_params(
                    self.reference_model, writeback=False, recurse=False
                )
                if "FSDP" in self.config.trainer
                else contextlib.nullcontext()
            )
            with ctx():
                reference_output = self.reference_model.generate(
                    batch["prompt_input_ids"],
                    attention_mask=batch["prompt_attention_mask"],
                    max_length=self.config.max_length,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

        policy_output = pad_to_length(
            policy_output, self.config.max_length, self.tokenizer.pad_token_id
        )
        policy_output = all_gather_if_needed(
            policy_output, self.rank, self.world_size
        )
        policy_output_decoded = self.tokenizer.batch_decode(
            policy_output, skip_special_tokens=True
        )

        reference_output_decoded = []
        if self.config.loss.name == "dpo":
            reference_output = pad_to_length(
                reference_output,
                self.config.max_length,
                self.tokenizer.pad_token_id,
            )
            reference_output = all_gather_if_needed(
                reference_output, self.rank, self.world_size
            )
            reference_output_decoded = self.tokenizer.batch_decode(
                reference_output, skip_special_tokens=True
            )

        return policy_output_decoded, reference_output_decoded

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Run the given model on the given batch of inputs,
        concatenating the positive and negative inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.

        :returns:
        :pos_logps: (batch)
        :neg_logps: (batch)
        :pos_logits: (batch, seq, vocab)
        :neg_logits: (batch, seq, vocab)
        """
        concatenated_batch = concatenated_inputs(batch)

        # [batch (*2), seq (prompt + response), vocab]
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
        ).logits.to(torch.float32)
        all_logps = get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_input_ids"],
            average_log_prob=False,
        )

        num_pos_samples = batch["pos_input_ids"].shape[0]
        pos_logps = all_logps[:num_pos_samples]
        neg_logps = all_logps[num_pos_samples:]
        pos_logits = all_logits[:num_pos_samples]
        neg_logits = all_logits[num_pos_samples:]
        return pos_logps, neg_logps, pos_logits, neg_logits

    def get_batch_metrics(
        self,
        batch: Dict[str, Union[List, torch.LongTensor]],
        loss_config: DictConfig,
        train=True,
    ):
        """
        Compute the SFT or DPO loss and other metrics for the given batch of inputs.
        """

        metrics = {}
        train_test = "train" if train else "valid"
        kl_loss = None

        if loss_config.name == "dpo":
            (
                policy_pos_logps,
                policy_neg_logps,
                policy_pos_logits,
                policy_neg_logits,
            ) = self.concatenated_forward(self.policy, batch)
            with torch.no_grad():
                (
                    ref_pos_logps,
                    ref_neg_logps,
                    ref_pos_logits,
                    ref_neg_logits,
                ) = self.concatenated_forward(self.reference_model, batch)
            losses, pos_rewards, neg_rewards = dpo_loss(
                policy_pos_logps,
                policy_neg_logps,
                ref_pos_logps,
                ref_neg_logps,
                beta=loss_config.beta,
                reference_free=loss_config.reference_free,
            )

            pos_kl_div, neg_kl_div = get_kl_div(
                self.kl_criterion,
                policy_pos_logits,
                policy_neg_logits,
                ref_pos_logits,
                ref_neg_logits,
            )

            reward_accuracies = (pos_rewards > neg_rewards).float()

            pos_rewards = all_gather_if_needed(
                pos_rewards, self.rank, self.world_size
            )
            neg_rewards = all_gather_if_needed(
                neg_rewards, self.rank, self.world_size
            )
            reward_accuracies = all_gather_if_needed(
                reward_accuracies, self.rank, self.world_size
            )

            metrics[f"rewards_{train_test}/positive"] = (
                pos_rewards.cpu().numpy().tolist()
            )
            metrics[f"rewards_{train_test}/negative"] = (
                neg_rewards.cpu().numpy().tolist()
            )
            metrics[f"rewards_{train_test}/accuracies"] = (
                reward_accuracies.cpu().numpy().tolist()
            )
            metrics[f"rewards_{train_test}/margins"] = (
                (pos_rewards - neg_rewards).cpu().numpy().tolist()
            )

            policy_neg_logps = all_gather_if_needed(
                policy_neg_logps.detach(), self.rank, self.world_size
            )
            metrics[f"logps_{train_test}/negative"] = (
                policy_neg_logps.cpu().numpy().tolist()
            )

            metrics[f"kl_div_{train_test}/positive"] = (
                pos_kl_div.detach().cpu().numpy().tolist()
            )

            metrics[f"kl_div_{train_test}/negative"] = (
                neg_kl_div.detach().cpu().numpy().tolist()
            )

        elif loss_config.name == "sft":
            policy_pos_logits = self.policy(
                batch["pos_input_ids"],
                attention_mask=batch["pos_attention_mask"],
            ).logits.to(torch.float32)
            policy_pos_logps = get_batch_logps(
                policy_pos_logits,
                batch["pos_labels"],
                average_log_prob=False,
            )

            losses = -policy_pos_logps

        policy_pos_logps = all_gather_if_needed(
            policy_pos_logps.detach(), self.rank, self.world_size
        )
        metrics[f"logps_{train_test}/positive"] = (
            policy_pos_logps.cpu().numpy().tolist()
        )

        all_devices_losses = all_gather_if_needed(
            losses.detach(), self.rank, self.world_size
        )
        metrics[f"loss/{train_test}"] = (
            all_devices_losses.cpu().numpy().tolist()
        )

        return losses.mean(), metrics

    def train_loop(self):
        """Begin either SFT or DPO training, with periodic evaluation."""

        rank0_print(f"Using {self.config.optimizer} optimizer")
        self.optimizer = getattr(torch.optim, self.config.optimizer)(
            self.policy.parameters(), lr=self.config.lr
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(
                1.0, (step + 1) / (self.config.warmup_steps + 1)
            ),
        )

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.config.loss.name == "dpo":
            self.reference_model.eval()

        for batch in self.train_iterator:
            if self.example_counter % self.config.eval_every == 0 and (
                self.example_counter > 0 or self.config.do_first_eval
            ):
                result = self.eval()
                if result == -1:
                    return

            self.train(batch)

    def train(self, batch):
        """
        Run single train step.
        """
        self.policy.train()

        start_time = time.time()
        batch_metrics = defaultdict(list)
        for microbatch_idx in range(self.config.gradient_accumulation_steps):
            # batch:
            # {
            #   "pos_input_ids": Tensor[batch, seq],
            #   "pos_attention_mask": Tensor[batch, seq],
            #   "neg_input_ids": Tensor[batch, seq],
            #   "neg_attention_mask": Tensor[batch, seq],
            # }
            self.policy.train()
            global_microbatch = slice_and_move_batch_for_device(
                batch,
                microbatch_idx,
                self.config.gradient_accumulation_steps,
                self.rank,
            )
            local_microbatch = slice_and_move_batch_for_device(
                global_microbatch, self.rank, self.world_size, self.rank
            )
            loss, metrics = self.get_batch_metrics(
                local_microbatch, self.config.loss, train=True
            )
            (loss / self.config.gradient_accumulation_steps).backward()

            for k, v in metrics.items():
                batch_metrics[k].extend(v)

        grad_norm = self.clip_gradient()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        step_time = time.time() - start_time
        examples_per_second = self.config.batch_size / step_time
        batch_metrics["examples_per_second"].append(examples_per_second)
        batch_metrics["grad_norm"].append(grad_norm)

        self.batch_counter += 1
        self.example_counter += self.config.batch_size

        if (
            self.last_log is None
            or time.time() - self.last_log
            > self.config.minimum_log_interval_secs
        ):
            mean_train_metrics = {
                k: sum(v) / len(v) for k, v in batch_metrics.items()
            }
            mean_train_metrics["counters/examples"] = self.example_counter
            mean_train_metrics["counters/updates"] = self.batch_counter
            rank0_print(
                f"train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}"
            )

            if self.config.wandb.enabled and self.rank == 0:
                wandb.log(mean_train_metrics, step=self.example_counter)

            self.last_log = time.time()

    def eval(self):
        """
        Run evaluation.
        """
        rank0_print(
            f"Running evaluation after {self.example_counter} train examples"
        )
        self.policy.eval()

        all_eval_metrics = defaultdict(list)
        if self.config.sample_during_eval:
            all_policy_samples, all_reference_samples = [], []

        for eval_batch in (
            tqdm(self.eval_batches, desc="Computing eval metrics")
            if self.rank == 0
            else self.eval_batches
        ):

            local_eval_batch = slice_and_move_batch_for_device(
                eval_batch, self.rank, self.world_size, self.rank
            )
            with torch.no_grad():
                _, eval_metrics = self.get_batch_metrics(
                    local_eval_batch, self.config.loss, train=False
                )

            for k, v in eval_metrics.items():
                all_eval_metrics[k].extend(v)

        if (
            self.config.sample_during_eval
            and self.example_counter % self.config.sample_every == 0
        ):
            if self.config.n_eval_model_samples < self.config.eval_batch_size:
                rank0_print(
                    f"Warning: n_eval_model_samples ({self.config.n_eval_model_samples}) < \
                    eval_batch_size ({self.config.eval_batch_size}). \
                    Sampling from the first complete eval batch of prompts."
                )
                sample_batches = self.eval_batches[:1]
            else:
                n_sample_batches = (
                    self.config.n_eval_model_samples
                    // self.config.eval_batch_size
                )
                sample_batches = self.eval_batches[:n_sample_batches]

            for eval_batch in (
                tqdm(sample_batches, desc="Generating samples...")
                if self.rank == 0
                else sample_batches
            ):
                local_eval_batch = slice_and_move_batch_for_device(
                    eval_batch, self.rank, self.world_size, self.rank
                )
                (
                    policy_samples,
                    reference_samples,
                ) = self.get_batch_samples(local_eval_batch)

                all_policy_samples.extend(policy_samples)
                all_reference_samples.extend(reference_samples)

            rank0_print("Policy samples:")
            rank0_print(json.dumps(all_policy_samples[:10], indent=2))

        mean_eval_metrics = {
            k: sum(v) / len(v) for k, v in all_eval_metrics.items()
        }
        self.val_metric_value = mean_eval_metrics[
            self.config.validation_metric
        ]

        rank0_print(
            f"eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}"
        )

        if self.config.wandb.enabled and self.rank == 0:
            wandb.log(mean_eval_metrics, step=self.example_counter)

        if self.example_counter == 0:
            return 0

        if (
            self.val_metric_value is not None
            and self.val_metric_value * self.val_direction
            > self.val_direction * self.best_val_metric
        ):
            self.best_val_metric = self.val_metric_value

            rank0_print(
                f"\n=====\nNew best for {self.config.validation_metric}: {self.best_val_metric}.\n=====\n"
            )
            self.patience = 0

            if self.example_counter % self.config.save_every == 0:
                if self.config.debug:
                    rank0_print("skipping save in debug mode")
                else:
                    output_dir = os.path.join(self.run_dir, "checkpoints")
                    rank0_print(
                        f"Creating checkpoint to write to {output_dir}..."
                    )
                    self.save(output_dir, mean_eval_metrics)
        else:
            self.patience += 1
            if self.patience >= self.config.validation_patience:
                rank0_print("Ran out of patience, stopping training...")
                return -1

        return 0

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of a non-FSDP policy."""
        return torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.config.max_grad_norm
        ).item()

    def write_state_dict(
        self,
        step: int,
        state: Dict[str, torch.Tensor],
        metrics: Dict,
        filename: str,
        dir_name: Optional[str] = None,
    ):
        """Write a checkpoint to disk."""
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, f"LATEST")

        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        rank0_print(f"writing checkpoint to {output_path}...")
        torch.save(
            {
                "step_idx": step,
                "state": state,
                "metrics": metrics if metrics is not None else {},
            },
            output_path,
        )

    def save(
        self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None
    ):
        """Save policy, optimizer, and scheduler state to disk."""

        policy_state_dict = self.policy.state_dict()
        self.write_state_dict(
            self.example_counter,
            policy_state_dict,
            metrics,
            "policy.pt",
            output_dir,
        )
        del policy_state_dict

        optimizer_state_dict = self.optimizer.state_dict()
        self.write_state_dict(
            self.example_counter,
            optimizer_state_dict,
            metrics,
            "optimizer.pt",
            output_dir,
        )
        del optimizer_state_dict

        scheduler_state_dict = self.scheduler.state_dict()
        self.write_state_dict(
            self.example_counter,
            scheduler_state_dict,
            metrics,
            "scheduler.pt",
            output_dir,
        )


class FSDPTrainer(BasicTrainer):
    def __init__(
        self,
        policy: nn.Module,
        config: DictConfig,
        seed: int,
        run_dir: str,
        reference_model: Optional[nn.Module] = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        """A trainer subclass that uses PyTorch FSDP to shard the model across multiple GPUs.

        This trainer will shard both the policy and reference model across all available GPUs.
        Models are sharded at the block level, where the block class name is provided in the config.
        """

        super().__init__(
            policy, config, seed, run_dir, reference_model, rank, world_size
        )
        assert (
            config.model.block_name is not None
        ), "must specify model.block_name (e.g., GPT2Block or GPTNeoXLayer) for FSDP"

        wrap_class = get_block_class_from_model(
            policy, config.model.block_name
        )
        model_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={wrap_class},
        )

        shared_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=False,
            sync_module_states=False,
        )

        rank0_print("Sharding policy...")
        mp_dtype = (
            getattr(torch, config.model.fsdp_policy_mp)
            if config.model.fsdp_policy_mp is not None
            else None
        )
        policy_mp_policy = MixedPrecision(
            param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype
        )
        self.policy = FSDP(
            policy, **shared_fsdp_kwargs, mixed_precision=policy_mp_policy
        )

        if config.activation_checkpointing:
            rank0_print("Attempting to enable activation checkpointing...")
            try:
                # use activation checkpointing, according to:
                # https://pytorch.org/blog/scaling-multimodal-foundation-models-in-torchmultimodal-with-pytorch-distributed/
                #
                # first, verify we have FSDP activation support ready by importing:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    checkpoint_wrapper,
                    apply_activation_checkpointing,
                    CheckpointImpl,
                )

                non_reentrant_wrapper = functools.partial(
                    checkpoint_wrapper,
                    offload_to_cpu=False,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )
            except Exception as e:
                rank0_print("FSDP activation checkpointing not available:", e)
            else:
                check_fn = lambda submodule: isinstance(submodule, wrap_class)
                rank0_print(
                    "Applying activation checkpointing wrapper to policy..."
                )
                apply_activation_checkpointing(
                    self.policy,
                    checkpoint_wrapper_fn=non_reentrant_wrapper,
                    check_fn=check_fn,
                )
                rank0_print("FSDP activation checkpointing enabled!")

        if config.loss.name == "dpo":
            rank0_print("Sharding reference model...")
            self.reference_model = FSDP(reference_model, **shared_fsdp_kwargs)

        print("Loaded model on rank", rank)
        dist.barrier()

    def clip_gradient(self):
        """
        Clip the gradient norm of the parameters of an FSDP policy,
        gathering the gradients across all GPUs.
        """
        return self.policy.clip_grad_norm_(self.config.max_grad_norm).item()

    def save(self, output_dir=None, metrics=None):
        """
        Save policy, optimizer, and scheduler state to disk,
        gathering from all processes and saving only on the rank 0 process.
        """
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            self.policy,
            StateDictType.FULL_STATE_DICT,
            state_dict_config=save_policy,
        ):
            policy_state_dict = self.policy.state_dict()

        if self.rank == 0:
            self.write_state_dict(
                self.example_counter,
                policy_state_dict,
                metrics,
                "policy.pt",
                output_dir,
            )
        del policy_state_dict
        dist.barrier()

        save_policy = FullOptimStateDictConfig(
            offload_to_cpu=True, rank0_only=True
        )
        with FSDP.state_dict_type(
            self.policy,
            StateDictType.FULL_STATE_DICT,
            optim_state_dict_config=save_policy,
        ):
            optimizer_state_dict = FSDP.optim_state_dict(
                self.policy, self.optimizer
            )

        if self.rank == 0:
            self.write_state_dict(
                self.example_counter,
                optimizer_state_dict,
                metrics,
                "optimizer.pt",
                output_dir,
            )
        del optimizer_state_dict
        dist.barrier()

        if self.rank == 0:
            scheduler_state_dict = self.scheduler.state_dict()
            self.write_state_dict(
                self.example_counter,
                scheduler_state_dict,
                metrics,
                "scheduler.pt",
                output_dir,
            )
        dist.barrier()
