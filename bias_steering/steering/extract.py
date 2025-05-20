import os, logging
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from torchtyping import TensorType

from . import ModelBase
from .steering_utils import get_all_layer_activations
from ..config import Config


def mean_diff(pos_acts: TensorType[-1, -1], neg_acts: TensorType[-1, -1]) -> TensorType[-1]:
    """Mean Activation Difference"""
    return pos_acts.mean(dim=0) - neg_acts.mean(dim=0)


def weighted_mean(acts, weights):
    """Weighted Mean"""
    w = weights / weights.sum()
    return (acts * w.unsqueeze(-1)).sum(dim=0)

def get_activations(model, examples: pd.DataFrame, save_dir: Path, label: str, batch_size: int, use_cache=False, output_prefix=True):
    activation_path = save_dir / f"{label}.pt"
    if use_cache and activation_path.exists():
        acts = torch.load(activation_path)
    else:
        if output_prefix:
            prompts = model.apply_chat_template(examples["prompt"].tolist(), output_prefix=examples["output_prefix"].tolist())
        else:
            prompts = model.apply_chat_template(examples["prompt"].tolist())
        acts = get_all_layer_activations(model, prompts, batch_size).to(torch.float64)
        torch.save(acts, activation_path)
    return acts


def extract_candidate_vectors(
    cfg: Config, 
    model: ModelBase, 
    pos_examples: pd.DataFrame, 
    neg_examples: pd.DataFrame, 
    neutral_examples= None,
):  
    save_dir = cfg.artifact_path() / "activations"
    os.makedirs(save_dir, exist_ok=True)

    pos_acts = get_activations(model, pos_examples, save_dir, "positive", cfg.batch_size, cfg.use_cache, cfg.data_cfg.output_prefix)
    neg_acts = get_activations(model, neg_examples, save_dir, "negative", cfg.batch_size, cfg.use_cache, cfg.data_cfg.output_prefix)

    if neutral_examples is not None:
        neutral_acts = get_activations(model, neutral_examples, save_dir, label="neutral", batch_size=cfg.batch_size, use_cache=cfg.use_cache, output_prefix=cfg.data_cfg.output_prefix)

    if cfg.method == "WMD":
        pos_weights = torch.Tensor(pos_examples.bias.tolist())
        neg_weights = torch.Tensor(neg_examples.bias.tolist())

    extracted_vectors = []
        
    for layer in range(model.n_layer):
        pos = pos_acts[layer]
        neg = neg_acts[layer]

        if neutral_examples is not None:
            offset = neutral_acts[layer].mean(dim=0)
            pos -= offset
            neg -= offset

        if cfg.method == "WMD":
            pos_mean = weighted_mean(pos, pos_weights)
            neg_mean = weighted_mean(neg, neg_weights)
            vec = F.normalize(pos_mean, dim=-1) - F.normalize(neg_mean, dim=-1)
        else:
            vec = mean_diff(pos, neg)

        extracted_vectors.append(vec)

    os.makedirs(cfg.artifact_path() / "activations", exist_ok=True)
    filepath = cfg.artifact_path() / "activations" / "candidate_vectors.pt"
    torch.save(torch.vstack(extracted_vectors), filepath)
    logging.info(f"Candidate vectors saved to: {filepath}")
    