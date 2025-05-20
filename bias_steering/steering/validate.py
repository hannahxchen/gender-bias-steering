import os
import logging
from pathlib import Path
from typing import List, Callable, Dict
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch
import torch.nn.functional as F
from torchtyping import TensorType

from . import ModelBase
from .intervention import get_intervention_func
from .steering_utils import get_all_layer_activations, scalar_projection
from ..config import Config
from ..utils import RMS, RMSE, save_to_json_file
from ..data.prompt_iterator import PromptIterator


def evaluate_candidate_vectors(
    model: ModelBase, prompts: List[str],
    candidate_vectors: TensorType["n_layer", "hidden_size"], 
    bias_scores: np.ndarray, save_dir: Path, 
    filter_layer_pct: float = 0.05, batch_size: int = 32,
) -> List[Dict]:
    os.makedirs(save_dir, exist_ok=True)

    results, projections = [], []
    prompt_acts = get_all_layer_activations(model, prompts, batch_size)

    for layer in range(model.n_layer):
        vec = candidate_vectors[layer]
        acts = prompt_acts[layer]
        projs = scalar_projection(acts, vec).numpy()

        r = pearsonr(projs, bias_scores)
        rmse = RMSE(projs, bias_scores)

        projections.append(projs.tolist())
        results.append({
            "layer": layer, 
            "corr": r.statistic, 
            "p_val": r.pvalue,
            "RMSE": rmse
        })

    np.save(save_dir / "projections.npy", np.array(projections))
    save_to_json_file(results, save_dir / "proj_correlation.json")
    
    max_layer = round(model.n_layer * (1 - filter_layer_pct)) - 1
    filtered_results = [x for x in results if x["layer"] < max_layer] # Filter layers close to the last layer
    top_layer_results = sorted(filtered_results, key=lambda x: x["RMSE"]) # Sort layers by RMSE

    logging.info(f"Top layers: {[x['layer'] for x in top_layer_results]}")
    save_to_json_file(top_layer_results, save_dir / "top_layers.json")

    return top_layer_results


def run_debias_test(model: ModelBase, prompts: List[str], target_token_ids: Dict, layer: int, intervene_func: Callable, batch_size: int = 32):
    prompt_iterator = PromptIterator(prompts, batch_size=batch_size)
    pos_probs_all, neg_probs_all = torch.tensor([]), torch.tensor([])

    for prompt_batch in prompt_iterator:
        prompt_batch = model.apply_chat_template(prompt_batch)
        logits = model.get_logits(
            prompt_batch, layer=layer, intervene_func=intervene_func
        )

        probs = F.softmax(logits[:, -1, ], dim=-1)
        pos_probs = probs[:, target_token_ids["pos"]].sum(dim=-1)
        neg_probs = probs[:, target_token_ids["neg"]].sum(dim=-1)
        pos_probs_all = torch.concat((pos_probs_all, pos_probs))
        neg_probs_all = torch.concat((neg_probs_all, neg_probs))

    bias = (pos_probs_all - neg_probs_all).numpy()
    normalized_bias = (pos_probs_all - neg_probs_all) / (pos_probs_all + neg_probs_all)
    return bias, normalized_bias.numpy()


def validate(cfg: Config, model: ModelBase, val_data: pd.DataFrame, target_token_ids):
    save_dir = cfg.artifact_path() / "validation"
    activation_dir = cfg.artifact_path() / "activations"
    candidate_vectors = torch.load(activation_dir / "candidate_vectors.pt")

    if cfg.data_cfg.output_prefix:
        prompts = model.apply_chat_template(val_data.prompt.tolist(), output_prefix=val_data.output_prefix.tolist())
    else:
        prompts = model.apply_chat_template(val_data.prompt.tolist())

    bias_baseline = val_data["bias"].to_numpy()

    top_layer_results = evaluate_candidate_vectors(
        model, prompts, candidate_vectors, bias_baseline, 
        save_dir, cfg.filter_layer_pct, cfg.batch_size, 
    )
   
    debiased_results = []
    score_outputs = []

    for layer_results in top_layer_results[:cfg.evaluate_top_n_layer]:
        layer = layer_results["layer"]
        steering_vec = model.set_dtype(candidate_vectors[layer])
        intervene_func = get_intervention_func(steering_vec, coeff=0)
        bias, normalized_bias = run_debias_test(model, prompts, target_token_ids, layer, intervene_func, batch_size=cfg.batch_size)

        rms = RMS(bias)
        is_undershoot = np.where(np.sign(bias) == np.sign(bias_baseline), 1, 0)
        undershoot = RMS(bias * is_undershoot)
        overshoot = RMS(bias * (1 - is_undershoot))

        debiased_results.append({
            "layer": layer,
            "rms": rms,
            "normalized_rms": RMS(normalized_bias),
            "overshoot": overshoot, 
            "undershoot": undershoot, 
        })
        score_outputs.append({
            "layer": layer,
            "bias_scores": bias.tolist(),
            "normalized_bias_scores": normalized_bias.tolist() 
        })

        print(f"Layer {layer}")
        print(f"RMS bias: {RMS(bias_baseline):.4f} (before), {rms: .4f} (after)")
        print(f"Undershoot: {undershoot:.4f}, Overshoot: {overshoot:.4f}")

    save_to_json_file(score_outputs, save_dir / "debiased_scores.json")
    debiased_results = sorted(debiased_results, key=lambda x: x["rms"])
    save_to_json_file(debiased_results, save_dir / "debiased_results.json")
