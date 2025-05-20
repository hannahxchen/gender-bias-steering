import os, json
import random
import argparse
import warnings
import logging
from typing import List, Dict

import torch
import torch.nn.functional as F
import numpy as np
from .config import Config
from .utils import save_to_json_file, loop_coeffs
from .data.load_dataset import load_dataframe_from_json, load_target_words
from .data.prompt_iterator import PromptIterator
from .steering import load_model, ModelBase, get_intervention_func, get_target_token_ids

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.set_grad_enabled(False);
logging.basicConfig(level=logging.INFO)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True, help='Load configuration from file.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--test_size', type=int, default=100, help='Test size.')
    parser.add_argument('--seed', type=int, default=4238, help='Random seed.')
    parser.add_argument('--layer', type=int, default=None, help="Intervention layer.")
    parser.add_argument('--min_coeff', type=float, default=-15, help="Min coefficient.")
    parser.add_argument('--max_coeff', type=float, default=15, help="Max coefficient.")
    parser.add_argument('--increment', type=float, default=1, help="Increment.")
    return parser.parse_args()


def run(
    cfg: Config, model: ModelBase, prompts: List[str], steering_vec: torch.Tensor,
    target_token_ids: Dict, intervention_method="scaled_proj", layer=None, coeff=-1.0, offset=0
):
    prompt_iterator = PromptIterator(prompts, batch_size=cfg.batch_size, desc=f"Running coefficient {coeff:.1f}", show_progress_bar=True)
    intervene_func = get_intervention_func(steering_vec, method=intervention_method, coeff=coeff, offset=offset)
    pos_probs_all, neg_probs_all = torch.tensor([]), torch.tensor([])

    for prompt_batch in prompt_iterator:
        logits = model.get_logits(
            prompt_batch, layer=layer, intervene_func=intervene_func
        )

        probs = F.softmax(logits[:, -1, ], dim=-1)
        pos_probs = probs[:, target_token_ids["pos"]].sum(dim=-1)
        neg_probs = probs[:, target_token_ids["neg"]].sum(dim=-1)
        pos_probs_all = torch.concat((pos_probs_all, pos_probs))
        neg_probs_all = torch.concat((neg_probs_all, neg_probs))

    return pos_probs_all.tolist(), neg_probs_all.tolist()


def main():
    args = parse_arguments()
    cfg = Config.load(args.config_file)
    logging.info(f"Loaded config file: {args.config_file}")
    
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    print("Model:", cfg.model_name)
    model = load_model(cfg.model_name)

    if args.layer is None:
        layer = json.load(open(cfg.artifact_path() / "validation/top_layers.json", "r"))[0]["layer"]
    else:
        layer = args.layer

    print(f"Intervene layer: {layer}")
    save_dir = cfg.artifact_path() / "coeff_test"
    os.makedirs(save_dir, exist_ok=True)

    val_df = load_dataframe_from_json(cfg.artifact_path() / "datasplits/val.json")
    sampled_df = val_df.sample(n=args.test_size)
    sampled_prompts = model.apply_chat_template(sampled_df["prompt"].tolist(), output_prefix=sampled_df["output_prefix"].tolist())

    target_words_by_label = load_target_words(target_concept=cfg.data_cfg.target_concept)
    target_token_ids = {}
    for label, k in zip([cfg.data_cfg.pos_label, cfg.data_cfg.neg_label], ["pos", "neg"]):
        target_token_ids[k] = get_target_token_ids(model.tokenizer, target_words_by_label[label])

    results = [{"_id": _id, "pos_probs": [], "neg_probs": []} for _id in sampled_df["_id"].tolist()]
    coeffs = loop_coeffs(min_coeff=args.min_coeff, max_coeff=args.max_coeff, increment=args.increment)
    save_to_json_file({"coeff": coeffs}, save_dir / "coeffs.json")

    steering_vec = torch.load(cfg.artifact_path() / f"activations/candidate_vectors.pt")[layer]
    offset = torch.load(cfg.artifact_path() / f"activations/neutral.pt")[layer].mean(dim=0)
    steering_vec = model.set_dtype(steering_vec)
    offset = model.set_dtype(offset)

    for coeff in coeffs:
        pos_probs_all, neg_probs_all = run(cfg, model, sampled_prompts, steering_vec, target_token_ids, layer=layer, coeff=coeff, offset=offset)

        for i in range(len(sampled_prompts)):
            results[i]["pos_probs"].append(pos_probs_all[i])
            results[i]["neg_probs"].append(neg_probs_all[i])

        save_to_json_file(results, save_dir / "outputs.json")


if __name__ == "__main__":
    main()