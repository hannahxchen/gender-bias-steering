import os, json
import random
import argparse
import warnings
import logging
from pathlib import Path
from typing import Tuple, List, Dict

import torch
import torch.nn.functional as F
import numpy as np
from .config import Config, DataConfig
from .utils import save_to_json_file
from .data.load_dataset import load_datasplits, load_target_words
from .data.prompt_iterator import PromptIterator
from .steering import load_model, ModelBase, extract_candidate_vectors, \
    validate, get_intervention_func, get_target_token_ids, compute_projections
from .eval import load_evaluation_task

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.set_grad_enabled(False);
logging.basicConfig(level=logging.INFO)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default=None, help='Load configuration from file.')
    parser.add_argument('--model_name', type=str, help='Model name')
    parser.add_argument('--method', type=str, default="WMD", choices=["WMD", "MD"], help='Method for computing candidate vectors.')
    parser.add_argument('--use_offset', action='store_true', help="Offset by neutral examples.")
    parser.add_argument('--n_train_per_label', type=int, default=800, help="Number of training examples per label.")
    parser.add_argument('--n_val', type=int, default=1600, help="Number of validation examples.")
    parser.add_argument('--bias_threshold', type=float, default=0.1)
    parser.add_argument('--filter_layer_pct', type=float, default=0.05, help='Filter last N percentage layers.')
    parser.add_argument('--evaluate_top_n_layer', type=int, default=5, help='Evaluate top n layers.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--seed', type=int, default=4238, help='Random seed.')
    parser.add_argument('--save_dir', type=str, default=None, help='Save results to specified directory.')
    parser.add_argument('--use_cache', action='store_true', help='Reuse stored cached results.')
    parser.add_argument('--run_eval', action='store_true', help='Run transferability evaluation.')
    parser.add_argument('--layer', type=int, help="Intervention layer.")
    parser.add_argument('--intervention_method', type=str, default="scaled_proj", choices=["scaled_proj", "constant"], help="Intervention method")
    parser.add_argument('--coeff', type=float, default=-1.0, help="Steering coefficient.")
    return parser.parse_args()


def get_baseline_results(
    model: ModelBase, prompts: List[str],
    target_token_ids: Dict[str, List], 
    batch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    pos_probs_all, neg_probs_all = torch.tensor([]), torch.tensor([])
    prompt_iterator = PromptIterator(prompts, batch_size=batch_size)

    for prompt_batch in prompt_iterator:
        logits = model.get_last_position_logits(prompt_batch)
        probs = F.softmax(logits, dim=-1)

        pos_probs = probs[:, target_token_ids["pos"]].sum(dim=-1)
        neg_probs = probs[:, target_token_ids["neg"]].sum(dim=-1)

        pos_probs_all = torch.concat((pos_probs_all, pos_probs))
        neg_probs_all = torch.concat((neg_probs_all, neg_probs))

    return pos_probs_all.numpy(), neg_probs_all.numpy()


def train_and_validate(cfg: Config, model: ModelBase):
    datasplits_dir = cfg.baseline_artifact_path() / "datasplits"
    data_cfg = cfg.data_cfg
    datasets = load_datasplits(data_cfg, datasplits_dir, use_cache=cfg.use_cache)
    os.makedirs(datasplits_dir, exist_ok=True)

    logging.info("Preprocessing train/val data")
    target_words_by_label = load_target_words()
    target_token_ids = {}
    for label, k in zip([data_cfg.pos_label, data_cfg.neg_label], ["pos", "neg"]):
        target_token_ids[k] = get_target_token_ids(model.tokenizer, target_words_by_label[label])

    for split in ["train", "val"]:
        df = datasets[split].copy()

        if cfg.use_cache is True and ("pos_prob" in df.columns):
            continue

        logging.info(f"Getting baseline results for {split} split")
        if data_cfg.output_prefix:
            prompts = model.apply_chat_template(df["prompt"].tolist(), output_prefix=df["output_prefix"].tolist())
        else:
            prompts = model.apply_chat_template(df["prompt"].tolist())
            
        pos_probs, neg_probs = get_baseline_results(model, prompts, target_token_ids, batch_size=cfg.batch_size)
        df["pos_prob"] = pos_probs
        df["neg_prob"] = neg_probs
        df["bias"] = pos_probs - neg_probs
            
        datasets[split] = df
        save_to_json_file(df.to_dict("records"), datasplits_dir / f"{split}.json")

    if not cfg.use_cache or not Path(cfg.artifact_path() / "activations/candidate_vectors.pt").is_file():
        train_data = datasets["train"]
        pos_examples = train_data[(train_data.bias > data_cfg.bias_threshold)]
        neg_examples = train_data[(train_data.bias < -data_cfg.bias_threshold)]

        if data_cfg.n_train is not None:
            n = min([data_cfg.n_train, pos_examples.shape[0], neg_examples.shape[0]])
            pos_examples = pos_examples.sample(n=n)
            neg_examples = neg_examples.sample(n=n)

        if cfg.use_offset:
            neutral_examples = train_data[(train_data.bias.abs() <= data_cfg.bias_threshold)]
            neutral_examples = neutral_examples.sample(n=min(n, len(neutral_examples)))
        else:
            neutral_examples = None
        extract_candidate_vectors(cfg, model, pos_examples, neg_examples, neutral_examples)

    validate(cfg, model, datasets["val"], target_token_ids)


def eval(
    cfg: Config, model: ModelBase, 
    intervention_method="scaled_proj", layer=None, coeff=-1.0, 
    eval_tasks=["winogenerated", "occupational_stereotypes"]
):
    if layer is None:
        layer = json.load(open(cfg.artifact_path() / "validation/top_layers.json", "r"))[0]["layer"]

    print(f"Intervene layer: {layer}")
    save_dir = cfg.artifact_path() / "evaluation"
    os.makedirs(save_dir, exist_ok=True)

    candidate_vectors = torch.load(cfg.artifact_path() / f"activations/candidate_vectors.pt")
    steering_vec = candidate_vectors[layer]
    steering_vec = model.set_dtype(steering_vec)
    intervene_func = get_intervention_func(steering_vec, method=intervention_method, coeff=coeff)

    for task_name in eval_tasks:
        logging.info(f"Running evaluation task: {task_name}")
        task = load_evaluation_task(task_name)
        for subtask in task.get_subtask_list():
            task.run_eval(model, layer, intervene_func, save_dir=save_dir, subtask=subtask, batch_size=cfg.batch_size)
            inputs = task.prepare_inputs(model.apply_chat_template, subtask=subtask)
            prompts = [x["prompt"] for x in inputs]
            projections = compute_projections(model, steering_vec, layer, prompts, batch_size=cfg.batch_size)
            np.save(save_dir / f"{task.task_name}_{subtask}_projections.npy", projections.numpy())

def main():
    args = parse_arguments()
    
    if args.config_file is not None:
        cfg = Config.load(args.config_file)
        logging.info(f"Loaded config file: {args.config_file}")
    else:
        data_cfg = DataConfig(
            n_train=args.n_train_per_label, n_val=args.n_val,
            bias_threshold=args.bias_threshold, 
        )
        cfg = Config(
            model_name=args.model_name, data_cfg=data_cfg, 
            method=args.method, use_offset=args.use_offset, seed=args.seed,
            evaluate_top_n_layer=args.evaluate_top_n_layer, 
            filter_layer_pct=args.filter_layer_pct, save_dir=args.save_dir,
            batch_size=args.batch_size, use_cache=args.use_cache,
        )
        cfg.save()

    print("Model:", cfg.model_name)
    print("Configuration:\n", repr(cfg))

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = load_model(cfg.model_name)

    if args.run_eval:
        eval(cfg, model, args.intervention_method, args.layer, args.coeff)
    else:
        train_and_validate(cfg, model)


if __name__ == "__main__":
    main()