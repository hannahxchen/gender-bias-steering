import json
import logging
from pathlib import Path
from typing import Dict
import pandas as pd
from .template import Template
from ..config import DataConfig

DATASET_DIR = Path(__file__).resolve().parent / "datasets"


def load_dataframe_from_json(filepath):
    data = json.load(open(filepath, "r"))
    return pd.DataFrame.from_records(data)


def load_target_words():
    return json.load(open(DATASET_DIR / "target_words.json", "r"))


def load_gendered_language_dataset(split: str, include_neutral=True, sample_size=None):
    data = pd.read_csv(DATASET_DIR / f"splits/{split}.csv")

    if not include_neutral:
        data = data[~data.is_neutral]

    if sample_size is not None:
        data = data.sample(n=sample_size)
    
    instructions = [line.strip() for line in open(DATASET_DIR / f"instructions/{split}.txt", "r").readlines()]
    instruction_set = Template(instructions)

    instructions = [instruction_set.get_template() for _ in range(len(data))]
    prompts, output_prefixes = [], []

    for inst, text in zip(instructions, data["text"]):
        inst, output_prefix = inst.split(" | ")
        prompts.append(f'{inst}\n{text}')
        output_prefixes.append(output_prefix)
    
    data["prompt"] = prompts
    data["output_prefix"] = output_prefixes

    return data


def load_datasplits(cfg: DataConfig, save_dir: Path, use_cache: bool = False) -> Dict[str, pd.DataFrame]:
    datasets = {}        
    for split in ["train", "val"]:
        if use_cache and Path(save_dir / f"{split}.json").exists():
            logging.info(f"Loading cached data from {save_dir}/{split}.json")
            datasets[split] = load_dataframe_from_json(save_dir / f"{split}.json")
        else:
            if split == "val":
                sample_size = cfg.n_val
            else:
                sample_size = None
            datasets[split] = load_gendered_language_dataset(split, include_neutral=True, sample_size=sample_size)
    
    return datasets
