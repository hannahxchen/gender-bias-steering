import os
from pathlib import Path
from dataclasses import dataclass
from dataclass_wizard import YAMLWizard
from typing import Self


@dataclass
class DataConfig:
    target_concept: str = "gender"
    pos_label: str = "F" # Positive label
    neg_label: str = "M" # Negative label
    n_train: int = 800
    n_val: int = 1000
    bias_threshold: float = 0.05
    output_prefix: bool = True
    weighted_sample: bool = False


@dataclass
class Config(YAMLWizard):
    model_name: str
    data_cfg: DataConfig
    method: str # Vector extraction method
    use_offset: bool # Offset by neutral examples
    evaluate_top_n_layer: int = 5 # Evaluate intervention performance for top layers
    filter_layer_pct: float = 0.05 # Filter the last 5% layers
    save_dir: str = None
    use_cache: bool = True
    batch_size: int = 32
    seed: int = 4238

    def __post_init__(self):
        self.model_alias = os.path.basename(self.model_name)
        if self.save_dir is None:
            self.save_dir = f"runs_{self.data_cfg.target_concept}"
    
    def artifact_path(self) -> Path:
        return Path().absolute() / self.save_dir / self.model_alias

    def save(self):
        os.makedirs(self.artifact_path(), exist_ok=True)
        self.to_yaml_file(self.artifact_path() / 'config.yaml')

    def load(filepath: str) -> Self:
        try:
            return Config.from_yaml_file(filepath)
        
        except FileNotFoundError:
            return None
