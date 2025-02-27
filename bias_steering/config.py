import os
from pathlib import Path
from dataclasses import dataclass
from dataclass_wizard import YAMLWizard
from typing import Self


@dataclass
class DataConfig:
    pos_label: str = "F" # Positive label
    neg_label: str = "M" # Negative label
    n_train: int = 800 # Size per label
    n_val: int = 1600 # Total size
    bias_threshold: float = 0.1
    output_prefix: bool = True


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
    seed: int = 4278

    def __post_init__(self):
        self.model_alias = os.path.basename(self.model_name)
        if self.save_dir is None:
            self.save_dir = "runs"
    
    def artifact_path(self) -> Path:
        return Path().absolute() / self.save_dir / self.model_alias / self.method
    
    def baseline_artifact_path(self) -> Path:
        return Path().absolute() / self.save_dir / self.model_alias

    def save(self):
        os.makedirs(self.artifact_path(), exist_ok=True)
        self.to_yaml_file(self.artifact_path() / 'config.yaml')

    def load(filepath: str) -> Self:
        try:
            return Config.from_yaml_file(filepath)
        
        except FileNotFoundError:
            return None
