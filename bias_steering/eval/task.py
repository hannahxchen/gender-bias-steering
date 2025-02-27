from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Callable
import torch
from torchtyping import TensorType
from ..steering import ModelBase


class Task(ABC):
    def __init__(
        self, task_name: str, 
        run_generation: bool = False 
    ):
        self.task_name = task_name
        self.run_generation = run_generation
        self.max_new_tokens = 50
        self.subtasks = self.get_subtask_list()
        self.dataset = self.load_dataset()
        self.eval_metrics = None

    @abstractmethod
    def load_dataset(self):
        pass

    @abstractmethod
    def get_subtask_list(self):
        pass
    
    @abstractmethod
    def prepare_inputs(self, chat_template_func: Callable, subtask_name=None, answer_prefix=" ") -> List[Dict]:
        pass

    def get_answer_token_ids(self, tokenizer, data) -> Dict[str, TensorType[-1]]:
        unique_answers = set([ans for x in data for ans in x["answer_options"]])
        if hasattr(tokenizer, "add_prefix_space"):
            if tokenizer.add_prefix_space is True:
                return {ans: torch.tensor(tokenizer(ans.lstrip(), add_special_tokens=False).input_ids) for ans in unique_answers}
        return {ans: torch.tensor(tokenizer(ans, add_special_tokens=False).input_ids) for ans in unique_answers}
    
    @abstractmethod
    def run_eval(self, model: ModelBase, layer: int, intervene_func: Callable, save_dir: Path, subtask: str, batch_size=32):
        pass

    