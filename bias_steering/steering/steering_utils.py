import copy
from typing import List, Optional
import torch
import torch.nn.functional as F
from torchtyping import TensorType
from transformers import AutoTokenizer
from ..data.prompt_iterator import PromptIterator


def get_token_ids(tokenizer, words):
    token_ids = tokenizer(words, add_special_tokens=False).input_ids
    token_ids = [_ids[0] for _ids in token_ids if len(_ids) == 1]
    return list(set(token_ids))


def get_target_token_ids(tokenizer: AutoTokenizer, target_words: List[str]) -> List[int]:
    words = copy.deepcopy(target_words)
    words += [w.capitalize() for w in words]

    # Handle cases like ' male', ' female'
    words += [" " + w for w in words]
    token_ids = get_token_ids(tokenizer, words)

    # Handle cases without prefix space
    token_ids += tokenizer.convert_tokens_to_ids(words)
    token_ids = [_ids for _ids in token_ids if _ids != tokenizer.unk_token_id]

    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    target_token_ids = tokenizer.convert_tokens_to_ids(tokens)
    target_token_ids = list(set(target_token_ids))
    
    return target_token_ids


def get_all_layer_activations(
    model, prompts: List[str], batch_size: Optional[int] = 32
) -> TensorType["n_layer", "n_prompt", "hidden_size"]:
    acts_all = []
    layers = list(range(model.n_layer))
    prompt_iterator = PromptIterator(prompts, batch_size=batch_size)
    if prompt_iterator.pbar is not None:
        prompt_iterator.pbar.set_description("Extracting activations")

    for prompt_batch in prompt_iterator:
        acts = model.get_activations(layers, prompt_batch, positions=[-1]).squeeze(-2)
        acts_all.append(acts)

    return torch.concat(acts_all, dim=1)


def scalar_projection(acts: TensorType[..., -1], steering_vec: TensorType[-1]):
    cosin_sim = F.cosine_similarity(acts, steering_vec, dim=-1)
    projs = acts.norm(dim=-1) * cosin_sim
    return projs.to(torch.float64)


def compute_projections(model, steering_vec: TensorType, layer: int, prompts: List[str], offset=0, batch_size=32):
        activations = []
        prompt_iterator = PromptIterator(prompts, batch_size=batch_size, desc="Extracting activations for computing projections")
        for prompt_batch in prompt_iterator:
            acts = model.get_activations(layer, prompt_batch, positions=[-1]).squeeze()
            activations.append(acts)
        activations = torch.vstack(activations)
        return scalar_projection(activations - offset, steering_vec)