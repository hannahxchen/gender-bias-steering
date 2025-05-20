from typing import Callable
import torch
import torch.nn.functional as F
from torchtyping import TensorType
from transformers import BatchEncoding
from nnsight import LanguageModel
from nnsight.envoy import Envoy


def orthogonal_projection(a: TensorType[..., -1], unit_vec: TensorType[-1]) -> TensorType[..., -1]:
    return a @ unit_vec.unsqueeze(-1) * unit_vec


def apply_intervention(
    model: LanguageModel, 
    inputs: BatchEncoding, 
    layer_block: Envoy,
    intervene_func: Callable,
) -> TensorType["n_prompt", "seq_len", "vocab_size"]:

    with model.trace(inputs) as tracer:
        acts = layer_block.output[0].clone()
        new_acts = intervene_func(acts)
        layer_block.output = (new_acts,) + layer_block.output[1:]
        logits = model.lm_head.output.detach().to("cpu").to(torch.float64).save()

    return logits


def intervene_generation(
    model: LanguageModel, 
    inputs: BatchEncoding, 
    layer_block: Envoy,
    intervene_func: Callable, 
    max_new_tokens: int = 10, 
    do_sample: bool = False, **kwargs
) -> TensorType["n_prompt", "seq_len"]:

    with model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, **kwargs) as tracer:
        acts = layer_block.output[0].clone()
        new_acts = intervene_func(acts)
        layer_block.output = (new_acts,) + layer_block.output[1:]

        for _ in range(max_new_tokens - 1):
            acts = layer_block.next().output[0].t[-1]
            new_acts = intervene_func(acts)
            layer_block.output[0].t[-1] = new_acts

        outputs = model.generator.output.detach().to("cpu").save()
    return outputs.value


def get_intervention_func(steering_vec: TensorType, method="default", offset=0, coeff=-1.0) -> Callable:
    """Get function for model intervention.
    Methods:
    - default: Proposed method.
    - constant: Use a constant steering coefficient.
    """
    unit_vec = F.normalize(steering_vec, dim=-1)

    if method == "default":
        return lambda acts: acts - orthogonal_projection(acts - offset, unit_vec) + unit_vec * coeff
    elif method == "constant":
        return lambda acts: acts + steering_vec * coeff
    else:
        return None