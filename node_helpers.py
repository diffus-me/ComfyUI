import inspect
import hashlib
import torch
import logging

from comfy.cli_args import args

from PIL import ImageFile, UnidentifiedImageError

import execution_context


def conditioning_set_values(conditioning, values={}, append=False):
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]
        for k in values:
            val = values[k]
            if append:
                old_val = n[1].get(k, None)
                if old_val is not None:
                    val = old_val + val

            n[1][k] = val
        c.append(n)

    return c

def conditioning_set_values_with_timestep_range(conditioning, values={}, start_percent=0.0, end_percent=1.0):
    """
    Apply values to conditioning only during [start_percent, end_percent], keeping the
    original conditioning active outside that range. Respects existing per-entry ranges.
    """
    if start_percent > end_percent:
        logging.warning(f"start_percent ({start_percent}) must be <= end_percent ({end_percent})")
        return conditioning

    EPS = 1e-5 # the sampler gates entries with strict > / <, shift boundaries slightly to ensure only one conditioning is active per timestep
    c = []
    for t in conditioning:
        cond_start = t[1].get("start_percent", 0.0)
        cond_end   = t[1].get("end_percent",   1.0)
        intersect_start = max(start_percent, cond_start)
        intersect_end   = min(end_percent,   cond_end)

        if intersect_start >= intersect_end: # no overlap: emit unchanged
            c.append(t)
            continue

        if intersect_start > cond_start: # part before the requested range
            c.extend(conditioning_set_values([t], {"start_percent": cond_start, "end_percent": intersect_start - EPS}))

        c.extend(conditioning_set_values([t], {**values, "start_percent": intersect_start, "end_percent": intersect_end}))

        if intersect_end < cond_end: # part after the requested range
            c.extend(conditioning_set_values([t], {"start_percent": intersect_end + EPS, "end_percent": cond_end}))
    return c

def pillow(fn, arg):
    prev_value = None
    try:
        x = fn(arg)
    except (OSError, UnidentifiedImageError, ValueError): #PIL issues #4472 and #2445, also fixes ComfyUI issue #3416
        prev_value = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        x = fn(arg)
    finally:
        if prev_value is not None:
            ImageFile.LOAD_TRUNCATED_IMAGES = prev_value
    return x


def get_node_input_types(
        context: execution_context.ExecutionContext,
        node_class,
):
    signature = inspect.signature(node_class.INPUT_TYPES)
    positional_args = []
    inputs = []
    for i, param in enumerate(signature.parameters.values()):
        if param.kind not in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            break
        positional_args.append(param)
    for i, param in enumerate(positional_args):
        if (param.annotation == str or param.annotation == "str") and param.name == 'user_hash':
            inputs.insert(i, context.user_hash)
        elif param.annotation == execution_context.ExecutionContext or param.annotation == "execution_context.ExecutionContext":
            inputs.insert(i, context)

    while len(inputs) < len(positional_args):
        i = len(inputs)
        param = positional_args[i]
        if param.default == param.empty:
            inputs.append(None)
        else:
            inputs.append(param.default)
    return node_class.INPUT_TYPES(*inputs)


def hasher():
    hashfuncs = {
        "md5": hashlib.md5,
        "sha1": hashlib.sha1,
        "sha256": hashlib.sha256,
        "sha512": hashlib.sha512
    }
    return hashfuncs[args.default_hashing_function]

def string_to_torch_dtype(string):
    if string == "fp32":
        return torch.float32
    if string == "fp16":
        return torch.float16
    if string == "bf16":
        return torch.bfloat16

def image_alpha_fix(destination, source):
    if destination.shape[-1] < source.shape[-1]:
        source = source[...,:destination.shape[-1]]
    elif destination.shape[-1] > source.shape[-1]:
        source = torch.nn.functional.pad(source, (0, 1))
        source[..., -1] = 1.0
    return destination, source
