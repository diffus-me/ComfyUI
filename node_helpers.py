import inspect
import hashlib
import torch

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


def pillow(fn, arg):
    prev_value = None
    try:
        x = fn(arg)
    except (OSError, UnidentifiedImageError, ValueError):  # PIL issues #4472 and #2445, also fixes ComfyUI issue #3416
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
        include_hidden=True,
        return_schema=False
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
        elif param.name == "include_hidden":
            inputs.insert(i, include_hidden)
        elif param.name == "return_schema":
            inputs.insert(i, return_schema)

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
        source = source[..., :destination.shape[-1]]
    elif destination.shape[-1] > source.shape[-1]:
        destination = torch.nn.functional.pad(destination, (0, 1))
        destination[..., -1] = 1.0
    return destination, source
