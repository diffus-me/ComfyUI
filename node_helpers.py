import inspect
import hashlib

from comfy.cli_args import args


from PIL import ImageFile, UnidentifiedImageError

import execution_context


def conditioning_set_values(conditioning, values={}):
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]
        for k in values:
            n[1][k] = values[k]
        c.append(n)

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


def get_node_input_types(context: execution_context.ExecutionContext, node_class):
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
        if param.annotation == execution_context.ExecutionContext or param.annotation == "execution_context.ExecutionContext":
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

