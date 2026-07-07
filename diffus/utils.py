import math

_SDXL_BASELINE_PARAMS = 2_600_000_000
_SDXL_BASELINE_DTYPE_BYTES = 2


def _unwrap_model(model):
    # ComfyUI MODEL wrapper -> BaseModel
    if hasattr(model, "model"):
        model = model.model

    return model


def _count_model_params(model):
    model = _unwrap_model(model)

    # BaseModel -> actual torch module
    if hasattr(model, "diffusion_model"):
        model = model.diffusion_model

    return sum(_get_tensor_param_count(p) for p in model.parameters())


def _get_tensor_param_count(tensor):
    shape = getattr(tensor, "tensor_shape", None)
    if shape is None:
        shape = getattr(tensor, "orig_shape", None)
    if shape is None:
        return tensor.numel()

    params = 1
    for dim in shape:
        params *= dim
    return params


def _get_model_dtype(model):
    model = _unwrap_model(model)

    if hasattr(model, "get_dtype_inference"):
        dtype = model.get_dtype_inference()
        if dtype is not None:
            return dtype

    if hasattr(model, "get_dtype"):
        dtype = model.get_dtype()
        if dtype is not None:
            return dtype

    if hasattr(model, "dtype"):
        dtype = model.dtype
        if dtype is not None:
            return dtype

    if hasattr(model, "diffusion_model") and hasattr(model.diffusion_model, "dtype"):
        dtype = model.diffusion_model.dtype
        if dtype is not None:
            return dtype

    for param in model.parameters():
        return param.dtype

    return None


def _get_tensor_dtype(tensor):
    if hasattr(tensor, "get_dtype"):
        dtype = tensor.get_dtype()
        if dtype is not None:
            return dtype

    if hasattr(tensor, "dtype"):
        dtype = tensor.dtype
        if dtype is not None:
            return dtype

    return None


def _get_dtype_size(dtype):
    if dtype is None:
        return _SDXL_BASELINE_DTYPE_BYTES

    try:
        import comfy.model_management
        return comfy.model_management.dtype_size(dtype)
    except Exception:
        return getattr(dtype, "itemsize", _SDXL_BASELINE_DTYPE_BYTES)


def _get_ggml_tensor_storage_bytes(tensor):
    tensor_type = getattr(tensor, "tensor_type", None)
    tensor_shape = getattr(tensor, "tensor_shape", None)
    if tensor_type is None or tensor_shape is None:
        return None

    data = getattr(tensor, "data", None)
    if data is not None:
        try:
            return data.numel() * data.element_size()
        except Exception:
            pass

    try:
        from custom_nodes_builtin.gguf.gguf_connector.const import GGML_QUANT_SIZES
        block_size, type_size = GGML_QUANT_SIZES[tensor_type]

        params = _get_tensor_param_count(tensor)
        blocks = math.ceil(params / block_size)
        return blocks * type_size
    except Exception:
        return None


def _get_quantized_tensor_storage_bytes(tensor):
    ggml_storage_bytes = _get_ggml_tensor_storage_bytes(tensor)
    if ggml_storage_bytes is not None:
        return ggml_storage_bytes

    storage_attrs = (
        "qdata",
        "quantized_data",
        "quantized",
        "weight",
        "_data",
    )
    total = 0
    found_storage = False
    for attr in storage_attrs:
        value = getattr(tensor, attr, None)
        if value is None:
            continue
        if hasattr(value, "numel") and hasattr(value, "element_size"):
            total += value.numel() * value.element_size()
            found_storage = True

    return total if found_storage else None


def _get_tensor_storage_bytes(tensor, fallback_dtype_size=None):
    quantized_storage_bytes = _get_quantized_tensor_storage_bytes(tensor)
    if quantized_storage_bytes is not None:
        return quantized_storage_bytes

    tensor_param_count = _get_tensor_param_count(tensor)

    tensor_dtype = _get_tensor_dtype(tensor)
    if tensor_dtype is not None:
        dtype_size = _get_dtype_size(tensor_dtype)
    elif fallback_dtype_size is not None:
        dtype_size = fallback_dtype_size
    else:
        dtype_size = _SDXL_BASELINE_DTYPE_BYTES

    return tensor_param_count * dtype_size


def _get_model_storage_bytes(model):
    unwrapped_model = _unwrap_model(model)
    model_dtype = _get_model_dtype(unwrapped_model)

    # BaseModel -> actual torch module
    if hasattr(unwrapped_model, "diffusion_model"):
        model = unwrapped_model.diffusion_model
    else:
        model = unwrapped_model

    dtype_size = _get_dtype_size(model_dtype) if model_dtype is not None else None
    return sum(_get_tensor_storage_bytes(p, dtype_size) for p in model.parameters())


def calc_model_consumption_ratio(model) -> float:
    try:
        storage_bytes = _get_model_storage_bytes(model)
    except Exception:
        return 1.0

    if storage_bytes <= 0:
        return 1.0

    baseline = _SDXL_BASELINE_PARAMS * _SDXL_BASELINE_DTYPE_BYTES
    ratio = storage_bytes / baseline
    return max(round(ratio), 1)


def get_model_name(model):
    return getattr(model, "model_name", "")