def _k_sampler_consumption(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, context=None):
    n_iter = latent_image.get("batch_index", 1)

    latent = latent_image["samples"]
    latent_size = latent.size()
    batch_size = latent_size[0]
    image_height = latent_size[2] * 8
    image_width = latent_size[3] * 8
    return {
        'width': image_width,
        'height': image_height,
        'steps': steps,
        'n_iter': n_iter,
        'batch_size': batch_size
    }


_NODE_CONSUMPTION_MAPPING = {
    'KSampler': _k_sampler_consumption
}


def _slice_dict(d, i):
    d_new = dict()
    for k, v in d.items():
        d_new[k] = v[i if len(v) > i else -1]
    return d_new


def _map_node_consumption_over_list(obj, input_data_all, func):
    # check if node wants the lists
    input_is_list = getattr(obj, "INPUT_IS_LIST", False)

    if len(input_data_all) == 0:
        max_len_input = 0
    else:
        max_len_input = max([len(x) for x in input_data_all.values()])

    if input_is_list:
        return func(**input_data_all)
    elif max_len_input == 0:
        return func()
    else:
        results = []
        for i in range(max_len_input):
            results.append(func(**_slice_dict(input_data_all, i)))
        if len(results) == 1:
            return results[0]
        else:
            return results


def _default_consumption_maker(*args, **kwargs):
    return {}


def get_monitor_params(obj, obj_type, input_data_all):
    func = _NODE_CONSUMPTION_MAPPING.get(obj_type, _default_consumption_maker)
    return _map_node_consumption_over_list(obj, input_data_all, func)
