import execution_context
import math


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


def _reactor_restore_face_consumption(image, model, visibility, codeformer_weight, facedetection, context: execution_context.ExecutionContext):
    return {
        'width': image.shape[2],
        'height': image.shape[1],
        'steps': 30,
        'n_iter': 0 if model == 'none' else 1,
        'batch_size': image.shape[0]
    }


def _reactor_face_swap_consumption(enabled,
                                   input_image,
                                   swap_model,
                                   detect_gender_source,
                                   detect_gender_input,
                                   source_faces_index,
                                   input_faces_index,
                                   console_log_level,
                                   face_restore_model,
                                   face_restore_visibility,
                                   codeformer_weight,
                                   facedetection,
                                   source_image=None,
                                   face_model=None,
                                   faces_order=None,
                                   context: execution_context.ExecutionContext = None):
    return {
        'width': input_image.shape[2],
        'height': input_image.shape[1],
        'steps': 30,
        'n_iter': 2,  # 1 for k_sampler, 1 for face detection
        'batch_size': input_image.shape[0],
    }


def _k_sampler_advanced_consumption(model,
                                    add_noise,
                                    noise_seed,
                                    steps,
                                    cfg,
                                    sampler_name,
                                    scheduler,
                                    positive,
                                    negative,
                                    latent_image,
                                    start_at_step,
                                    end_at_step,
                                    return_with_leftover_noise,
                                    denoise=1.0,
                                    context=None):
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


def _tsc_k_sampler_consumption(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                               preview_method, vae_decode, denoise=1.0, prompt=None, extra_pnginfo=None, my_unique_id=None,
                               context: execution_context.ExecutionContext = None,
                               optional_vae=(None,), script=None, add_noise=None, start_at_step=None, end_at_step=None,
                               return_with_leftover_noise=None, sampler_type="regular"):
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


def _ultimate_sd_upscale_consumption(image, model, positive, negative, vae, upscale_by, seed,
                                     steps, cfg, sampler_name, scheduler, denoise, upscale_model,
                                     mode_type, tile_width, tile_height, mask_blur, tile_padding,
                                     seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                                     seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode, context: execution_context.ExecutionContext):
    batch_size = image.shape[0]
    image_width = image.shape[2]
    image_height = image.shape[1]

    if upscale_model is not None:
        enable_hr = True
        hr_resize_x = image_width * upscale_model.scale
        hr_resize_y = image_height * upscale_model.scale
    else:
        enable_hr = False
        hr_resize_x = 0
        hr_resize_y = 0

    redraw_width = math.ceil((image_width * upscale_by) / 64) * 64
    redraw_height = math.ceil((image_height * upscale_by) / 64) * 64

    return {
        'width': redraw_width,
        'height': redraw_height,
        'enable_hr': enable_hr,
        'hr_resize_x': hr_resize_x,
        'hr_resize_y': hr_resize_y,
        'steps': steps,
        'n_iter': 1,
        'batch_size': batch_size,
    }


def _image_upscale_with_model_consumption(upscale_model, image):
    return {
        'resize_x': image.shape[2] * upscale_model.scale,
        'resize_y': image.shape[1] * upscale_model.scale,
        'batch_size': image.shape[0],
    }


model_upscale_cache = {
    '16xPSNR.pth': 16,
    '1x_NMKD-BrightenRedux_200k.pth': 1,
    '1x_NMKD-YandereInpaint_375000_G.pth': 1,
    '1x_NMKDDetoon_97500_G.pth': 1,
    '1x_NoiseToner-Poisson-Detailed_108000_G.pth': 1,
    '1x_NoiseToner-Uniform-Detailed_100000_G.pth': 1,
    '4x-AnimeSharp.pth': 4,
    '4x-UltraSharp.pth': 4,
    '4xPSNR.pth': 4,
    '4x_CountryRoads_377000_G.pth': 4,
    '4x_Fatality_Comix_260000_G.pth': 4,
    '4x_NMKD-Siax_200k.pth': 4,
    '4x_NMKD-Superscale-Artisoftject_210000_G.pth': 4,
    '4x_NMKD-Superscale-SP_178000_G.pth': 4,
    '4x_NMKD-UltraYandere-Lite_280k.pth': 4,
    '4x_NMKD-UltraYandere_300k.pth': 4,
    '4x_NMKD-YandereNeoXL_200k.pth': 4,
    '4x_NMKDSuperscale_Artisoft_120000_G.pth': 4,
    '4x_NickelbackFS_72000_G.pth': 4,
    '4x_Nickelback_70000G.pth': 4,
    '4x_RealisticRescaler_100000_G.pth': 4,
    '4x_UniversalUpscalerV2-Neutral_115000_swaG.pth': 4,
    '4x_UniversalUpscalerV2-Sharp_101000_G.pth': 4,
    '4x_UniversalUpscalerV2-Sharper_103000_G.pth': 4,
    '4x_Valar_v1.pth': 4,
    '4x_fatal_Anime_500000_G.pth': 4,
    '4x_foolhardy_Remacri.pth': 4,
    '4x_foolhardy_Remacri_ExtraSmoother.pth': 4,
    '8xPSNR.pth': 8,
    '8x_NMKD-Superscale_150000_G.pth': 8,
    '8x_NMKD-Typescale_175k.pth': 8,
    "A_ESRGAN_Single.pth": 4,
    "BSRGAN.pth": 4,
    'BSRGANx2.pth': 2,
    "BSRNet.pth": 4,
    'ESRGAN_4x.pth': 4,
    "LADDIER1_282500_G.pth": 4,
    'RealESRGAN_x4plus.pth': 4,
    'RealESRGAN_x4plus_anime_6B.pth': 4,
    'SwinIR_4x.pth': 4,
    "WaifuGAN_v3_30000.pth": 4,
    "lollypop.pth": 4,
}


def _easy_hires_fix_consumption(
        model_name, rescale_after_model, rescale_method, rescale, percent, width, height,
        longer_side, crop, image_output, link_id, save_prefix, pipe=None, image=None, vae=None, prompt=None,
        extra_pnginfo=None, my_unique_id=None, context: execution_context.ExecutionContext = None):
    if model_name not in model_upscale_cache:
        try:
            import folder_paths
            import comfy
            from comfy_extras.chainner_models import model_loading
            model_path = folder_paths.get_full_path(context, "upscale_models", model_name)
            sd = comfy.utils.load_torch_file(model_path, safe_load=True)
            upscale_model = model_loading.load_state_dict(sd).eval()
            model_upscale_cache[model_name] = upscale_model.scale
            del upscale_model
        except Exception as e:
            model_upscale_cache[model_name] = 4
    model_scale = model_upscale_cache[model_name]

    if pipe is not None:
        image = image if image is not None else pipe["images"]
    if image is not None:
        return {
            'resize_x': image.shape[2] * model_scale,
            'resize_y': image.shape[1] * model_scale,
            'batch_size': image.shape[0],
        }
    else:
        return {
            'resize_x': 1,
            'resize_y': 1,
            'batch_size': 0,
        }


def _vhs_video_combine_consumption(
        images,
        frame_rate: int,
        loop_count: int,
        filename_prefix="AnimateDiff",
        format="image/gif",
        pingpong=False,
        save_output=True,
        prompt=None,
        extra_pnginfo=None,
        audio=None,
        unique_id=None,
        manual_format_widgets=None,
        meta_batch=None,
        context: execution_context.ExecutionContext = None,
):
    return {
        'resize_x': images.shape[2],
        'resize_y': images.shape[1],
        'batch_size': images.shape[0],
    }


def _face_detailer_pipe_consumption(
        image, detailer_pipe, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name, scheduler,
        denoise, feather, noise_mask, force_inpaint, bbox_threshold, bbox_dilation, bbox_crop_factor,
        sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion,
        sam_mask_hint_threshold, sam_mask_hint_use_negative, drop_size, refiner_ratio=None,
        cycle=1, inpaint_model=False, noise_mask_feather=0
):
    model, clip, vae, positive, negative, wildcard, bbox_detector, segm_detector_opt, sam_model_opt, detailer_hook, \
        refiner_model, refiner_clip, refiner_positive, refiner_negative = detailer_pipe

    n_iter = 1
    if sam_model_opt is not None or segm_detector_opt is not None:
        n_iter += 1
    return [{
        'width': img.shape[3],
        'height': img.shape[2],
        'steps': steps,
        'n_iter': n_iter,
        'batch_size': img.shape[0],
    } for img in image]


def _face_detailer_consumption(
        image, model, clip, vae, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name, scheduler,
        positive, negative, denoise, feather, noise_mask, force_inpaint,
        bbox_threshold, bbox_dilation, bbox_crop_factor,
        sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
        sam_mask_hint_use_negative, drop_size, bbox_detector, wildcard, cycle=1,
        sam_model_opt=None, segm_detector_opt=None, detailer_hook=None, inpaint_model=False, noise_mask_feather=0
):
    n_iter = 1
    if sam_model_opt is not None or segm_detector_opt is not None:
        n_iter += 1
    return {
        'width': image.shape[3],
        'height': image.shape[2],
        'steps': steps,
        'n_iter': n_iter,
        'batch_size': image.shape[0],
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


_NODE_CONSUMPTION_MAPPING = {
    'KSampler': _k_sampler_consumption,
    'KSamplerAdvanced': _k_sampler_advanced_consumption,
    'KSampler (Efficient)': _tsc_k_sampler_consumption,
    'ReActorRestoreFace': _reactor_restore_face_consumption,
    'ReActorFaceSwap': _reactor_face_swap_consumption,
    'ImageUpscaleWithModel': _image_upscale_with_model_consumption,
    'UltimateSDUpscale': _ultimate_sd_upscale_consumption,
    'easy hiresFix': _easy_hires_fix_consumption,
    'VHS_VideoCombine': _vhs_video_combine_consumption,
    'FaceDetailer': _face_detailer_consumption,
    'FaceDetailerPipe': _face_detailer_pipe_consumption,
}


def get_monitor_params(obj, obj_type, input_data_all):
    func = _NODE_CONSUMPTION_MAPPING.get(obj_type, _default_consumption_maker)
    return _map_node_consumption_over_list(obj, input_data_all, func)
