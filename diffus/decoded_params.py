import math

import execution_context


def _sample_consumption_ratio(
        context: execution_context.ExecutionContext,
        model,
):
    import comfy.model_base
    import comfy.model_patcher
    if model:
        if isinstance(model, comfy.model_patcher.ModelPatcher):
            model = model.model
        if isinstance(model, comfy.model_base.SD3) or isinstance(model, comfy.model_base.Flux):
            return 3

    for model_info in context.loaded_checkpoints:
        if model_info.base and model_info.base.lower() in ("sd3", "flux"):
            return 3
    return 1


def __update_context_checkpoints_model_base(context: execution_context.ExecutionContext, model):
    import comfy.model_base
    import comfy.model_patcher
    if isinstance(model, comfy.model_patcher.ModelPatcher):
        model = model.model
        if isinstance(model, comfy.model_base.SD3):
            context.checkpoints_model_base = "SD3"
        elif isinstance(model, comfy.model_base.Flux):
            context.checkpoints_model_base = "SD3"
        elif isinstance(model, comfy.model_base.GenmoMochi):
            context.checkpoints_model_base = "MOCHI"


def __get_image_size(model, latent_image):
    import comfy.model_base
    import comfy.model_patcher
    latent = latent_image["samples"]
    latent_size = latent.size()
    batch_size = latent_size[0]
    if isinstance(model, comfy.model_patcher.ModelPatcher) and isinstance(model.model, comfy.model_base.GenmoMochi):
        image_height = latent_size[3] * 8
        image_width = latent_size[4] * 8
        n_iter = max(1, (latent_size[2] - 1) * 6 + 1)
    else:
        image_height = latent_size[2] * 8
        image_width = latent_size[3] * 8
        n_iter = 1
    return image_height, image_width, n_iter, batch_size


def __sample_opt_from_latent(context: execution_context.ExecutionContext, model, latent_image, steps):
    __update_context_checkpoints_model_base(context, model)
    image_height, image_width, n_iter, batch_size = __get_image_size(model, latent_image)
    return {
        'opt_type': 'ksampler',
        'width': image_width,
        'height': image_height,
        'steps': steps,
        'n_iter': n_iter,
        'batch_size': batch_size,
        "ratio": _sample_consumption_ratio(context, model)
    }


def _k_sampler_consumption(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                           denoise=1.0, context=None):
    context.set_geninfo(
        positive_prompt=positive,
        negative_prompt=negative,
        steps=steps,
        sampler=sampler_name,
        cfg_scale=cfg,
        seed=seed,
    )
    return {'opts': [__sample_opt_from_latent(context, model, latent_image, steps, )]}


def _reactor_restore_face_consumption(image, model, visibility, codeformer_weight, facedetection,
                                      context: execution_context.ExecutionContext):
    if model != 'none':
        opts = [{
            'opt_type': 'detect_face',
            'width': image.shape[2],
            'height': image.shape[1],
            'steps': 30,
            'n_iter': 1,
            'batch_size': image.shape[0]
        }, {
            'opt_type': 'swap_face',
            'width': image.shape[2],
            'height': image.shape[1],
            'steps': 30,
            'n_iter': 1,
            'batch_size': image.shape[0]
        }, {
            'opt_type': 'restore_face',
            'width': image.shape[2],
            'height': image.shape[1],
            'steps': 30,
            'n_iter': 1,
            'batch_size': image.shape[0]
        }]
    else:
        opts = []
    return {'opts': opts}


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
        'opts': [{
            'opt_type': 'detect_face',
            'width': input_image.shape[2],
            'height': input_image.shape[1],
            'batch_size': input_image.shape[0]
        }, {
            'opt_type': 'restore_face',
            'width': input_image.shape[2],
            'height': input_image.shape[1],
            'batch_size': input_image.shape[0]
        }]
    }


def _reactor_face_swap_opt_consumption(enabled, input_image, swap_model, facedetection, face_restore_model,
                                       face_restore_visibility, codeformer_weight, source_image=None, face_model=None,
                                       options=None, context: execution_context.ExecutionContext = None):
    return {
        'opts': [{
            'opt_type': 'detect_face',
            'width': input_image.shape[2],
            'height': input_image.shape[1],
            'batch_size': input_image.shape[0]
        }, {
            'opt_type': 'restore_face',
            'width': input_image.shape[2],
            'height': input_image.shape[1],
            'batch_size': input_image.shape[0]
        }]
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
    context.set_geninfo(
        positive_prompt=positive,
        negative_prompt=negative,
        steps=steps,
        sampler=sampler_name,
        cfg_scale=cfg,
        seed=noise_seed,
    )
    return {'opts': [__sample_opt_from_latent(context, model, latent_image, steps, )]}


def _tsc_ksampler_advanced_consumption(model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive,
                                       negative,
                                       latent_image, start_at_step, end_at_step, return_with_leftover_noise,
                                       preview_method, vae_decode,
                                       prompt=None, extra_pnginfo=None, my_unique_id=None,
                                       context: execution_context.ExecutionContext = None,
                                       optional_vae=(None,), script=None):
    context.set_geninfo(
        positive_prompt=positive,
        negative_prompt=negative,
        steps=steps,
        sampler=sampler_name,
        cfg_scale=cfg,
        seed=noise_seed,
    )
    return {'opts': [__sample_opt_from_latent(context, model, latent_image, steps, )]}


def _tsc_ksampler_sdxl_consumption(sdxl_tuple, noise_seed, steps, cfg, sampler_name, scheduler, latent_image,
                                   start_at_step, refine_at_step, preview_method, vae_decode, prompt=None,
                                   extra_pnginfo=None,
                                   my_unique_id=None, context: execution_context.ExecutionContext = None,
                                   optional_vae=(None,), refiner_extras=None,
                                   script=None):
    context.set_geninfo(
        positive_prompt=prompt,
        negative_prompt=None,
        steps=steps,
        sampler=sampler_name,
        cfg_scale=cfg,
        seed=noise_seed,
    )
    model = sdxl_tuple[0]
    return {'opts': [__sample_opt_from_latent(context, model, latent_image, steps, )]}


def _tsc_k_sampler_consumption(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                               preview_method, vae_decode, denoise=1.0, prompt=None, extra_pnginfo=None,
                               my_unique_id=None,
                               context: execution_context.ExecutionContext = None,
                               optional_vae=(None,), script=None, add_noise=None, start_at_step=None, end_at_step=None,
                               return_with_leftover_noise=None, sampler_type="regular"):
    context.set_geninfo(
        positive_prompt=positive,
        negative_prompt=negative,
        steps=steps,
        sampler=sampler_name,
        cfg_scale=cfg,
        seed=seed,
    )
    return {'opts': [__sample_opt_from_latent(context, model, latent_image, steps, )]}


def _xlabs_sampler_consumption(model, conditioning, neg_conditioning,
                               noise_seed, steps, timestep_to_start_cfg, true_gs,
                               image_to_image_strength, denoise_strength,
                               latent_image=None, controlnet_condition=None,
                               context=None):
    context.set_geninfo(
        positive_prompt=conditioning,
        negative_prompt=neg_conditioning,
        steps=steps,
        sampler="",
        cfg_scale=timestep_to_start_cfg,
        seed=noise_seed,
    )
    return {'opts': [__sample_opt_from_latent(context, model, latent_image, steps, )]}


def _impact_k_sampler_basic_pipe_consumption(basic_pipe, seed, steps, cfg, sampler_name, scheduler, latent_image,
                                             denoise=1.0, context=None):
    model, clip, vae, positive, negative = basic_pipe

    context.set_geninfo(
        positive_prompt=positive,
        negative_prompt=negative,
        steps=steps,
        sampler=sampler_name,
        cfg_scale=cfg,
        seed=seed,
    )

    return {'opts': [__sample_opt_from_latent(context, model, latent_image, steps, )]}


def _tiled_k_sampler_consumption(model, seed, tile_width, tile_height, tiling_strategy, steps, cfg, sampler_name,
                                 scheduler, positive, negative, latent_image, denoise, context=None):
    context.set_geninfo(
        positive_prompt=positive,
        negative_prompt=negative,
        steps=steps,
        sampler=sampler_name,
        cfg_scale=cfg,
        seed=seed,
    )
    return {'opts': [__sample_opt_from_latent(context, model, latent_image, steps, )]}


def _easy_full_k_sampler_consumption(pipe, steps, cfg, sampler_name, scheduler, denoise, image_output, link_id,
                                     save_prefix, seed=None, model=None, positive=None, negative=None, latent=None,
                                     vae=None, clip=None, xyPlot=None, tile_size=None, prompt=None, extra_pnginfo=None,
                                     my_unique_id=None, context: execution_context.ExecutionContext = None,
                                     force_full_denoise=False, disable_noise=False, downscale_options=None, image=None):
    samp_samples = latent if latent is not None else pipe["samples"]
    samp_vae = vae if vae is not None else pipe["vae"]
    if image is not None and latent is None:
        samp_samples = {"samples": samp_vae.encode(image[:, :, :, :3])}
    samp_model = model if model is not None else pipe["model"]
    context.set_geninfo(
        positive_prompt=positive,
        negative_prompt=negative,
        steps=steps,
        sampler=sampler_name,
        cfg_scale=cfg,
        seed=seed,
    )
    return {'opts': [__sample_opt_from_latent(context, samp_model, samp_samples, steps, )]}


def _mochi_sampler_consumption(model, positive, negative, steps, cfg, seed, height, width, num_frames,
                               cfg_schedule=None, opt_sigmas=None, samples=None, fastercache=None,
                               context: execution_context.ExecutionContext = None):
    context.set_geninfo(
        positive_prompt=positive,
        negative_prompt=negative,
        steps=steps,
        sampler=samples,
        cfg_scale=cfg,
        seed=seed,
    )
    return {
        'opts': [
            {
                'opt_type': 'mochi_sampler',
                'width': width,
                'height': height,
                'steps': steps,
                'n_iter': num_frames,
                'batch_size': 1,
                "ratio": _sample_consumption_ratio(context, model)
            }
        ]
    }


def _cog_video_sampler_consumption(model, positive, negative, steps, cfg, seed, scheduler, num_frames, samples=None,
                                   denoise_strength=1.0, image_cond_latents=None, context_options=None, controlnet=None,
                                   tora_trajectory=None, fastercache=None,
                                   context: execution_context.ExecutionContext = None):
    context.set_geninfo(
        positive_prompt=positive,
        negative_prompt=negative,
        steps=steps,
        sampler=samples,
        cfg_scale=cfg,
        seed=seed,
    )

    H = 768
    W = 768
    B = 1
    if samples is not None:
        if len(samples["samples"].shape) == 5:
            B, T, C, H, W = samples["samples"].shape
        if len(samples["samples"].shape) == 4:
            B, C, H, W = samples["samples"].shape
    if image_cond_latents is not None:
        B, T, C, H, W = image_cond_latents["samples"].shape
    height = H * 8
    width = W * 8
    return {
        'opts': [
            {
                'opt_type': 'cog_video_sampler',
                'width': width,
                'height': height,
                'steps': steps,
                'n_iter': num_frames,
                'batch_size': B,
                "ratio": _sample_consumption_ratio(context, model)
            }
        ]
    }


def _tiled_k_sampler_advanced_consumption(model, add_noise, noise_seed, tile_width, tile_height, tiling_strategy, steps,
                                          cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step,
                                          end_at_step, return_with_leftover_noise, preview, denoise=1.0,
                                          context: execution_context.ExecutionContext = None):
    context.set_geninfo(
        positive_prompt=positive,
        negative_prompt=negative,
        steps=steps,
        sampler=sampler_name,
        cfg_scale=cfg,
        seed=noise_seed,
    )
    return {'opts': [__sample_opt_from_latent(context, model, latent_image, steps, )]}


def _sampler_custom_consumption(model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image,
                                context):
    steps = len(sigmas)
    context.set_geninfo(
        positive_prompt=positive,
        negative_prompt=negative,
        steps=steps,
        sampler=sampler,
        cfg_scale=cfg,
        seed=noise_seed,
    )
    return {'opts': [__sample_opt_from_latent(context, model, latent_image, steps, )]}


def _sampler_custom_advanced_consumption(noise, guider, sampler, sigmas, latent_image, context):
    steps = len(sigmas)
    return {'opts': [__sample_opt_from_latent(context, guider.model_patcher, latent_image, steps, )]}


def _k_sampler_inspire_consumption(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                   denoise, noise_mode, batch_seed_mode="comfy", variation_seed=None,
                                   variation_strength=None, variation_method="linear",
                                   context=None):
    context.set_geninfo(
        positive_prompt=positive,
        negative_prompt=negative,
        steps=steps,
        sampler=sampler_name,
        cfg_scale=cfg,
        seed=seed,
    )
    return {'opts': [__sample_opt_from_latent(context, model, latent_image, steps, )]}


def _was_k_sampler_cycle_consumption(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                     tiled_vae, latent_upscale, upscale_factor,
                                     upscale_cycles, starting_denoise, cycle_denoise, scale_denoise, scale_sampling,
                                     vae, secondary_model=None, secondary_start_cycle=None,
                                     pos_additive=None, pos_add_mode=None, pos_add_strength=None,
                                     pos_add_strength_scaling=None, pos_add_strength_cutoff=None,
                                     neg_additive=None, neg_add_mode=None, neg_add_strength=None,
                                     neg_add_strength_scaling=None, neg_add_strength_cutoff=None,
                                     upscale_model=None, processor_model=None, sharpen_strength=0, sharpen_radius=2,
                                     steps_scaling=None, steps_control=None,
                                     steps_scaling_value=None, steps_cutoff=None, denoise_cutoff=0.25,
                                     context=None):
    context.set_geninfo(
        positive_prompt=positive,
        negative_prompt=negative,
        steps=steps,
        sampler=sampler_name,
        cfg_scale=cfg,
        seed=seed,
    )

    result = []
    upscale_steps = upscale_cycles
    division_factor = upscale_steps if steps >= upscale_steps else steps
    current_upscale_factor = upscale_factor ** (1 / (division_factor - 1))
    n_iter = latent_image.get("batch_index", 1)

    latent = latent_image["samples"]
    latent_size = latent.size()
    batch_size = latent_size[0]
    latent_image_height = latent_size[2] * 8
    latent_image_width = latent_size[3] * 8

    for i in range(division_factor):
        if steps_scaling and i > 0:
            steps = (
                steps + steps_scaling_value
                if steps_control == 'increment'
                else steps - steps_scaling_value
            )
            steps = (
                (steps
                 if steps <= steps_cutoff
                 else steps_cutoff)
                if steps_control == 'increment'
                else (steps
                      if steps >= steps_cutoff
                      else steps_cutoff)
            )
        result.append({
            'opt_type': 'sample',
            'width': latent_image_width,
            'height': latent_image_height,
            'steps': steps,
            'n_iter': n_iter,
            'batch_size': batch_size,
            "ratio": _sample_consumption_ratio(context, model)
        })
        if i < division_factor - 1 and latent_upscale == 'disable':
            if processor_model:
                scale_factor = _get_upscale_model_size(context, processor_model)
                result.append({
                    'opt_type': 'upscale',
                    'width': latent_image_width * scale_factor,
                    'height': latent_image_height * scale_factor,
                })
            if upscale_model:
                scale_factor = _get_upscale_model_size(context, upscale_model)
                result.append({
                    'opt_type': 'upscale',
                    'width': latent_image_width * scale_factor,
                    'height': latent_image_height * scale_factor,
                })
                latent_image_width = int(round(round(latent_image_width * current_upscale_factor) / 32) * 32)
                latent_image_height = int(round(round(latent_image_height * current_upscale_factor) / 32) * 32)
        else:
            latent_image_height *= current_upscale_factor
            latent_image_width *= current_upscale_factor

    return {'opts': result}


def _searge_sdxl_image2image_sampler2_consumption(base_model, base_positive, base_negative, refiner_model,
                                                  refiner_positive, refiner_negative,
                                                  image, vae, noise_seed, steps, cfg, sampler_name, scheduler,
                                                  base_ratio, denoise, softness,
                                                  upscale_model=None, scaled_width=None, scaled_height=None,
                                                  noise_offset=None, refiner_strength=None,
                                                  context: execution_context.ExecutionContext = None):
    context.set_geninfo(
        positive_prompt=base_positive,
        negative_prompt=base_negative,
        steps=steps,
        sampler=sampler_name,
        cfg_scale=cfg,
    )

    result = []

    if steps < 1:
        return result

    if upscale_model is not None and softness < 0.9999:
        use_upscale_model = True
        model_scale = _get_upscale_model_size(context, upscale_model)
    else:
        use_upscale_model = False
        model_scale = 1

    image_width = image.shape[2]
    image_height = image.shape[1]
    batch_size = image.shape[0]
    if use_upscale_model:
        result.append({
            'opt_type': 'upscale',
            'width': image_width * model_scale,
            'height': image_height * model_scale,
            'batch_size': batch_size,
        })

    if denoise < 0.01:
        return result

    n_iter = 1
    if scaled_width is not None and scaled_height is not None:
        sample_height = scaled_height
        sample_width = scaled_width
    elif use_upscale_model:
        sample_height = image_height * model_scale
        sample_width = image_width * model_scale
    else:
        sample_height = image_height
        sample_width = image_width

    result.append({
        'opt': 'sample',
        'width': sample_width,
        'height': sample_height,
        'steps': steps,
        'n_iter': n_iter,
        'batch_size': batch_size,
        'ratio': _sample_consumption_ratio(context, base_model)
    })
    return {'opts': result}


def _ultimate_sd_upscale_consumption(image, model, positive, negative, vae, upscale_by, seed,
                                     steps, cfg, sampler_name, scheduler, denoise, upscale_model,
                                     mode_type, tile_width, tile_height, mask_blur, tile_padding,
                                     seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                                     seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode,
                                     context: execution_context.ExecutionContext):
    context.set_geninfo(
        positive_prompt=positive,
        negative_prompt=negative,
        steps=steps,
        sampler=sampler_name,
        cfg_scale=cfg,
        seed=seed,
    )

    batch_size = image.shape[0]
    image_width = image.shape[2]
    image_height = image.shape[1]

    if upscale_model is not None:
        enable_hr = True
        hr_width = image_width * upscale_model.scale
        hr_height = image_height * upscale_model.scale
    else:
        enable_hr = False
        hr_width = 0
        hr_height = 0

    redraw_width = math.ceil((image_width * upscale_by) / 64) * 64
    redraw_height = math.ceil((image_height * upscale_by) / 64) * 64
    result = [{
        'opt_type': 'sample',
        'width': redraw_width,
        'height': redraw_height,
        'steps': steps,
        'n_iter': 1,
        'batch_size': batch_size,
        'ratio': _sample_consumption_ratio(context, model)
    }]
    if enable_hr:
        result.append({
            'opt_type': 'hires_fix',
            'width': hr_width,
            'height': hr_height,
        })
    return {'opts': result}


def _image_upscale_with_model_consumption(upscale_model, image):
    return {
        'opts': [{
            'opt_type': 'upscale',
            'width': image.shape[2] * upscale_model.scale,
            'height': image.shape[1] * upscale_model.scale,
            'batch_size': image.shape[0],
        }]
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


def _get_upscale_model_size(context, model_name):
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
    return model_upscale_cache[model_name]


def _easy_hires_fix_consumption(
        model_name, rescale_after_model, rescale_method, rescale, percent, width, height,
        longer_side, crop, image_output, link_id, save_prefix, pipe=None, image=None, vae=None, prompt=None,
        extra_pnginfo=None, my_unique_id=None, context: execution_context.ExecutionContext = None):
    model_scale = _get_upscale_model_size(context, model_name)

    if pipe is not None:
        image = image if image is not None else pipe["images"]
    if image is not None:
        return {
            'opts': {
                'opt_type': 'hires_fix',
                'width': image.shape[2] * model_scale,
                'height': image.shape[1] * model_scale,
                'batch_size': image.shape[0],
            }
        }
    else:
        return {
        }


def _cr_upscale_image_consumption(image, upscale_model, rounding_modulus=8, loops=1, mode="rescale", supersample='true',
                                  resampling_method="lanczos", rescale_factor=2, resize_width=1024,
                                  context: execution_context.ExecutionContext = None):
    model_scale = _get_upscale_model_size(context, upscale_model)
    if image is not None:
        return {
            'opts': [{
                'opt_type': 'upscale',
                'width': image.shape[2] * model_scale,
                'height': image.shape[1] * model_scale,
                'batch_size': image.shape[0],
            }]
        }
    else:
        return {
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
        'opts': [{
            'opt_type': 'generate',
            'width': images.shape[2],
            'height': images.shape[1],
            'batch_size': images.shape[0],
        }]
    }


def _face_detailer_pipe_consumption(image, detailer_pipe, guide_size, guide_size_for, max_size, seed, steps, cfg,
                                    sampler_name, scheduler,
                                    denoise, feather, noise_mask, force_inpaint, bbox_threshold, bbox_dilation,
                                    bbox_crop_factor,
                                    sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion,
                                    sam_mask_hint_threshold, sam_mask_hint_use_negative, drop_size, refiner_ratio=None,
                                    cycle=1, inpaint_model=False, noise_mask_feather=0,
                                    context: execution_context.ExecutionContext = None, ):
    model, clip, vae, positive, negative, wildcard, bbox_detector, segm_detector_opt, sam_model_opt, detailer_hook, \
        refiner_model, refiner_clip, refiner_positive, refiner_negative = detailer_pipe

    image_width = image.shape[2]
    image_height = image.shape[1]
    batch_size = image.shape[0]
    opts = [{
        'opt_type': 'face_detector',
        'width': image_width,
        'height': image_height,
        'steps': steps,
        'n_iter': 1,
        'batch_size': batch_size,
    }]
    if sam_model_opt is not None:
        opts.append({
            'opt_type': 'sam',
            'width': image_width,
            'height': image_height,
            'steps': steps,
            'n_iter': 1,
            'batch_size': batch_size,
        })
    elif segm_detector_opt is not None:
        opts.append({
            'opt_type': 'sam',
            'width': image_width,
            'height': image_height,
            'steps': steps,
            'n_iter': 1,
            'batch_size': batch_size,
        })
    opts.append({
        'opt_type': 'face_enhance',
        'width': image_width,
        'height': image_height,
        'steps': steps,
        'n_iter': 1,
        'batch_size': batch_size,
    })

    return {'opts': opts}


def _face_detailer_consumption(image, model, clip, vae, guide_size, guide_size_for, max_size, seed, steps, cfg,
                               sampler_name, scheduler,
                               positive, negative, denoise, feather, noise_mask, force_inpaint,
                               bbox_threshold, bbox_dilation, bbox_crop_factor,
                               sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion,
                               sam_mask_hint_threshold,
                               sam_mask_hint_use_negative, drop_size, bbox_detector, wildcard, cycle=1,
                               sam_model_opt=None, segm_detector_opt=None, detailer_hook=None, inpaint_model=False,
                               noise_mask_feather=0,
                               context: execution_context.ExecutionContext = None):
    image_width = image.shape[2]
    image_height = image.shape[1]
    batch_size = image.shape[0]
    opts = [{
        'opt_type': 'face_detector',
        'width': image_width,
        'height': image_height,
        'steps': steps,
        'n_iter': 1,
        'batch_size': batch_size,
    }]
    if sam_model_opt is not None:
        opts.append({
            'opt_type': 'sam',
            'width': image_width,
            'height': image_height,
            'steps': steps,
            'n_iter': 1,
            'batch_size': batch_size,
        })
    elif segm_detector_opt is not None:
        opts.append({
            'opt_type': 'sam',
            'width': image_width,
            'height': image_height,
            'steps': steps,
            'n_iter': 1,
            'batch_size': batch_size,
        })
    opts.append({
        'opt_type': 'face_enhance',
        'width': image_width,
        'height': image_height,
        'steps': steps,
        'n_iter': 1,
        'batch_size': batch_size,
    })

    return {'opts': opts}


def _detailer_for_each_consumption(image, segs, model, clip, vae, guide_size, guide_size_for, max_size, seed, steps,
                                   cfg, sampler_name,
                                   scheduler, positive, negative, denoise, feather, noise_mask, force_inpaint, wildcard,
                                   cycle=1,
                                   detailer_hook=None, inpaint_model=False, noise_mask_feather=0,
                                   context: execution_context.ExecutionContext = None):
    image_width = image.shape[2]
    image_height = image.shape[1]
    batch_size = image.shape[0]
    opts = [{
        'opt_type': 'enhance_detail',
        'width': image_width,
        'height': image_height,
        'steps': steps,
        'n_iter': cycle,
        'batch_size': batch_size,
    }]
    return {'opts': opts}


def _detailer_for_each_pipe_consumption(image, segs, guide_size, guide_size_for, max_size, seed, steps, cfg,
                                        sampler_name, scheduler,
                                        denoise, feather, noise_mask, force_inpaint, basic_pipe, wildcard,
                                        refiner_ratio=None, detailer_hook=None, refiner_basic_pipe_opt=None,
                                        cycle=1, inpaint_model=False, noise_mask_feather=0,
                                        context: execution_context.ExecutionContext = None):
    image_width = image.shape[2]
    image_height = image.shape[1]
    batch_size = image.shape[0]
    opts = [{
        'opt_type': 'enhance_detail',
        'width': image_width,
        'height': image_height,
        'steps': steps,
        'n_iter': cycle,
        'batch_size': batch_size,
    }]
    return {'opts': opts}


def _impact_simple_detector_segs_for_ad_consumption(bbox_detector, image_frames, bbox_threshold, bbox_dilation,
                                                    crop_factor, drop_size,
                                                    sub_threshold, sub_dilation, sub_bbox_expansion,
                                                    sam_mask_hint_threshold,
                                                    masking_mode="Pivot SEGS", segs_pivot="Combined mask",
                                                    sam_model_opt=None, segm_detector_opt=None):
    image_width = image_frames.shape[2]
    image_height = image_frames.shape[1]
    batch_size = image_frames.shape[0]
    opts = [{
        'opt_type': 'detect_box',
        'width': image_width,
        'height': image_height,
        'batch_size': batch_size,
    }]
    if sam_model_opt is not None:
        opts.append({
            'opt_type': 'sam',
            'width': image_width,
            'height': image_height,
            'batch_size': batch_size,
        })
    elif segm_detector_opt is not None:
        opts.append({
            'opt_type': 'detect_seg',
            'width': image_width,
            'height': image_height,
            'batch_size': batch_size,
        })
    return {'opts': opts}


def _re_actor_build_face_model_consumption(image, det_size=(640, 640)):
    return {
        'opts': [{
            'opt_type': 'build_face_model',
            'width': det_size[0],
            'height': det_size[1],
            'steps': 1,
            'n_iter': 1,
            'batch_size': 1
        }]
    }


def _supir_decode_consumption(SUPIR_VAE, latents, use_tiled_vae, decoder_tile_size):
    opt = __sample_opt_from_latent(latents, 30)
    opt['opt_type'] = 'supir_decode'
    del opt['steps']
    return {'opts': [opt, ]}


def _supir_encode_consumption(SUPIR_VAE, image, encoder_dtype, use_tiled_vae, encoder_tile_size):
    image_width = image.shape[2]
    image_height = image.shape[1]
    batch_size = image.shape[0]
    opts = [{
        'opt_type': 'supir_encode',
        'width': image_width,
        'height': image_height,
        'batch_size': batch_size,
    }]
    return {'opts': opts}


def _supir_sample_consumption(SUPIR_model, latents, steps, seed, cfg_scale_end, EDM_s_churn, s_noise, positive,
                              negative,
                              cfg_scale_start, control_scale_start, control_scale_end, restore_cfg, keep_model_loaded,
                              DPMPP_eta,
                              sampler, sampler_tile_size=1024, sampler_tile_stride=512):
    return {'opts': [__sample_opt_from_latent(latents, steps, )]}


def _supir_first_stage_consumption(SUPIR_VAE, image, encoder_dtype, use_tiled_vae, encoder_tile_size,
                                   decoder_tile_size):
    image_width = image.shape[2]
    image_height = image.shape[1]
    batch_size = image.shape[0]
    opts = [{
        'opt_type': 'supir_first_stage',
        'width': image_width,
        'height': image_height,
        'batch_size': batch_size,
    }]
    return {'opts': opts}


def _bbox_detector_segs_consumption(bbox_detector, image, threshold, dilation, crop_factor, drop_size, labels=None,
                                    detailer_hook=None):
    image_width = image.shape[2]
    image_height = image.shape[1]
    batch_size = image.shape[0]
    opts = [{
        'opt_type': 'bbox_detector',
        'width': image_width,
        'height': image_height,
        'batch_size': batch_size,
    }]
    return {'opts': opts}


def _segm_detector_for_each_consumption(segm_detector, image, threshold, dilation, crop_factor, drop_size, labels=None,
                                        detailer_hook=None):
    image_width = image.shape[2]
    image_height = image.shape[1]
    batch_size = image.shape[0]
    opts = [{
        'opt_type': 'segm_detector',
        'width': image_width,
        'height': image_height,
        'batch_size': batch_size,
    }]
    return {'opts': opts}


def _impact_simple_detector_segs_consumption(bbox_detector, image, bbox_threshold, bbox_dilation, crop_factor,
                                             drop_size,
                                             sub_threshold, sub_dilation, sub_bbox_expansion,
                                             sam_mask_hint_threshold, post_dilation=0, sam_model_opt=None,
                                             segm_detector_opt=None):
    image_width = image.shape[2]
    image_height = image.shape[1]
    batch_size = image.shape[0]
    opts = [{
        'opt_type': 'detector_segs',
        'width': image_width,
        'height': image_height,
        'batch_size': batch_size,
    }]
    return {'opts': opts}


def _clip_seg_masking_consumption(image, text=None, clipseg_model=None):
    image_width = image.shape[2]
    image_height = image.shape[1]
    batch_size = image.shape[0]
    opts = [{
        'opt_type': 'clip_seg_masking',
        'width': image_width,
        'height': image_height,
        'batch_size': batch_size,
    }]
    return {'opts': opts}


def _layermask_person_mask_ultra_consumption(images, face, hair, body, clothes,
                                             accessories, background, confidence,
                                             detail_range, black_point, white_point, process_detail):
    image_width = images.shape[2]
    image_height = images.shape[1]
    batch_size = images.shape[0]
    opts = [{
        'opt_type': 'layermask_person_mask_ultra',
        'width': image_width,
        'height': image_height,
        'batch_size': batch_size,
    }]
    return {'opts': opts}


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


def _ultimate_sd_upscale_no_upscale_consumption(upscaled_image, model, positive, negative, vae, seed,
                                                steps, cfg, sampler_name, scheduler, denoise,
                                                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                                                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                                                seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode,
                                                context: execution_context.ExecutionContext):
    return {
        'opts': [{
            'opt_type': 'upscale',
            'width': upscaled_image.shape[2],
            'height': upscaled_image.shape[1],
            'steps': steps,
            'n_iter': 1,
            'batch_size': upscaled_image.shape[0],
            'ratio': _sample_consumption_ratio(context, model)
        }]
    }


def image_rembg_consumption(
        images,
        transparency=True,
        model="u2net",
        alpha_matting=False,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10,
        post_processing=False,
        only_mask=False,
        background_color="none",
):
    image_width = images.shape[2]
    image_height = images.shape[1]
    batch_size = images.shape[0]
    opts = [{
        'opt_type': 'rembg',
        'width': image_width,
        'height': image_height,
        'batch_size': batch_size,
    }]
    return {'opts': opts}


def _sam_detector_combined_consumption(
        sam_model, segs, image, detection_hint, dilation,
        threshold, bbox_expansion, mask_hint_threshold, mask_hint_use_negative):
    image_width = image.shape[2]
    image_height = image.shape[1]
    batch_size = image.shape[0]
    opts = [{
        'opt_type': 'segm_detector_combined',
        'width': image_width,
        'height': image_height,
        'batch_size': batch_size,
    }]
    return {'opts': opts}


def was_remove_background_consumption(images, mode='background', threshold=127, threshold_tolerance=2):
    image_width = images.shape[2]
    image_height = images.shape[1]
    batch_size = images.shape[0]
    opts = [{
        'opt_type': 'was_remove_background',
        'width': image_width,
        'height': image_height,
        'batch_size': batch_size,
    }]
    return {'opts': opts}


def _default_consumption_maker(*args, **kwargs):
    return {}


def _none_consumption_maker(*args, **kwargs):
    return None


_NODE_CONSUMPTION_MAPPING = {
    'KSampler': _k_sampler_consumption,
    'KSamplerAdvanced': _k_sampler_advanced_consumption,
    'KSampler (Efficient)': _tsc_k_sampler_consumption,
    'KSampler Adv. (Efficient)': _tsc_ksampler_advanced_consumption,
    'KSampler SDXL (Eff.)': _tsc_ksampler_sdxl_consumption,
    'XlabsSampler': _xlabs_sampler_consumption,
    'ImpactKSamplerBasicPipe': _impact_k_sampler_basic_pipe_consumption,
    'ReActorRestoreFace': _reactor_restore_face_consumption,
    'ReActorFaceSwap': _reactor_face_swap_consumption,
    'ReActorFaceSwapOpt': _reactor_face_swap_opt_consumption,
    'ReActorBuildFaceModel': _re_actor_build_face_model_consumption,
    'ImageUpscaleWithModel': _image_upscale_with_model_consumption,
    'UltimateSDUpscale': _ultimate_sd_upscale_consumption,
    'easy hiresFix': _easy_hires_fix_consumption,
    'VHS_VideoCombine': _vhs_video_combine_consumption,
    'FaceDetailer': _face_detailer_consumption,
    'FaceDetailerPipe': _face_detailer_pipe_consumption,
    'SamplerCustom': _sampler_custom_consumption,
    'SamplerCustomAdvanced': _sampler_custom_advanced_consumption,
    'SeargeSDXLImage2ImageSampler2': _searge_sdxl_image2image_sampler2_consumption,
    'BNK_TiledKSamplerAdvanced': _tiled_k_sampler_advanced_consumption,
    'BNK_TiledKSampler': _tiled_k_sampler_consumption,
    'easy fullkSampler': _easy_full_k_sampler_consumption,
    "CogVideoSampler": _cog_video_sampler_consumption,
    "MochiSampler": _mochi_sampler_consumption,

    'UltimateSDUpscaleNoUpscale': _ultimate_sd_upscale_no_upscale_consumption,
    'CR Upscale Image': _cr_upscale_image_consumption,
    'KSampler //Inspire': _k_sampler_inspire_consumption,
    'KSampler Cycle': _was_k_sampler_cycle_consumption,
    'ImpactSimpleDetectorSEGS_for_AD': _impact_simple_detector_segs_for_ad_consumption,
    'DetailerForEach': _detailer_for_each_consumption,
    'DetailerForEachPipe': _detailer_for_each_pipe_consumption,
    'SUPIR_decode': _supir_decode_consumption,
    'SUPIR_encode': _supir_encode_consumption,
    'SUPIR_sample': _supir_sample_consumption,
    'SUPIR_first_stage': _supir_first_stage_consumption,
    'BboxDetectorSEGS': _bbox_detector_segs_consumption,
    'ONNXDetectorSEGS': _bbox_detector_segs_consumption,
    'SegmDetectorSEGS': _segm_detector_for_each_consumption,
    'ImpactSimpleDetectorSEGS': _impact_simple_detector_segs_consumption,
    'CLIPSeg Masking': _clip_seg_masking_consumption,
    'LayerMask: PersonMaskUltra': _layermask_person_mask_ultra_consumption,
    'Image Rembg (Remove Background)': image_rembg_consumption,
    'SAMDetectorCombined': _sam_detector_combined_consumption,
    'Image Remove Background (Alpha)': was_remove_background_consumption,

    'ADE_UseEvolvedSampling': _none_consumption_maker,
    'ModelSamplingSD3': _none_consumption_maker,
    'HighRes-Fix Script': _none_consumption_maker,
    'ImageBatch': _none_consumption_maker,
    'ControlNetApplyAdvanced': _none_consumption_maker,
    'ReActorLoadFaceModel': _none_consumption_maker,
    'SVD_img2vid_Conditioning': _none_consumption_maker,
    'VideoLinearCFGGuidance': _none_consumption_maker,
    'ConstrainImage|pysssss': _none_consumption_maker,
    'MiDaS-DepthMapPreprocessor': _none_consumption_maker,
    'VHS_BatchManager': _none_consumption_maker,
    'easy ultralyticsDetectorPipe': _none_consumption_maker,
    'ColorPreprocessor': _none_consumption_maker,
    'DWPreprocessor': _none_consumption_maker,
    'FreeU_V2': _none_consumption_maker,
    'ImageInvert': _none_consumption_maker,
    'XY Plot': _none_consumption_maker,
    'ApplyInstantID': _none_consumption_maker,
    'CR Apply Multi-ControlNet': _none_consumption_maker,
    'CR Multi-ControlNet Stack': _none_consumption_maker,
    'AnimeLineArtPreprocessor': _none_consumption_maker,
    'InstantIDFaceAnalysis': _none_consumption_maker,
    'IPAdapterAdvanced': _none_consumption_maker,
    'IPAdapter': _none_consumption_maker,
    'RepeatLatentBatch': _none_consumption_maker,
    'OpenposePreprocessor': _none_consumption_maker,
    'ADE_AnimateDiffSamplingSettings': _none_consumption_maker,
    'ADE_StandardStaticContextOptions': _none_consumption_maker,
    'SaveImage': _none_consumption_maker,
    'VAEDecode': _none_consumption_maker,
    'CLIPTextEncode': _none_consumption_maker,
    'LoraLoader': _none_consumption_maker,
    'CheckpointLoaderSimple': _none_consumption_maker,
    'VAEEncode': _none_consumption_maker,
    'Image Resize': _none_consumption_maker,
    'EmptyLatentImage': _none_consumption_maker,
    'ImageScale': _none_consumption_maker,
    'CLIPSetLastLayer': _none_consumption_maker,
    'LoadImage': _none_consumption_maker,
    'easy promptReplace': _none_consumption_maker,
    'Text Multiline': _none_consumption_maker,
    'VAELoader': _none_consumption_maker,
    'ConditioningSetTimestepRange': _none_consumption_maker,
    'UpscaleModelLoader': _none_consumption_maker,
    'LoraLoader|pysssss': _none_consumption_maker,
    'RebatchLatents': _none_consumption_maker,
    'LatentBatchSeedBehavior': _none_consumption_maker,
    'Efficient Loader': _none_consumption_maker,
    'SDXLPromptStyler': _none_consumption_maker,
    'ConditioningCombine': _none_consumption_maker,
    'ConditioningZeroOut': _none_consumption_maker,
    'TripleCLIPLoader': _none_consumption_maker,
    'LatentUpscale': _none_consumption_maker,
    'easy stylesSelector': _none_consumption_maker,
    'ComfyUIStyler': _none_consumption_maker,
    'CLIPTextEncodeSDXLRefiner': _none_consumption_maker,
    'easy ipadapterApply': _none_consumption_maker,
    'ArtistStyler': _none_consumption_maker,
    'FantasyStyler': _none_consumption_maker,
    'ADE_AnimateDiffLoaderGen1': _none_consumption_maker,
    'AestheticStyler': _none_consumption_maker,
    'ControlNetLoader': _none_consumption_maker,
    'SaveAnimatedWEBP': _none_consumption_maker,
    'CLIPTextEncodeSDXL': _none_consumption_maker,
    'ImageScaleBy': _none_consumption_maker,
    'ImageOnlyCheckpointLoader': _none_consumption_maker,
    'IPAdapterUnifiedLoader': _none_consumption_maker,
    'EnvironmentStyler': _none_consumption_maker,
    'MilehighStyler': _none_consumption_maker,
    'AnimeStyler': _none_consumption_maker,
    'ADE_AnimateDiffLoaderWithContext': _none_consumption_maker,
    'VHS_LoadVideo': _none_consumption_maker,
    'MoodStyler': _none_consumption_maker,
    'ReActorSaveFaceModel': _none_consumption_maker,
    'Camera_AnglesStyler': _none_consumption_maker,
    'TimeofdayStyler': _none_consumption_maker,
    'FaceStyler': _none_consumption_maker,
    'Breast_StateStyler': _none_consumption_maker,
    'easy seed': _none_consumption_maker,
    'EmptySD3LatentImage': _none_consumption_maker,
    'UltralyticsDetectorProvider': _none_consumption_maker,
    'CR Apply LoRA Stack': _none_consumption_maker,
    'Upscale Model Loader': _none_consumption_maker,
    'PortraitMaster': _none_consumption_maker,
    'PlaySound|pysssss': _none_consumption_maker,
    'WD14Tagger|pysssss': _none_consumption_maker,
    'SAMLoader': _none_consumption_maker,
    'ADE_AnimateDiffUniformContextOptions': _none_consumption_maker,
    'ToBasicPipe': _none_consumption_maker,
    'easy pipeOut': _none_consumption_maker,
    'easy pipeIn': _none_consumption_maker,
    'CR LoRA Stack': _none_consumption_maker,
    'InstantIDModelLoader': _none_consumption_maker,
    'LatentUpscaleBy': _none_consumption_maker,
    'ToDetailerPipe': _none_consumption_maker,
    'easy ipadapterStyleComposition': _none_consumption_maker,
    'Canny': _none_consumption_maker,
    'BaseModel_Loader_local': _none_consumption_maker,
    'CR Load LoRA': _none_consumption_maker,
    'SAM Model Loader': _none_consumption_maker,
    'CLIPLoader': _none_consumption_maker,
    'VHS_LoadImages': _none_consumption_maker,
    'easy fullLoader': _none_consumption_maker,
    'XY Input: Checkpoint': _none_consumption_maker,
    'MaskToImage': _none_consumption_maker,
    'CR Text Concatenate': _none_consumption_maker,
    'CR Text': _none_consumption_maker,
    'easy loadImageBase64': _none_consumption_maker,
    'easy clearCacheAll': _none_consumption_maker,
    'ShowText|pysssss': _none_consumption_maker,
    'ADE_AnimateDiffLoRALoader': _none_consumption_maker,
    'easy showTensorShape': _none_consumption_maker,
    'ConditioningConcat': _none_consumption_maker,
    'ConditioningAverage': _none_consumption_maker,
    'KSamplerSelect': _none_consumption_maker,
    'AlignYourStepsScheduler': _none_consumption_maker,
    'BasicScheduler': _none_consumption_maker,
    'ModelMergeSimple': _none_consumption_maker,
    'CLIPMergeSimple': _none_consumption_maker,
    'ControlNetApply': _none_consumption_maker,
    'ADE_LoadAnimateDiffModel': _none_consumption_maker,
    'LatentSwitch': _none_consumption_maker,
    'ADE_ApplyAnimateDiffModelSimple': _none_consumption_maker,
    'IPAdapterModelLoader': _none_consumption_maker,
    'ImageCrop': _none_consumption_maker,
    'CLIPSegDetectorProvider': _none_consumption_maker,
    'PreviewImage': _none_consumption_maker,
    'Eff. Loader SDXL': _none_consumption_maker,
    'Unpack SDXL Tuple': _none_consumption_maker,
    'Automatic CFG': _none_consumption_maker,
    'easy textSwitch': _none_consumption_maker,
    'ApplyInstantIDAdvanced': _none_consumption_maker,
    'easy a1111Loader': _none_consumption_maker,
    'Text to Conditioning': _none_consumption_maker,
    'SeargeSamplerInputs': _none_consumption_maker,
    'LineArtPreprocessor': _none_consumption_maker,
    'IPAdapterFaceID': _none_consumption_maker,
    'unCLIPCheckpointLoader': _none_consumption_maker,
    'UNETLoader': _none_consumption_maker,
    'ControlNetLoaderAdvanced': _none_consumption_maker,
    'EmptyImage': _none_consumption_maker,
    'ACN_AdvancedControlNetApply': _none_consumption_maker,
    'LoraLoaderModelOnly': _none_consumption_maker,
    'LoadAnimateDiffModelNode': _none_consumption_maker,
    'ADE_AnimateDiffKeyframe': _none_consumption_maker,
    'DiffControlNetLoader': _none_consumption_maker,
    'DiffControlNetLoaderAdvanced': _none_consumption_maker,
    'SDTurboScheduler': _none_consumption_maker,
    'Image Crop Face': _none_consumption_maker,
    'IPAdapterTiled': _none_consumption_maker,
    'SeargeInput1': _none_consumption_maker,
    'SeargeInput2': _none_consumption_maker,
    'SeargeInput3': _none_consumption_maker,
    'SeargeInput4': _none_consumption_maker,
    'SeargeInput5': _none_consumption_maker,
    'SeargeInput6': _none_consumption_maker,
    'SeargeInput7': _none_consumption_maker,
    'SeargeOutput1': _none_consumption_maker,
    'SeargeOutput2': _none_consumption_maker,
    'SeargeOutput3': _none_consumption_maker,
    'SeargeOutput4': _none_consumption_maker,
    'SeargeOutput5': _none_consumption_maker,
    'SeargeOutput6': _none_consumption_maker,
    'SeargeOutput7': _none_consumption_maker,
    'SeargeGenerated1': _none_consumption_maker,
    'SeargeVAELoader': _none_consumption_maker,
    'TilePreprocessor': _none_consumption_maker,
    'TTPlanet_TileGF_Preprocessor': _none_consumption_maker,
    'TTPlanet_TileSimple_Preprocessor': _none_consumption_maker,
    'IPAdapterNoise': _none_consumption_maker,
    'SetLatentNoiseMask': _none_consumption_maker,
    'easy loraStack': _none_consumption_maker,
    'ScaledSoftControlNetWeights': _none_consumption_maker,
    'ScaledSoftMaskedUniversalWeights': _none_consumption_maker,
    'SoftControlNetWeights': _none_consumption_maker,
    'CustomControlNetWeights': _none_consumption_maker,
    'SoftT2IAdapterWeights': _none_consumption_maker,
    'CustomT2IAdapterWeights': _none_consumption_maker,
    'ACN_DefaultUniversalWeights': _none_consumption_maker,
    'ACN_ReferencePreprocessor': _none_consumption_maker,
    'ACN_ReferenceControlNet': _none_consumption_maker,
    'ACN_ReferenceControlNetFnetune': _none_consumption_maker,
    'easy controlnetStack': _none_consumption_maker,
    'Control Net Stacker': _none_consumption_maker,
    'easy globalSeed': _none_consumption_maker,
    "easy positive": _none_consumption_maker,
    "easy negative": _none_consumption_maker,
    "easy wildcards": _none_consumption_maker,
    "easy prompt": _none_consumption_maker,
    "easy promptList": _none_consumption_maker,
    "easy promptLine": _none_consumption_maker,
    "easy promptConcat": _none_consumption_maker,
    "easy portraitMaster": _none_consumption_maker,
    'AIO_Preprocessor': _none_consumption_maker,
    'ImageResizeKJ': _none_consumption_maker,
    'ImagePadForOutpaint': _none_consumption_maker,
    'CannyEdgePreprocessor': _none_consumption_maker,
    'DepthAnythingPreprocessor': _none_consumption_maker,
    'Zoe_DepthAnythingPreprocessor': _none_consumption_maker,
    'IPAdapterInsightFaceLoader': _none_consumption_maker,
    'ReActorMaskHelper': _none_consumption_maker,
    'easy imageColorMatch': _none_consumption_maker,
    'Checkpoint Selector': _none_consumption_maker,
    'Save Image w/Metadata': _none_consumption_maker,
    'Sampler Selector': _none_consumption_maker,
    'Scheduler Selector': _none_consumption_maker,
    'Seed Generator': _none_consumption_maker,
    'String Literal': _none_consumption_maker,
    'Width/Height Literal': _none_consumption_maker,
    'Cfg Literal': _none_consumption_maker,
    'Int Literal': _none_consumption_maker,
    'ImpactImageBatchToImageList': _none_consumption_maker,
    'ImpactMakeImageList': _none_consumption_maker,
    'ImpactMakeImageBatch': _none_consumption_maker,
    'PhotoMakerLoader': _none_consumption_maker,
    'PhotoMakerEncode': _none_consumption_maker,
    'ImpactSEGSToMaskList': _none_consumption_maker,
    'GroundingDinoModelLoader (segment anything)': _none_consumption_maker,
    'VAEEncodeForInpaint': _none_consumption_maker,
    'ImpactWildcardProcessor': _none_consumption_maker,
    'ImpactWildcardEncode': _none_consumption_maker,
    'StringFunction|pysssss': _none_consumption_maker,
    'SAMModelLoader (segment anything)': _none_consumption_maker,
    "INTConstant": _none_consumption_maker,
    "FloatConstant": _none_consumption_maker,
    "StringConstant": _none_consumption_maker,
    "StringConstantMultiline": _none_consumption_maker,
    "CR Image Input Switch": _none_consumption_maker,
    "CR Image Input Switch (4 way)": _none_consumption_maker,
    "CR Latent Input Switch": _none_consumption_maker,
    "CR Conditioning Input Switch": _none_consumption_maker,
    "CR Clip Input Switch": _none_consumption_maker,
    "CR Model Input Switch": _none_consumption_maker,
    "CR ControlNet Input Switch": _none_consumption_maker,
    "CR VAE Input Switch": _none_consumption_maker,
    "CR Text Input Switch": _none_consumption_maker,
    "CR Text Input Switch (4 way)": _none_consumption_maker,
    "CR Switch Model and CLIP": _none_consumption_maker,
    'SeargeFloatConstant': _none_consumption_maker,
    "SeargeFloatPair": _none_consumption_maker,
    "SeargeFloatMath": _none_consumption_maker,
    'easy showAnything': _none_consumption_maker,
    'CLIPVisionEncode': _none_consumption_maker,
    'unCLIPConditioning': _none_consumption_maker,
    'SeargeDebugPrinter': _none_consumption_maker,
    'LayerUtility: SaveImagePlus': _none_consumption_maker,
    'DetailerForEachDebug': _none_consumption_maker,
    'GlobalSeed //Inspire': _none_consumption_maker,
    'VAEDecodeTiled': _none_consumption_maker,
    'VAEEncodeTiled': _none_consumption_maker,
    'DualCLIPLoader': _none_consumption_maker,
    'RandomNoise': _none_consumption_maker,
    'BasicGuider': _none_consumption_maker,
    'SDXLPromptStylerAdvanced': _none_consumption_maker,
    'PerturbedAttentionGuidance': _none_consumption_maker,
    'Anything Everywhere3': _none_consumption_maker,
    "CR SD1.5 Aspect Ratio": _none_consumption_maker,
    "CR SDXL Aspect Ratio": _none_consumption_maker,
    "CR Aspect Ratio": _none_consumption_maker,
    "CR Aspect Ratio Banners": _none_consumption_maker,
    "CR Aspect Ratio Social Media": _none_consumption_maker,
    "CR_Aspect Ratio For Print": _none_consumption_maker,
    'Text Concatenate': _none_consumption_maker,
    'ImageBatchMulti': _none_consumption_maker,
    'MeshGraphormer-DepthMapPreprocessor': _none_consumption_maker,
    'MeshGraphormer+ImpactDetector-DepthMapPreprocessor': _none_consumption_maker,
    'LeReS-DepthMapPreprocessor': _none_consumption_maker,
    'IPAdapterUnifiedLoaderFaceID': _none_consumption_maker,
    "ImageBlend": _none_consumption_maker,
    "ImageBlur": _none_consumption_maker,
    "ImageQuantize": _none_consumption_maker,
    "ImageSharpen": _none_consumption_maker,
    "ImageScaleToTotalPixels": _none_consumption_maker,
    'SD_4XUpscale_Conditioning': _none_consumption_maker,
    'Latent Input Switch': _none_consumption_maker,
    'FluxGuidance': _none_consumption_maker,
    'VAE Input Switch': _none_consumption_maker,
    "Logic Comparison OR": _none_consumption_maker,
    "Logic Comparison AND": _none_consumption_maker,
    "Logic Comparison XOR": _none_consumption_maker,
    'ImpactControlNetApplySEGS': _none_consumption_maker,
    'DWPreprocessor_Provider_for_SEGS //Inspire': _none_consumption_maker,
    'CLIPVisionLoader': _none_consumption_maker,
    'SUPIR_conditioner': _none_consumption_maker,
    'ImageResize+': _none_consumption_maker,
    'SUPIR_model_loader_v2': _none_consumption_maker,
    "SUPIR_Upscale": _none_consumption_maker,
    "SUPIR_model_loader": _none_consumption_maker,
    "SUPIR_tiles": _none_consumption_maker,
    "SUPIR_model_loader_v2_clip": _none_consumption_maker,
    'ColorMatch': _none_consumption_maker,
    'EmptySegs': _none_consumption_maker,
    'VHS_VideoInfo': _none_consumption_maker,
    'ModelSamplingFlux': _none_consumption_maker,
    'SplitSigmas': _none_consumption_maker,
    'SegsToCombinedMask': _none_consumption_maker,
    'VHS_DuplicateImages': _none_consumption_maker,
    'CLIPTextEncodeFlux': _none_consumption_maker,
    'GetImageSize+': _none_consumption_maker,
    'MathExpression|pysssss': _none_consumption_maker,
    'SeargeIntegerConstant': _none_consumption_maker,
    "SeargeIntegerPair": _none_consumption_maker,
    "SeargeIntegerMath": _none_consumption_maker,
    "SeargeIntegerScaler": _none_consumption_maker,
    'Seed': _none_consumption_maker,
    'FeatherMask': _none_consumption_maker,
    'CheckpointLoader|pysssss': _none_consumption_maker,
    'ImageColorMatch+': _none_consumption_maker,
    'PreviewDetailerHookProvider': _none_consumption_maker,
    'Image Analyze': _none_consumption_maker,
    'easy comfyLoader': _none_consumption_maker,
    'easy controlnetLoaderADV': _none_consumption_maker,
    'LoRALoader': _none_consumption_maker,
    'LoadCLIPSegModels+': _none_consumption_maker,
    'ApplyCLIPSeg+': _none_consumption_maker,
    'LayerMask: MaskPreview': _none_consumption_maker,
    'CLIPSeg Model Loader': _none_consumption_maker,
    'Text to Console': _none_consumption_maker,
    'FromBasicPipe': _none_consumption_maker,
    'Lora Loader': _none_consumption_maker,
    'SDXLEmptyLatentSizePicker+': _none_consumption_maker,
    'LayerColor: AutoAdjust': _none_consumption_maker,
    'LayerUtility: PurgeVRAM': _none_consumption_maker,
    'LoRA Stacker': _none_consumption_maker,
    'Text Parse A1111 Embeddings': _none_consumption_maker,
    'easy showSpentTime': _none_consumption_maker,
    'Latent Upscale by Factor (WAS)': _none_consumption_maker,
    'ImpactControlBridge': _none_consumption_maker,
    'LoRA Stack to String converter': _none_consumption_maker,
    'LatentFromBatch': _none_consumption_maker,
    'AnimateDiffModuleLoader': _none_consumption_maker,
    'Image Size to Number': _none_consumption_maker,
    'Constant Number': _none_consumption_maker,
    'Number Operation': _none_consumption_maker,
    'Image Bounds': _none_consumption_maker,
    'Inset Image Bounds': _none_consumption_maker,
    'Bounded Image Crop': _none_consumption_maker,
    'PreviewBridge': _none_consumption_maker,
    'ImageToMask': _none_consumption_maker,
    'ToBinaryMask': _none_consumption_maker,
    'Mask Smooth Region': _none_consumption_maker,
    'Mask Erode Region': _none_consumption_maker,
    'Mask Dilate Region': _none_consumption_maker,
    'MaskToSEGS': _none_consumption_maker,
    'ImpactSEGSOrderedFilter': _none_consumption_maker,
    'Image Blank': _none_consumption_maker,
    'Image Blend by Mask': _none_consumption_maker,
    'SubtractMask': _none_consumption_maker,
    'InvertMask': _none_consumption_maker,
    'ImpactDilateMask': _none_consumption_maker,
    'BitwiseAndMask': _none_consumption_maker,
    'Mask Crop Region': _none_consumption_maker,
    'Image Crop Location': _none_consumption_maker,
    'Image Select Channel': _none_consumption_maker,
    'Image Levels Adjustment': _none_consumption_maker,
    'Image Mix RGB Channels': _none_consumption_maker,
    'Image Filter Adjustments': _none_consumption_maker,
    'Image to Noise': _none_consumption_maker,
    'AnimateDiffSlidingWindowOptions': _none_consumption_maker,
    'CR Seed': _none_consumption_maker,
    'LayerFilter: SoftLight': _none_consumption_maker,
    'LayerColor: Color of Shadow & Highlight': _none_consumption_maker,
    'easy imageSize': _none_consumption_maker,
    'easy imageScaleDownBy': _none_consumption_maker,
    'CFGGuider': _none_consumption_maker,
    'SolidMask': _none_consumption_maker,
    'MaskComposite': _none_consumption_maker,
    'ConditioningSetMask': _none_consumption_maker,
    'Image Blending Mode': _none_consumption_maker,
    'MaskBlur+': _none_consumption_maker,
    'SamplerDPMPP_3M_SDE': _none_consumption_maker,
    'KarrasScheduler': _none_consumption_maker,
    'SamplerEulerCFGpp': _none_consumption_maker,
    'ADE_AnimateDiffCombine': _none_consumption_maker,
    'comfy.Seed (rgthree)': _none_consumption_maker,

    "CogVideoDecode": _none_consumption_maker,
    "CogVideoTextEncode": _none_consumption_maker,
    "CogVideoImageEncode": _none_consumption_maker,
    "CogVideoTextEncodeCombine": _none_consumption_maker,
    "CogVideoTransformerEdit": _none_consumption_maker,
    "CogVideoContextOptions": _none_consumption_maker,
    "CogVideoControlNet": _none_consumption_maker,
    "ToraEncodeTrajectory": _none_consumption_maker,
    "ToraEncodeOpticalFlow": _none_consumption_maker,
    "CogVideoXFasterCache": _none_consumption_maker,
    "CogVideoXFunResizeToClosestBucket": _none_consumption_maker,
    "CogVideoLatentPreview": _none_consumption_maker,
    "CogVideoXTorchCompileSettings": _none_consumption_maker,
    "CogVideoImageEncodeFunInP": _none_consumption_maker,

    "DownloadAndLoadMochiModel": _none_consumption_maker,
    "MochiDecode": _none_consumption_maker,
    "MochiTextEncode": _none_consumption_maker,
    "MochiModelLoader": _none_consumption_maker,
    "MochiVAELoader": _none_consumption_maker,
    "MochiVAEEncoderLoader": _none_consumption_maker,
    "MochiDecodeSpatialTiling": _none_consumption_maker,
    "MochiTorchCompileSettings": _none_consumption_maker,
    "MochiImageEncode": _none_consumption_maker,
    "MochiLatentPreview": _none_consumption_maker,
    "MochiSigmaSchedule": _none_consumption_maker,
    "MochiFasterCache": _none_consumption_maker,

    "Lora Loader (JPS)": _none_consumption_maker,
    "SDXL Resolutions (JPS)": _none_consumption_maker,
    "SDXL Basic Settings (JPS)": _none_consumption_maker,
    "SDXL Settings (JPS)": _none_consumption_maker,
    "Generation TXT IMG Settings (JPS)": _none_consumption_maker,
    "Crop Image Settings (JPS)": _none_consumption_maker,
    "ImageToImage Settings (JPS)": _none_consumption_maker,
    "CtrlNet CannyEdge Settings (JPS)": _none_consumption_maker,
    "CtrlNet ZoeDepth Settings (JPS)": _none_consumption_maker,
    "CtrlNet MiDaS Settings (JPS)": _none_consumption_maker,
    "CtrlNet OpenPose Settings (JPS)": _none_consumption_maker,
    "Revision Settings (JPS)": _none_consumption_maker,
    "IP Adapter Settings (JPS)": _none_consumption_maker,
    "IP Adapter Tiled Settings (JPS)": _none_consumption_maker,
    "InstantID Settings (JPS)": _none_consumption_maker,
    "Image Prepare Settings (JPS)": _none_consumption_maker,
    "InstantID Source Prepare Settings (JPS)": _none_consumption_maker,
    "InstantID Pose Prepare Settings (JPS)": _none_consumption_maker,
    "InstantID Mask Prepare Settings (JPS)": _none_consumption_maker,
    "Sampler Scheduler Settings (JPS)": _none_consumption_maker,
    "Integer Switch (JPS)": _none_consumption_maker,
    "Image Switch (JPS)": _none_consumption_maker,
    "Latent Switch (JPS)": _none_consumption_maker,
    "Conditioning Switch (JPS)": _none_consumption_maker,
    "Model Switch (JPS)": _none_consumption_maker,
    "IPA Switch (JPS)": _none_consumption_maker,
    "VAE Switch (JPS)": _none_consumption_maker,
    "Mask Switch (JPS)": _none_consumption_maker,
    "ControlNet Switch (JPS)": _none_consumption_maker,
    "Disable Enable Switch (JPS)": _none_consumption_maker,
    "Enable Disable Switch (JPS)": _none_consumption_maker,
    "SDXL Basic Settings Pipe (JPS)": _none_consumption_maker,
    "SDXL Settings Pipe (JPS)": _none_consumption_maker,
    "Crop Image Pipe (JPS)": _none_consumption_maker,
    "ImageToImage Pipe (JPS)": _none_consumption_maker,
    "CtrlNet CannyEdge Pipe (JPS)": _none_consumption_maker,
    "CtrlNet ZoeDepth Pipe (JPS)": _none_consumption_maker,
    "CtrlNet MiDaS Pipe (JPS)": _none_consumption_maker,
    "CtrlNet OpenPose Pipe (JPS)": _none_consumption_maker,
    "IP Adapter Settings Pipe (JPS)": _none_consumption_maker,
    "IP Adapter Tiled Settings Pipe (JPS)": _none_consumption_maker,
    "InstantID Pipe (JPS)": _none_consumption_maker,
    "Image Prepare Pipe (JPS)": _none_consumption_maker,
    "InstantID Source Prepare Pipe (JPS)": _none_consumption_maker,
    "InstantID Pose Prepare Pipe (JPS)": _none_consumption_maker,
    "InstantID Mask Prepare Pipe (JPS)": _none_consumption_maker,
    "Revision Settings Pipe (JPS)": _none_consumption_maker,
    "SDXL Fundamentals MultiPipe (JPS)": _none_consumption_maker,
    "Images Masks MultiPipe (JPS)": _none_consumption_maker,
    "SDXL Recommended Resolution Calc (JPS)": _none_consumption_maker,
    "Resolution Multiply (JPS)": _none_consumption_maker,
    "Largest Int (JPS)": _none_consumption_maker,
    "Multiply Int Int (JPS)": _none_consumption_maker,
    "Multiply Int Float (JPS)": _none_consumption_maker,
    "Multiply Float Float (JPS)": _none_consumption_maker,
    "Substract Int Int (JPS)": _none_consumption_maker,
    "Text Concatenate (JPS)": _none_consumption_maker,
    "Get Date Time String (JPS)": _none_consumption_maker,
    "Get Image Size (JPS)": _none_consumption_maker,
    "Crop Image Square (JPS)": _none_consumption_maker,
    "Crop Image TargetSize (JPS)": _none_consumption_maker,
    "Prepare Image (JPS)": _none_consumption_maker,
    "Prepare Image Plus (JPS)": _none_consumption_maker,
    "Prepare Image Tiled IPA (JPS)": _none_consumption_maker,
    "SDXL Prompt Styler (JPS)": _none_consumption_maker,
    "SDXL Prompt Handling (JPS)": _none_consumption_maker,
    "SDXL Prompt Handling Plus (JPS)": _none_consumption_maker,
    "Text Prompt (JPS)": _none_consumption_maker,
    "Text Prompt Combo (JPS)": _none_consumption_maker,
    "Save Images Plus (JPS)": _none_consumption_maker,
    "CLIPTextEncode SDXL Plus (JPS)": _none_consumption_maker,
    "Time Seed (JPS)": _none_consumption_maker,

    "Sine Curve [Dream]": _none_consumption_maker,
    "Linear Curve [Dream]": _none_consumption_maker,
    "CSV Curve [Dream]": _none_consumption_maker,
    "Beat Curve [Dream]": _none_consumption_maker,
    "Common Frame Dimensions [Dream]": _none_consumption_maker,
    "Image Motion [Dream]": _none_consumption_maker,
    "Noise from Palette [Dream]": _none_consumption_maker,
    "Analyze Palette [Dream]": _none_consumption_maker,
    "Palette Color Shift [Dream]": _none_consumption_maker,
    "File Count [Dream]": _none_consumption_maker,
    "Frame Counter Offset [Dream]": _none_consumption_maker,
    "Frame Counter (Directory) [Dream]": _none_consumption_maker,
    "Frame Counter (Simple) [Dream]": _none_consumption_maker,
    "Image Sequence Loader [Dream]": _none_consumption_maker,
    "Image Sequence Saver [Dream]": _none_consumption_maker,
    "CSV Generator [Dream]": _none_consumption_maker,
    "Sample Image Area as Palette [Dream]": _none_consumption_maker,
    "FFMPEG Video Encoder [Dream]": _none_consumption_maker,
    "Image Sequence Tweening [Dream]": _none_consumption_maker,
    "Image Sequence Blend [Dream]": _none_consumption_maker,
    "Palette Color Align [Dream]": _none_consumption_maker,
    "Sample Image as Palette [Dream]": _none_consumption_maker,
    "Noise from Area Palettes [Dream]": _none_consumption_maker,
    "String Input [Dream]": _none_consumption_maker,
    "Float Input [Dream]": _none_consumption_maker,
    "Int Input [Dream]": _none_consumption_maker,
    "Text Input [Dream]": _none_consumption_maker,
    "Big Latent Switch [Dream]": _none_consumption_maker,
    "Frame Count Calculator [Dream]": _none_consumption_maker,
    "Big Image Switch [Dream]": _none_consumption_maker,
    "Big Text Switch [Dream]": _none_consumption_maker,
    "Big Float Switch [Dream]": _none_consumption_maker,
    "Big Int Switch [Dream]": _none_consumption_maker,
    "Big Palette Switch [Dream]": _none_consumption_maker,
    "Build Prompt [Dream]": _none_consumption_maker,
    "Finalize Prompt [Dream]": _none_consumption_maker,
    "Frame Counter Info [Dream]": _none_consumption_maker,
    "Boolean To Float [Dream]": _none_consumption_maker,
    "Boolean To Int [Dream]": _none_consumption_maker,
    "Saw Curve [Dream]": _none_consumption_maker,
    "Triangle Curve [Dream]": _none_consumption_maker,
    "Triangle Event Curve [Dream]": _none_consumption_maker,
    "Smooth Event Curve [Dream]": _none_consumption_maker,
    "Calculation [Dream]": _none_consumption_maker,
    "Image Color Shift [Dream]": _none_consumption_maker,
    "Compare Palettes [Dream]": _none_consumption_maker,
    "Image Contrast Adjustment [Dream]": _none_consumption_maker,
    "Image Brightness Adjustment [Dream]": _none_consumption_maker,
    "Laboratory [Dream]": _none_consumption_maker,
    "String to Log Entry [Dream]": _none_consumption_maker,
    "Int to Log Entry [Dream]": _none_consumption_maker,
    "Float to Log Entry [Dream]": _none_consumption_maker,
    "Log Entry Joiner [Dream]": _none_consumption_maker,
    "String Tokenizer [Dream]": _none_consumption_maker,
    "WAV Curve [Dream]": _none_consumption_maker,
    "Frame Counter Time Offset [Dream]": _none_consumption_maker,
    "Lora Loader Stack (rgthree)": _none_consumption_maker,
    "PatchModelAddDownscale": _none_consumption_maker,
    "Power Prompt (rgthree)": _none_consumption_maker,
    "EmptyMochiLatentVideo": _none_consumption_maker,
    "Seed (rgthree)": _none_consumption_maker,
    "Image Comparer (rgthree)": _none_consumption_maker,
    "CR Text Replace": _none_consumption_maker,
    "VHS_MergeImages": _none_consumption_maker,
    "Ref_Image_Preprocessing": _none_consumption_maker,
    "ADE_AttachLoraHookToConditioning": _none_consumption_maker,
    "ADE_CombineLoraHooksFour": _none_consumption_maker,
    "ADE_RegisterModelAsLoraHook": _none_consumption_maker,
    "LoadImageMask": _none_consumption_maker,
    "JoinImageWithAlpha": _none_consumption_maker,
    "SplitImageWithAlpha": _none_consumption_maker,
    "DownloadAndLoadCogVideoModel": _none_consumption_maker,
    "easy loraStackApply": _none_consumption_maker,
    "DifferentialDiffusion": _none_consumption_maker,
    "InpaintModelConditioning": _none_consumption_maker,
    "ImpactGaussianBlurMask": _none_consumption_maker,
    "MaskPreview+": _none_consumption_maker,
    "ImageSender": _none_consumption_maker,
    "ReActorFaceBoost": _none_consumption_maker,
    "PrepImageForClipVision": _none_consumption_maker,
    "ModelMergeSDXL": _none_consumption_maker,
    "easy int": _none_consumption_maker,
    "Display Int (rgthree)": _none_consumption_maker,
    "Text Random Line": _none_consumption_maker,
    "ImpactFloat": _none_consumption_maker,
    "SeargeSDXLRefinerPromptEncoder": _none_consumption_maker,
    "Text Find and Replace": _none_consumption_maker,
    "Mask Invert": _none_consumption_maker,
    "MaskFlip+": _none_consumption_maker,
    "EmptyLatentImagePresets": _none_consumption_maker,
    "ModelSamplingDiscrete": _none_consumption_maker,
    "TripleCLIPLoaderGGUF": _none_consumption_maker,
    "SeargePromptText": _none_consumption_maker,
    "Prompts Everywhere": _none_consumption_maker,
    "Anything Everywhere": _none_consumption_maker,
    "IPAdapterStyleComposition": _none_consumption_maker,
    "RescaleCFG": _none_consumption_maker,
    "BAE-NormalMapPreprocessor": _none_consumption_maker,
    "InpaintPreprocessor": _none_consumption_maker,
    "ScribblePreprocessor": _none_consumption_maker,
    "CLIPTextEncodeSD3": _none_consumption_maker,
    "Image Dragan Photography Filter": _none_consumption_maker,
    "CreateFadeMaskAdvanced": _none_consumption_maker,
    "Power Lora Loader (rgthree)": _none_consumption_maker,
    "CLIPLoaderGGUF": _none_consumption_maker,
    "ConditioningUpscale //Inspire": _none_consumption_maker,
    "Paste By Mask": _none_consumption_maker,
    "Text Find and Replace Input": _none_consumption_maker,
    "Text String": _none_consumption_maker,
    "ReActorOptions": _none_consumption_maker,
    "ModelMergeAdd": _none_consumption_maker,
    "easy controlnetNames": _none_consumption_maker,
    "LayerColor: Color of Shadow & Highligh": _none_consumption_maker,
    "CreateHookLora": _none_consumption_maker,
    "CreateHookLoraModelOnly": _none_consumption_maker,
    "CreateHookModelAsLora": _none_consumption_maker,
    "CreateHookModelAsLoraModelOnly": _none_consumption_maker,
    "SetHookKeyframes": _none_consumption_maker,
    "CreateHookKeyframe": _none_consumption_maker,
    "CreateHookKeyframesInterpolated": _none_consumption_maker,
    "CreateHookKeyframesFromFloats": _none_consumption_maker,
    "CombineHooks2": _none_consumption_maker,
    "CombineHooks4": _none_consumption_maker,
    "CombineHooks8": _none_consumption_maker,
    "ConditioningSetProperties": _none_consumption_maker,
    "ConditioningSetPropertiesAndCombine": _none_consumption_maker,
    "PairConditioningCombine": _none_consumption_maker,
    "PairConditioningSetDefaultCombine": _none_consumption_maker,
    "ConditioningSetDefaultCombine": _none_consumption_maker,
    "SetClipHooks": _none_consumption_maker,
    "ConditioningTimestepsRange": _none_consumption_maker,
}


def get_monitor_params(obj, obj_type, input_data_all):
    func = _NODE_CONSUMPTION_MAPPING.get(obj_type, _default_consumption_maker)
    consumption = _map_node_consumption_over_list(obj, input_data_all, func)
    if consumption and 'opts' in consumption:
        for opt in consumption['opts']:
            opt["ratio"] = opt.get("ratio", 1) * 1.8

    return consumption