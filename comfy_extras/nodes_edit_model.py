import node_helpers


class ReferenceLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                            },
                "optional": {
                    "latent": ("LATENT", ),
                    "enable": ("BOOLEAN", {"default": True}),}
               }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"

    CATEGORY = "advanced/conditioning/edit_models"
    DESCRIPTION = "This node sets the guiding latent for an edit model. If the model supports it you can chain multiple to set multiple reference images."

    def append(self, conditioning, latent=None, enable: bool=True):
        if enable and latent is not None:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [latent["samples"]]}, append=True)
        return (conditioning, )


NODE_CLASS_MAPPINGS = {
    "ReferenceLatent": ReferenceLatent,
}
