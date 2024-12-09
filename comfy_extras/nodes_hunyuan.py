class CLIPTextEncodeHunyuanDiT:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP", ),
            "bert": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "mt5xl": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    def encode(self, clip, bert, mt5xl):
        tokens = clip.tokenize(bert)
        tokens["mt5xl"] = clip.tokenize(mt5xl)["mt5xl"]

        return (clip.encode_from_tokens_scheduled(tokens, add_dict={"_origin_text_": bert + " " + mt5xl}), )


NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeHunyuanDiT": CLIPTextEncodeHunyuanDiT,
}
