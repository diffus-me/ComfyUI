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

        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        output["_origin_text_"] = bert + " " + mt5xl
        return ([[cond, output]], )


NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeHunyuanDiT": CLIPTextEncodeHunyuanDiT,
}
