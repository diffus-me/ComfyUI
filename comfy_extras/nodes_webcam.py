import nodes
import folder_paths

import execution_context

MAX_RESOLUTION = nodes.MAX_RESOLUTION


class WebcamCapture(nodes.LoadImage):
    @classmethod
    def INPUT_TYPES(s, context: execution_context.ExecutionContext):
        return {
            "required": {
                "image": ("WEBCAM", {}),
                "width": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "height": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "capture_on_queue": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "context": "EXECUTION_CONTEXT"
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_capture"

    CATEGORY = "image"

    def load_capture(self, image, **kwargs):
        context = kwargs["context"]
        return super().load_image(folder_paths.get_annotated_filepath(image, context.user_hash), context)

    @classmethod
    def IS_CHANGED(cls, image, width, height, capture_on_queue, context: execution_context.ExecutionContext):
        return super().IS_CHANGED(image, context)


NODE_CLASS_MAPPINGS = {
    "WebcamCapture": WebcamCapture,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WebcamCapture": "Webcam Capture",
}
