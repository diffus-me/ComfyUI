import nodes
import folder_paths
import os

from typing_extensions import override
from comfy_api.latest import IO, ComfyExtension, InputImpl, UI

from pathlib import Path

import execution_context


def normalize_path(path):
    return path.replace('\\', '/')

class Load3D(IO.ComfyNode):
    @classmethod
    def define_schema(cls, exec_context: execution_context.ExecutionContext):
        if exec_context:
            input_dir = os.path.join(folder_paths.get_input_directory(user_hash=exec_context.user_hash), "3d")

            os.makedirs(input_dir, exist_ok=True)

            input_path = Path(input_dir)
            base_path = Path(folder_paths.get_input_directory(user_hash=exec_context.user_hash))

            files = [
                normalize_path(str(file_path.relative_to(base_path)))
                for file_path in input_path.rglob("*")
                if file_path.suffix.lower() in {'.gltf', '.glb', '.obj', '.fbx', '.stl'}
            ]
        else:
            files = []
        return IO.Schema(
            node_id="Load3D",
            display_name="Load 3D & Animation",
            category="3d",
            is_experimental=True,
            inputs=[
                IO.Combo.Input("model_file", options=sorted(files), upload=IO.UploadType.model),
                IO.Load3D.Input("image"),
                IO.Int.Input("width", default=1024, min=1, max=4096, step=1),
                IO.Int.Input("height", default=1024, min=1, max=4096, step=1),
            ],
            hidden=[IO.Hidden.exec_context],
            outputs=[
                IO.Image.Output(display_name="image"),
                IO.Mask.Output(display_name="mask"),
                IO.String.Output(display_name="mesh_path"),
                IO.Image.Output(display_name="normal"),
                IO.Load3DCamera.Output(display_name="camera_info"),
                IO.Video.Output(display_name="recording_video"),
            ],
        )

    @classmethod
    def execute(cls, model_file, image, **kwargs) -> IO.NodeOutput:
        exec_context = kwargs.get("exec_context")
        image_path = folder_paths.get_annotated_filepath(image['image'], user_hash=exec_context.user_hash)
        mask_path = folder_paths.get_annotated_filepath(image['mask'], user_hash=exec_context.user_hash)
        normal_path = folder_paths.get_annotated_filepath(image['normal'], user_hash=exec_context.user_hash)

        load_image_node = nodes.LoadImage()
        output_image, ignore_mask = load_image_node.load_image(image=image_path, exec_context=exec_context)
        ignore_image, output_mask = load_image_node.load_image(image=mask_path, exec_context=exec_context)
        normal_image, ignore_mask2 = load_image_node.load_image(image=normal_path, exec_context=exec_context)

        video = None

        if image['recording'] != "":
            recording_video_path = folder_paths.get_annotated_filepath(image['recording'], user_hash=exec_context.user_hash)

            video = InputImpl.VideoFromFile(recording_video_path)

        return IO.NodeOutput(output_image, output_mask, model_file, normal_image, image['camera_info'], video)

    process = execute  # TODO: remove


class Preview3D(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Preview3D",
            display_name="Preview 3D & Animation",
            category="3d",
            is_experimental=True,
            is_output_node=True,
            inputs=[
                IO.String.Input("model_file", default="", multiline=False),
                IO.Load3DCamera.Input("camera_info", optional=True),
                IO.Image.Input("bg_image", optional=True),
            ],
            hidden=[
                IO.Hidden.exec_context,
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, model_file, **kwargs) -> IO.NodeOutput:
        exec_context = kwargs.get("exec_context")
        camera_info = kwargs.get("camera_info", None)
        bg_image = kwargs.get("bg_image", None)
        return IO.NodeOutput(ui=UI.PreviewUI3D(model_file, camera_info, bg_image=bg_image, exec_context=exec_context))

    process = execute  # TODO: remove


class Load3DExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            Load3D,
            Preview3D,
        ]


async def comfy_entrypoint() -> Load3DExtension:
    return Load3DExtension()
