import os
if os.getenv('SQL_DATABASE_URL_COMFY'):
    FAVORITE_MODEL_TYPES = {
        'checkpoints': 'CHECKPOINT',
        'diffusion_models': 'CHECKPOINT',
        'unet_gguf': 'CHECKPOINT',
        'model_gguf': 'CHECKPOINT',
        'loras': 'LORA',
        'lycoris': 'LYCORIS',
        'embeddings': 'EMBEDDING',
    }
else:
    FAVORITE_MODEL_TYPES = {}
