import os
if os.getenv('SQL_DATABASE_URL_COMFY'):
    FAVORITE_MODEL_TYPES = {
        'checkpoints': 'CHECKPOINT',
        'loras': 'LORA',
        'lycoris': 'LYCORIS',
    }
else:
    FAVORITE_MODEL_TYPES = {}
