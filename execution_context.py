import diffus.models
import diffus.repository


class ExecutionContext:
    def __init__(self, request, extra_data={}):
        self._headers = dict(request.headers)
        self._extra_data = extra_data
        self._used_models: dict[str, dict] = {}
        self._positive_prompt = ""
        self._negative_prompt = ""
        self._checkpoints_model_base = ""

    def validate_model(self, model_type, model_name, model_info=None):
        if model_type not in diffus.models.FAVORITE_MODEL_TYPES:
            return
        if model_type not in self._used_models:
            self._used_models[model_type] = {}
        if model_name not in self._used_models[model_type]:
            if not model_info:
                model_info = diffus.repository.get_favorite_model_full_path(self.user_id, model_type, model_name)
            self._used_models[model_type][model_name] = model_info

    def get_model(self, model_type, model_name):
        return self._used_models.get(model_type, {}).get(model_name, None)

    @property
    def loaded_model_ids(self):
        result = []
        for model_type in diffus.models.FAVORITE_MODEL_TYPES:
            result += [model_info.id for model_info in self._used_models.get(model_type, {}).values()]
        return result

    @property
    def checkpoints_model_base(self):
        for model_info in self.loaded_checkpoints:
            if model_info.base:
                return model_info.base
        # return self._checkpoints_model_base
        return None

    @checkpoints_model_base.setter
    def checkpoints_model_base(self, model_base):
        self._checkpoints_model_base = model_base

    @property
    def loaded_checkpoints(self):
        return [model_info for model_info in self._used_models.get('checkpoints', {}).values() if model_info]

    @property
    def loaded_loras(self):
        return [
            model_info for model_info in self._used_models.get('loras', {}).values() if model_info
        ] + [
            model_info for model_info in self._used_models.get('lycoris', {}).values() if model_info
        ]

    @property
    def positive_prompt(self):
        return self._positive_prompt

    @positive_prompt.setter
    def positive_prompt(self, text: str):
        self._positive_prompt = text

    @property
    def negative_prompt(self):
        return self._negative_prompt

    @negative_prompt.setter
    def negative_prompt(self, text):
        self._negative_prompt = text

    @property
    def user_hash(self):
        if self._headers:
            return self._headers.get('X-Diffus-User-Hash', None) or self._headers.get('x-diffus-user-hash', '')
        else:
            return ''

    @property
    def user_id(self):
        if self._headers:
            return self._headers.get('User-Id', None) or self._headers.get('user-id', '')
        else:
            return ''

    @property
    def extra_data(self):
        return self._extra_data or {}
