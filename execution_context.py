import datetime
import uuid

import diffus.models
import diffus.repository


class Geninfo:
    def __init__(self, task_id):
        self.positive_prompt = ''
        self.negative_prompt = ''
        self.steps = 0
        self.sampler = ''
        self.cfg_scale = 0
        self.seed = 0
        self.task_id = task_id

    def dump(self):
        return {
            "Prompt": self.positive_prompt,
            "Negative prompt": self.negative_prompt,
            "Steps": self.steps,
            "Sampler": self.sampler,
            "CFG scale": self.cfg_scale,
            "Seed": self.seed,
            "Diffus task ID": self.task_id,
        }


class ExecutionContext:
    def __init__(self, request, extra_data={}):
        self._headers = dict(request.headers)
        self._extra_data = extra_data
        self._used_models: dict[str, dict] = {}
        self._checkpoints_model_base = ""
        self._task_id = self._headers.get('x-task-id', str(uuid.uuid4()))

        self._geninfo = Geninfo(self._task_id)

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
    def task_id(self):
        return self._task_id

    @property
    def geninfo(self):
        return self._geninfo.dump()

    @property
    def positive_prompt(self):
        return self._geninfo.positive_prompt

    @property
    def negative_prompt(self):
        return self._geninfo.negative_prompt

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

    @staticmethod
    def _get_origin_text_from_tokens(tokens):
        return [
            t[1]["_origin_text_"] for t in tokens if
            len(t) > 1 and isinstance(t[1], dict) and "_origin_text_" in t[1] and t[1]["_origin_text_"]
        ]

    def set_geninfo(
            self,
            positive_prompt=None,
            negative_prompt=None,
            steps=0,
            sampler='',
            cfg_scale=0,
            seed=0,
    ):
        if positive_prompt is None:
            positive_prompt = {}
        if negative_prompt is None:
            negative_prompt = {}

        self._geninfo.positive_prompt = self._concat_prompt(
            self._geninfo.positive_prompt,
            self._get_origin_text_from_tokens(positive_prompt)
        )
        self._geninfo.negative_prompt = self._concat_prompt(
            self._geninfo.negative_prompt,
            self._get_origin_text_from_tokens(negative_prompt)
        )
        self._geninfo.steps = steps
        self._geninfo.sampler = sampler
        self._geninfo.cfg_scale = cfg_scale
        self._geninfo.seed = seed

    @staticmethod
    def _concat_prompt(prompt_1: str, prompt_2: list[str]):
        if prompt_2:
            return prompt_1 + " " + " ".join(prompt_2)
        else:
            return prompt_1
