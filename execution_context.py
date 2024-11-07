class ExecutionContext:
    def __init__(self, request, extra_data={}):
        self._headers = dict(request.headers)
        self._extra_data = extra_data
        self._used_models = []

    @property
    def used_models(self):
        return self._used_models

    def add_used_models(self, model_info):
        self._used_models.append(model_info)

    @property
    def has_flux_model(self):
        for model in self._used_models:
            if model.model_type == 'checkpoint' and model.base and model.base.lower() in ('sd3', 'flux'):
                return True
        return False

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
