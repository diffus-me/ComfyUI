class ExecutionContext:
    def __init__(self, request, extra_data={}):
        self._headers = dict(request.headers)
        self._extra_data = extra_data

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
