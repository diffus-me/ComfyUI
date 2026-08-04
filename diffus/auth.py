import os

from aiohttp import web


def create_authenticate_middleware():
    api_secret = os.environ.get("API_SECRET")
    if not api_secret:
        return None

    @web.middleware
    async def authenticate_middleware(request: web.Request, handler):
        request_path = request.path
        if (
                request_path.startswith("/api/")
                or request_path.startswith("/internal/")
                or request_path.startswith("/daemon/")
        ) and request.headers.get("X-Api-Secret") != api_secret:
            return web.Response(status=401)
        else:
            return await handler(request)

    return authenticate_middleware
