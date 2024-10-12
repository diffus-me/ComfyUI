from aiohttp import web
from typing import Optional

import execution_context
from folder_paths import get_models_dir, get_user_directory, get_output_directory, folder_names_and_paths
from api_server.services.file_service import FileService
import app.logger

class InternalRoutes:
    '''
    The top level web router for internal routes: /internal/*
    The endpoints here should NOT be depended upon. It is for ComfyUI frontend use only.
    Check README.md for more information.
    
    '''
    def __init__(self):
        self.routes: web.RouteTableDef = web.RouteTableDef()
        self._app: Optional[web.Application] = None
        self.file_service = FileService({
            # "models": get_models_dir,
            "user": get_user_directory,
            "output": get_output_directory
        })

    def setup_routes(self):
        @self.routes.get('/files')
        async def list_files(request):
            directory_key = request.query.get('directory', '')
            context = execution_context.ExecutionContext(request)
            try:
                file_list = self.file_service.list_files(context, directory_key)
                return web.json_response({"files": file_list})
            except ValueError as e:
                return web.json_response({"error": str(e)}, status=400)
            except Exception as e:
                return web.json_response({"error": str(e)}, status=500)
        @self.routes.get('/logs')
        async def get_logs(request):
            # return web.json_response(app.logger.get_logs())
            return web.json_response([])

        @self.routes.get('/folder_paths')
        async def get_folder_paths(request):
            response = {}
            for key in folder_names_and_paths:
                response[key] = folder_names_and_paths[key][0]
            return web.json_response(response)
    def get_app(self):
        if self._app is None:
            self._app = web.Application()
            self.setup_routes()
            self._app.add_routes(self.routes)
        return self._app
