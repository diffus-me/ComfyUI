from typing import Dict, List, Optional, Callable

import execution_context
from api_server.utils.file_operations import FileSystemOperations, FileSystemItem

class FileService:
    def __init__(self, allowed_directories: Dict[str, Callable], file_system_ops: Optional[FileSystemOperations] = None):
        self.allowed_directories: Dict[str, Callable] = allowed_directories
        self.file_system_ops: FileSystemOperations = file_system_ops or FileSystemOperations()

    def list_files(self, context: execution_context.ExecutionContext, directory_key: str) -> List[FileSystemItem]:
        if directory_key not in self.allowed_directories:
            raise ValueError("Invalid directory key")
        directory_path: str = self.allowed_directories[directory_key](context.user_hash)
        return self.file_system_ops.walk_directory(directory_path)