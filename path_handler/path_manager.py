from pathlib import Path
import inspect


class PathManager:
    def __init__(self, base_directory=None, marker_file='.project_root'):
        # all paths is absolute
        str_file_path = self._get_path_of_first_importing_script()
        
        if str_file_path:
            self._file_path = self._convert_file_path(file_path=str_file_path)
            self._file_directory_path = self._convert_file_directory_path(file_path=str_file_path)
        else:
            raise FileNotFoundError('Can not find the path of file that imported this module')

        self._base_directory = Path(base_directory) if base_directory else self.find_project_root(marker_file)

    def _get_resolve_path(self, relative_custome_path: str, based: str) -> Path:
        if based == 'file':
            return (self.get_file_directory_path() / relative_custome_path).resolve()
        elif based == 'base':
            return (self._base_directory / relative_custome_path).resolve()

    def validate_path(self, path):
        path = Path(path)
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Path {path} does not exist or is not a directory")

    def find_project_root(self, marker_file):
        current_dir = self.get_file_directory_path()
        
        while current_dir != current_dir.parent:
            if (current_dir / marker_file).exists():
                return current_dir
            current_dir = current_dir.parent
        
        raise FileNotFoundError(f"project root marker file [name: {marker_file}] not found.")

    def get_base_directory(self):
        if self._base_directory:
            return self._base_directory
        else:
            raise NameError(f'the variable `self._base_directory` is not set! current value: {self._base_directory}')
    
    def get_file_path(self):
        if self._file_path:
            return self._file_path
        else:
            raise NameError(f'the variable `self._file_path` is not set! current value: {self._file_path}')
    
    def get_file_directory_path(self):
        if self._file_directory_path:
            return self._file_directory_path
        else:
            raise NameError(f'the variable `self._file_directory_path` is not set! current value: {self._file_directory_path}')
    
    def _convert_file_path(self, file_path: str) -> Path:
        return Path(file_path)

    def _convert_file_directory_path(self, file_path: str) -> Path:
        return Path(file_path).parent

    def get_custom_path(self, relative_custome_path: str, based: str = 'file'):
        return self._get_resolve_path(relative_custome_path, based)

    def _get_path_of_first_importing_script(self):
        if __name__ == '__main__':
            return __file__
        
        last_call_of_this_script = -1
        app_stack = inspect.stack()
        
        for index, stack_frame in enumerate(app_stack):
            if stack_frame.filename == __file__:
                last_call_of_this_script = index        
        
        return app_stack[last_call_of_this_script + 1].filename
