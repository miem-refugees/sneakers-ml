from helper import get_filenames
from pathlib import Path
from src.data.base import AbstractStorage


class LocalStorage(AbstractStorage):
    def get_filenames(dir):
        return [str(file) for file in Path(dir).iterdir() if file.is_file()]

    def same_file_exists(path: str, image_binary: bytes) -> bool:
        """
        Checks if the exact same file exists in path folder.
        """
        images = get_filenames(path)

        for image in images:
            if open(image, "rb").read() == image_binary:
                return True

        return False

    def get_max_file_name(path: str) -> int:
        """
        Returns the max integer file number in the "path" folder.
        Example files: 1.png, 2.png, 3.png. Returned value: 3.
        Returns -1 if the folder is empty.
        """
        filenames = get_filenames(path)

        if filenames:
            without_ext = [int(Path(fn).stem) for fn in filenames]
            return max(without_ext)
        else:
            return -1

    def upload_file():
        with open(image_path, "wb") as f:
            f.write(image_binary)

    def file_name_exists(path):
        return Path(path).is_file()
