from pathlib import Path
from base import AbstractStorage


class LocalStorage():
    def upload_binary(self, binary_data: bytes, local_path: str) -> None:
        with open(local_path, "wb") as file:
            file.write(binary_data)

    def download_binary(self, local_path: str) -> bytes:
        with open(local_path, "rb") as file:
            return file.read()

    def get_all_filenames(self, dir: str) -> list[str]:
        return [str(file) for file in Path(dir).iterdir() if file.is_file()]
