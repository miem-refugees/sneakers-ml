from pathlib import Path

from base import AbstractStorage


class LocalStorage(AbstractStorage):
    def upload_file(self, local_path: str, path: str) -> None:
        raise NotImplemented

    def upload_binary(self, binary_data: bytes, local_path: str) -> None:
        with open(local_path, "wb") as file:
            file.write(binary_data)

    def download_file(self, path: str, local_path: str) -> None:
        raise NotImplemented

    def download_binary(self, local_path: str) -> bytes:
        with open(local_path, "rb") as file:
            return file.read()

    def delete_file(self, path: str) -> None:
        Path(path).unlink()

    def get_all_filenames(self, directory: str) -> list[str]:
        return [str(file) for file in Path(directory).iterdir() if file.is_file()]
