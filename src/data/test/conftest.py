import os

from src.data.base import AbstractStorage


class DummyStorage(AbstractStorage):
    def __init__(self):
        self.storage = {}

    def download_file(self, path: str) -> bytes:
        return self.storage.get(path)

    def delete_file(self, s3_path: str) -> None:
        if self.storage.get(s3_path) is not None:
            del self.storage[s3_path]

    def get_all_files(self, dir: str) -> list[str]:
        raise NotImplemented

    def upload_file(self, local_path: str, path: str) -> None:
        with open(local_path, 'rb') as f:
            self.storage[path] = f.read()
