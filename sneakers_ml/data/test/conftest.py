from sneakers_ml.data.storage.base import AbstractStorage


class DummyStorage(AbstractStorage):
    def __init__(self):
        self.storage = {}

    def download_file(self, path: str) -> bytes:
        return self.storage.get(path)

    def delete_file(self, path: str) -> None:
        if self.storage.get(path) is not None:
            del self.storage[path]

    def get_all_files(self, dir: str) -> list[str]:
        raise NotImplementedError

    def upload_file(self, local_path: str, path: str) -> None:
        with open(local_path, "rb") as f:
            self.storage[path] = f.read()
