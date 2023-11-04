import base64
from abc import ABC, abstractmethod
from io import BytesIO

from PIL import Image


class AbstractStorage(ABC):
    @abstractmethod
    def upload_file(self, local_path: str, s3_path: str) -> None:
        raise NotImplemented

    @abstractmethod
    def download_file(self, s3_path: str) -> bytes:
        raise NotImplemented

    @abstractmethod
    def delete_file(self, s3_path: str) -> None:
        raise NotImplemented

    @abstractmethod
    def get_all_files(self, dir: str) -> list[str]:
        raise NotImplemented


class AbstractSneaker(ABC):
    def __init__(self, storage: AbstractStorage):
        self.storage = storage

    def preload_image(self, path: str) -> str:
        binary = self.storage.download_file(path)
        if len(binary) == 0: raise RuntimeError("")

        im = Image.frombytes(binary)

        with (BytesIO() as buffer):
            im.save(buffer, 'jpeg')
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            return f'<img src="data:image/jpeg;base64,{image_base64}">'
