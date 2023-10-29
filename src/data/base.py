from abc import ABC, abstractmethod


class AbstractData(ABC):
    @abstractmethod
    def __next__(self):
        pass


class AbstractStorage(ABC):
    @abstractmethod
    def upload_file(self, local_path: str, s3_path: str) -> None:
        pass

    @abstractmethod
    def download_file(self, s3_path: str) -> object:
        pass
