from abc import ABC, abstractmethod

# no side-imports in base (!)


class AbstractStorage(ABC):
    @abstractmethod
    def upload_file(self, local_path: str, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def upload_binary(self, binary_data: bytes, local_path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def download_file(self, path: str, local_path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def download_binary(self, local_path: str) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def delete_file(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_all_filenames(self, directory: str) -> list[str]:
        raise NotImplementedError
