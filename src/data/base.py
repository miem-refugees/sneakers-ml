from abc import ABC, abstractmethod


# no side-imports in base (!)


class AbstractStorage(ABC):
    @abstractmethod
    def upload_file(self, local_path: str, path: str) -> None:
        raise NotImplemented

    @abstractmethod
    def upload_binary(self, binary_data: bytes, path: str) -> None:
        raise NotImplemented

    @abstractmethod
    def download_file(self, path: str, local_path: str) -> None:
        raise NotImplemented

    @abstractmethod
    def download_binary(self, path: str) -> bytes:
        raise NotImplemented

    @abstractmethod
    def delete_file(self, path: str) -> None:
        raise NotImplemented

    @abstractmethod
    def get_all_files(self, dir: str) -> list[str]:
        raise NotImplemented

    @abstractmethod
    def download_all_files_binary(self, dir: str) -> list[bytes]:
        raise NotImplemented

    @abstractmethod
    def file_exists(self, local_path: str, s3_path: str) -> bool:
        raise NotImplemented

    @abstractmethod
    def exact_file_exists(self, dir: str, binary_data: bytes) -> bool:
        raise NotImplemented
