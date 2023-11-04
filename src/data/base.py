from abc import ABC, abstractmethod


# no side-imports in base (!)


class AbstractStorage(ABC):
    @abstractmethod
    def upload_file(self, local_path: str, path: str) -> None:
        raise NotImplemented

    @abstractmethod
    def download_file(self, path: str) -> bytes:
        raise NotImplemented

    @abstractmethod
    def delete_file(self, path: str) -> None:
        raise NotImplemented

    @abstractmethod
    def get_all_files(self, dir: str) -> list[str]:
        raise NotImplemented
