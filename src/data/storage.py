import io
from pathlib import Path

import pandas as pd

from base import AbstractStorage
from helper import split_dir_filename_ext
from src.data.local import LocalStorage


class StorageProcessor:
    def __init__(self, storage: AbstractStorage) -> None:
        self.storage = storage

    def download_all_files_binary(self, directory: str) -> list[bytes]:
        files = []
        for filename in self.storage.get_all_filenames(directory):
            files.append(self.storage.download_binary(filename))
        return files

    def filename_exists(self, name: str, directory: str) -> bool:
        if name in self.storage.get_all_filenames(directory):
            return True
        else:
            return False

    def get_max_filename(self, directory: str) -> int:
        filenames = self.storage.get_all_filenames(directory)
        if filenames:
            without_ext = [int(Path(fn).stem) for fn in filenames]
            return max(without_ext)
        else:
            return -1

    def exact_binary_exists(self, binary: bytes, directory: str) -> bool:
        binaries = self.download_all_files_binary(directory)
        for binary_ in binaries:
            if binary_ == binary:
                return True
        return False

    def images_to_storage(self, images: list[tuple[bytes, str]], directory: str):
        if isinstance(self.storage, LocalStorage):
            Path(directory).mkdir(parents=True, exist_ok=True)
        current_max_file_name = self.get_max_filename(directory)
        for image_binary, image_ext in images:
            current_max_file_name += 1
            image_path = str(Path(directory, str(current_max_file_name) + image_ext))
            self.storage.upload_binary(image_binary, image_path)

    def metadata_to_storage(self, metadata: list[dict[str, str]], path: str, index_columns: list[str]):
        df = pd.DataFrame(metadata)

        directory, filename, ext = split_dir_filename_ext(path)
        name = filename + ext

        if self.filename_exists(name, directory):
            csv_binary = self.storage.download_binary(path)
            old_df = pd.read_csv(csv_binary)
            df = pd.concat([old_df, df])
            df = df.drop_duplicates(subset=index_columns, keep="first").reset_index(drop=True)

        binary_io = io.BytesIO()
        df.to_csv(binary_io, index=False)
        self.storage.upload_binary(binary_io.getvalue(), path)
