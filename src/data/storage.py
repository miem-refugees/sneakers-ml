import io
from pathlib import Path
from typing import Union

import pandas as pd

from src.data.base import AbstractStorage
from src.data.local import LocalStorage
from src.data.s3 import S3Storage


class StorageProcessor:

    def __init__(self, storage: AbstractStorage) -> None:
        self.storage = storage

    def download_all_files_binary(self, directory: str) -> list[bytes]:
        files = []
        for filename in self.storage.get_all_filenames(directory):
            path = str(Path(directory, filename))
            files.append(self.storage.download_binary(path))
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

    def images_to_destination(self, source_list: list[str], destination: Union[str, Path]) -> None:
        if isinstance(self.storage, S3Storage):
            raise NotImplementedError

        if len(source_list) == 0:
            return

        images = []

        for source_path in source_list:
            source_path = Path(source_path)

            if source_path.is_dir():
                for source_file in source_path.glob('*'):
                    image_binary = self.storage.download_binary(str(source_file))
                    image_extension = source_file.suffix
                    images.append((image_binary, image_extension))

            elif source_path.is_file():
                image_binary = self.storage.download_binary(str(source_path))
                image_extension = source_path.suffix
                images.append((image_binary, image_extension))

        self.images_to_storage(images, destination)

    def images_to_storage(self, images: list[tuple[bytes, str]], directory: Union[Path, str]):
        if isinstance(self.storage, LocalStorage):
            Path(directory).mkdir(parents=True, exist_ok=True)
        current_max_file_name = self.get_max_filename(directory)
        # checking for existing images slows down process by a lot
        existing_images = set(self.download_all_files_binary(directory))
        for image_binary, image_ext in images:
            if image_binary not in existing_images:
                current_max_file_name += 1
                image_path = str(Path(directory, str(current_max_file_name) + image_ext))
                self.storage.upload_binary(image_binary, image_path)
                existing_images.add(image_binary)

    def metadata_to_storage(self, metadata: list[dict[str, str]], path: str, index_columns: list[str]) -> None:
        df = pd.DataFrame(metadata)

        directory, filename, ext = self.split_dir_filename_ext(path)
        name = filename + ext

        if isinstance(self.storage, LocalStorage):
            Path(directory).mkdir(parents=True, exist_ok=True)

        if self.filename_exists(name, directory):
            csv_binary = self.storage.download_binary(path)
            old_df = pd.read_csv(io.BytesIO(csv_binary))
            df = pd.concat([old_df, df])
            df = df.drop_duplicates(subset=index_columns, keep="first").reset_index(drop=True)

        binary_io = io.BytesIO()
        df.to_csv(binary_io, index=False)
        self.storage.upload_binary(binary_io.getvalue(), path)

    @staticmethod
    def split_dir_filename_ext(path):
        path_obj = Path(path)
        directory = path_obj.parent
        filename = path_obj.stem
        file_extension = path_obj.suffix
        return str(directory), str(filename), str(file_extension)
