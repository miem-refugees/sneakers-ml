import io
from pathlib import Path
from typing import Union

import pandas as pd
from PIL import Image

from sneakers_ml.data.storage.base import AbstractStorage
from sneakers_ml.data.storage.local import LocalStorage
from sneakers_ml.data.storage.s3 import S3Storage


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
        return name in self.storage.get_all_filenames(directory)

    def get_max_filename(self, directory: str) -> int:
        if filenames := self.storage.get_all_filenames(directory):
            without_ext = [int(Path(fn).stem) for fn in filenames]
            return max(without_ext)
        return -1

    def exact_binary_exists(self, binary: bytes, directory: str) -> bool:
        binaries = set(self.download_all_files_binary(directory))
        return binary in binaries

    def images_to_directory(self, source_list: list[str], directory: str) -> int:
        if isinstance(self.storage, S3Storage):
            raise NotImplementedError

        if len(source_list) == 0:
            return 0

        images = []

        for source_path in source_list:
            path = Path(source_path)

            if path.is_dir():
                for source_file in path.glob("*"):
                    image_binary_raw = self.storage.download_binary(str(source_file))
                    image_binary, image_extension = self.fix_image(image_binary_raw)
                    if image_binary and image_extension:
                        images.append((image_binary, image_extension))
            elif path.is_file():
                image_binary_raw = self.storage.download_binary(str(path))
                image_binary, image_extension = self.fix_image(image_binary_raw)
                if image_binary and image_extension:
                    images.append((image_binary, image_extension))

        return self.images_to_storage(images, directory)

    def images_to_storage(self, images: list[tuple[bytes, str]], directory: str) -> int:
        if isinstance(self.storage, LocalStorage):
            Path(directory).mkdir(parents=True, exist_ok=True)

        current_max_file_name = self.get_max_filename(directory)

        existing_images = set(self.download_all_files_binary(directory))
        for image_binary, image_ext in images:
            if image_binary not in existing_images:
                current_max_file_name += 1
                image_path = str(Path(directory, str(current_max_file_name) + image_ext))
                self.storage.upload_binary(image_binary, image_path)
                existing_images.add(image_binary)
        return len(existing_images)

    def metadata_to_storage(self, metadata: list[dict[str, str]], path: str, index_columns: list[str]) -> None:
        metadata_df = pd.DataFrame(metadata)

        directory, name = self.split_dir_name(path)

        if isinstance(self.storage, LocalStorage):
            Path(directory).mkdir(parents=True, exist_ok=True)

        if self.filename_exists(name, directory):
            csv_binary = self.storage.download_binary(path)
            old_df = pd.read_csv(io.BytesIO(csv_binary))
            metadata_df = pd.concat([old_df, metadata_df])
            metadata_df = metadata_df.drop_duplicates(subset=index_columns, keep="first").reset_index(drop=True)

        binary_io = io.BytesIO()
        metadata_df.to_csv(binary_io, index=False)
        self.storage.upload_binary(binary_io.getvalue(), path)

    @staticmethod
    def fix_image(image_binary: bytes) -> Union[tuple[bytes, str], tuple[None, None]]:
        image = Image.open(io.BytesIO(image_binary))

        if image.mode == "P":  # remove gif images
            return None, None

        if image.mode != "RGB":
            image = image.convert("RGB")

        output_buffer = io.BytesIO()
        image.save(output_buffer, format="JPEG")
        image_binary = output_buffer.getvalue()
        return image_binary, ".jpeg"

    @staticmethod
    def split_dir_filename_ext(path: str) -> tuple[str, str, str]:
        path_obj = Path(path)
        directory = path_obj.parent
        filename = path_obj.stem
        file_extension = path_obj.suffix
        return str(directory), filename, file_extension

    @staticmethod
    def split_dir_name(path: str) -> tuple[str, str]:
        path_obj = Path(path)
        directory = path_obj.parent
        name = path_obj.name
        return str(directory), name
