from base import AbstractStorage
from pathlib import Path
import pandas as pd
import io
from helper import split_dir_filename_ext


class StorageProcessor:
    def __init__(self, storage: AbstractStorage) -> None:
        self.storage = storage

    def download_all_files_binary(self, dir: str) -> list[bytes]:
        files = []
        for filename in self.get_all_filenames(dir):
            files.append(self.storage.download_binary(filename))
        return files

    def filename_exists(self, name: str, dir: str) -> bool:
        if name in self.storage.get_all_filenames(dir):
            return True
        else:
            return False

    def get_max_filename(self, dir: str) -> int:
        filenames = self.storage.get_all_filenames(dir)
        if filenames:
            without_ext = [int(Path(fn).stem) for fn in filenames]
            return max(without_ext)
        else:
            return -1

    def exact_binary_exists(self, binary: bytes, dir: str) -> bool:
        binaries = self.download_all_files_binary(dir)
        for binary_ in binaries:
            if binary_ == binary:
                return True
        return False

    def images_to_storage(self, images: tuple[bytes, str], dir: str):
        current_max_file_name = self.get_max_filename(dir)
        for image_binary, image_ext in images:
            current_max_file_name += 1
            image_path = str(Path(dir, str(current_max_file_name) + image_ext))
            self.storage.upload_binary(image_binary, image_path)

    def metadata_to_storage(
        self, metadata: dict[str, str], path: str, index_columns: str
    ):
        df = pd.DataFrame(metadata)

        dir, filename, ext = split_dir_filename_ext(path)
        name = filename + ext

        if self.file_name_exists(name, dir):
            csv_binary = self.storage.download_binary(path)
            old_df = pd.read_csv(csv_binary)
            df = pd.concat([old_df, df])
            df = df.drop_duplicates(subset=index_columns, keep="first").reset_index(
                drop=True
            )

        binary_io = io.BytesIO()
        df.to_csv(binary_io, index=False)
        self.storage.upload_binary(binary_io.getvalue(), path)
