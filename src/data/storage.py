from base import AbstractStorage
from pathlib import Path
import pandas as pd

def images_to_storage(storage: AbstractStorage, images: tuple[bytes, str], path: str):
    current_max_file_name = storage.get_max_file_name(path)

    for image_binary, image_ext in images:
        current_max_file_name += 1
        image_path = str(Path(path, str(current_max_file_name) + image_ext))
        storage.upload_binary(image_binary, image_path)


def metadata_to_storage(
    storage: AbstractStorage, metadata: dict[str, str], path: str, index_columns: str
):
    df = pd.DataFrame(metadata)

    if storage.file_name_exists(path):
        csv_binary = storage.download_file(path)
        old_df = pd.read_csv(csv_binary)
        df = pd.concat([old_df, df])
        df = df.drop_duplicates(subset=index_columns, keep="first").reset_index(
            drop=True
        )

    df.to_csv(path, index=False)
