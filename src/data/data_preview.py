import base64
import logging
import io

from typing import Optional

from pandas import DataFrame
from PIL import Image

from src.data.base import AbstractStorage


class ImagePreview:
    def __init__(self, storage: AbstractStorage):
        self.storage = storage
        self.logger = logging.getLogger(self.__class__.__name__)

    def preview(self, data_frame: DataFrame, path_column: str, preview_column: str = "preview") -> Optional[str]:
        """
        Loads image from storage to new column in frame
        :param data_frame: Pandas dataframe
        :param path_column: Column name with image path
        :param preview_column: Column name to watch sneaker
        :return: string
        """
        data_frame[preview_column] = data_frame[path_column].map(self.__load_image)
        return data_frame

    def __load_image(self, path: str):
        binary = self.storage.download_file(path)
        if len(binary) == 0:
            self.logger.error("Could not download file from path: {}", path)
            return None

        im = Image.open(io.BytesIO(binary))

        with (io.BytesIO() as buffer):
            im.save(buffer, 'jpeg')
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            return f'<img src="data:image/jpeg;base64,{image_base64}">'
