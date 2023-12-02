from pathlib import Path
from typing import Union

from PIL import Image


def get_images_count(directory: Union[str, Path]) -> int:
    directory = Path(directory)
    files = [file for file in directory.rglob('*') if file.is_file()]
    return len(files)


def get_images_suffixes(directory: Union[str, Path]) -> list[str]:
    directory = Path(directory)
    files = [file.suffix for file in directory.rglob('*') if file.is_file()]
    return files


def get_images_formats(directory: Union[str, Path]) -> list[str]:
    directory = Path(directory)
    formats = []
    for file in directory.rglob('*'):
        if file.is_file():
            image = Image.open(file)
            formats.append(image.format)
    return formats


def get_images_sizes(directory: Union[str, Path]) -> list[tuple[int, int]]:
    directory = Path(directory)
    sizes = []
    for file in directory.rglob('*'):
        if file.is_file():
            image = Image.open(file)
            sizes.append(image.size)
    return sizes
