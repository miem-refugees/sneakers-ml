from pathlib import Path
from typing import Optional

from PIL import Image


def get_all_images(directory: str) -> list[str]:
    directory_path = Path(directory)
    files = [str(file) for file in directory_path.rglob("*") if file.is_file()]
    return files


def get_images_count(directory: str) -> int:
    return len(get_all_images(directory))


def get_images_suffixes(directory: str) -> list[str]:
    directory_path = Path(directory)
    files = [file.suffix for file in directory_path.rglob("*") if file.is_file()]
    return files


def get_images_formats(directory: str) -> list[Optional[str]]:
    directory_path = Path(directory)
    formats = []
    for file in directory_path.rglob("*"):
        if file.is_file():
            image = Image.open(file)
            formats.append(image.format)
    return formats


def get_images_modes(directory: str) -> list[str]:
    directory_path = Path(directory)
    modes = []
    for file in directory_path.rglob("*"):
        if file.is_file():
            image = Image.open(file)
            modes.append(image.mode)
    return modes


def get_images_sizes(directory: str) -> list[tuple[int, int]]:
    directory_path = Path(directory)
    sizes = []
    for file in directory_path.rglob("*"):
        if file.is_file():
            image = Image.open(file)
            sizes.append(image.size)
    return sizes
