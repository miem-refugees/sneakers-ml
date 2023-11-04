import base64
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Union
from pathlib import Path
from PIL import Image
from bs4 import BeautifulSoup
import requests
from helper import (
    add_https,
    add_page,
    get_image_extension,
    remove_query,
    remove_params,
    fix_string,
    fix_html_text,
    save_images,
    save_metadata,
    HEADERS,
)
from fake_useragent import UserAgent
from tqdm.auto import tqdm, trange


class AbstractStorage(ABC):
    @abstractmethod
    def upload_file(self, local_path: str, s3_path: str) -> None:
        raise NotImplemented

    @abstractmethod
    def download_file(self, s3_path: str) -> bytes:
        raise NotImplemented

    @abstractmethod
    def delete_file(self, s3_path: str) -> None:
        raise NotImplemented

    @abstractmethod
    def get_all_files(self, dir: str) -> list[str]:
        raise NotImplemented


class AbstractSneaker(ABC):
    def __init__(self, storage: AbstractStorage):
        self.storage = storage

    def preload_image(self, path: str) -> str:
        binary = self.storage.download_file(path)
        if len(binary) == 0:
            raise RuntimeError("")

        im = Image.frombytes(binary)

        with BytesIO() as buffer:
            im.save(buffer, "jpeg")
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            return f'<img src="data:image/jpeg;base64,{image_base64}">'


class AbstractParser(ABC):
    def __init__(
        self,
        website_name: str,
        collections_url: str,
        hostname_url: str,
        collections: list[str],
        index_columns: list[str],
    ):
        self.website_name = website_name
        self.collections_url = collections_url
        self.hostname_url = hostname_url
        self.collections = collections
        self.index_columns = index_columns
        self.headers = {"User-Agent": UserAgent().random}

    @abstractmethod
    def get_collection_info(collection: str) -> dict[str, Union[str, int]]:
        raise NotImplemented

    @abstractmethod
    def get_sneakers_urls(page_url: str) -> set[str]:
        raise NotImplemented

    @abstractmethod
    def get_sneakers_metadata(soup: BeautifulSoup) -> dict[str, str]:
        raise NotImplemented

    @abstractmethod
    def get_sneakers_images_urls(self, soup: BeautifulSoup) -> list[str]:
        raise NotImplemented

    def get_sneakers_images(self, soup: BeautifulSoup) -> list[tuple[bytes, str]]:
        images_urls = self.get_sneakers_images_urls(soup)
        images = []
        for image_url in images_urls:
            image_binary = requests.get(image_url, headers=self.headers).content
            image_ext = get_image_extension(image_url)
            images.append((image_binary, image_ext))

        return images

    def parse_sneakers(
        self, url: str, collection_info: dict[str, Union[int, str]], dir: str, s3: bool
    ) -> dict[str, str]:
        r = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(r.text, "html.parser")

        metadata = self.get_sneakers_metadata(soup)
        images = self.get_sneakers_images(soup)

        metadata["collection_name"] = collection_info["name"]
        metadata["collection_url"] = collection_info["url"]
        metadata["url"] = url

        website_dir = str(Path(dir, self.website_name))
        images_dir = str(Path(metadata["collection_name"], "images"))
        brand_dir = str(Path(metadata["brand"], metadata["title"]))
        save_dir = str(Path(website_dir, images_dir, brand_dir))

        images_dir, s3_dir = save_images(images, save_dir.lower(), s3)

        metadata["images_dir"] = images_dir
        metadata["s3_dir"] = s3_dir

        return metadata

    def parse_page(
        self, dir: str, collection_info: dict[str, Union[int, str]], page: int, s3: bool
    ) -> list[dict[str, str]]:
        metadata_page = []

        page_url = add_page(collection_info["url"], page)
        sneakers_urls = self.get_sneakers_urls(page_url)

        for sneakers_url in tqdm(sneakers_urls, leave=False):
            metadata = self.parse_sneakers(sneakers_url, collection_info, dir, s3)
            metadata_page.append(metadata)

        return metadata_page

    def parse_collection(
        self, dir: str, collection: str, s3: bool
    ) -> list[dict[str, str]]:
        metadata_collection = []

        collection_info = self.get_collection_info(collection)

        pbar = trange(1, collection_info["number_of_pages"] + 1, leave=False)
        for page in pbar:
            pbar.set_description(f"Page {page}")
            metadata_collection += self.parse_page(dir, collection_info, page, s3)

        website_dir = str(Path(dir, self.website_name))
        csv_path = str(Path(collection, "metadata.csv"))
        metadata_path = str(Path(website_dir, csv_path)).lower()

        save_metadata(metadata_collection, metadata_path, self.index_columns, s3)

        return metadata_collection

    def parse_website(self, dir: str, s3: bool) -> None:
        full_metadata = []

        bar = tqdm(self.collections)
        for collection in bar:
            bar.set_description(f"Collection: {collection}")
            full_metadata += self.parse_collection(dir, collection, s3)

        website_dir = str(Path(dir, self.website_name))
        metadata_path = str(Path(website_dir, "metadata.csv"))
        save_metadata(full_metadata, metadata_path, self.index_columns, s3)
        print(
            f"Collected {len(full_metadata)} sneakers from {self.website_name} website"
        )
