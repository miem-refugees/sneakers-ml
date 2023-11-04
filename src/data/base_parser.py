from typing import Union
from abc import ABC, abstractmethod
from pathlib import Path
from bs4 import BeautifulSoup
import requests
from helper import (
    add_page,
    get_image_extension,
)
from fake_useragent import UserAgent
from tqdm.auto import tqdm, trange
from s3 import S3Storage
from local import LocalStorage
from base import AbstractStorage
import pandas as pd
from storage import images_to_storage, metadata_to_storage


class AbstractParser(ABC):
    def __init__(
        self,
        website_name: str,
        collections_url: str,
        hostname_url: str,
        collections: list[str],
        index_columns: list[str],
        path: str,
        save_local: bool,
        save_s3: bool,
    ):
        self.website_name = website_name
        self.collections_url = collections_url
        self.hostname_url = hostname_url
        self.collections = collections
        self.index_columns = index_columns

        self.path = path
        self.save_local = save_local
        self.save_s3 = save_s3

        self.headers = {"User-Agent": UserAgent().random}
        self.website_path = str(Path(self.path, self.website_name))

    @abstractmethod
    def get_collection_info(self, collection: str) -> dict[str, Union[str, int]]:
        raise NotImplemented

    @abstractmethod
    def get_sneakers_urls(self, page_url: str) -> set[str]:
        raise NotImplemented

    @abstractmethod
    def get_sneakers_metadata(self, soup: BeautifulSoup) -> dict[str, str]:
        raise NotImplemented

    @abstractmethod
    def get_sneakers_images_urls(self, soup: BeautifulSoup) -> list[str]:
        raise NotImplemented

    def get_sneakers_images(self, images_urls: list[str]) -> list[tuple[bytes, str]]:
        images = []
        for image_url in images_urls:
            image_binary = requests.get(image_url, headers=self.headers).content
            image_ext = get_image_extension(image_url)
            images.append((image_binary, image_ext))

        return images

    def parse_sneakers(
        self, url: str, collection_info: dict[str, Union[int, str]]
    ) -> dict[str, str]:
        """
        Parses metadata and images of one pair of sneakers
        """
        r = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(r.text, "html.parser")

        metadata = self.get_sneakers_metadata(soup)
        images_urls = self.get_sneakers_images_urls(soup)
        images = self.get_sneakers_images(images_urls)

        metadata["collection_name"] = collection_info["name"]
        metadata["collection_url"] = collection_info["url"]
        metadata["url"] = url

        images_path = str(Path(metadata["collection_name"], "images"))
        brand_path = str(Path(metadata["brand"], metadata["title"]))
        save_path = str(Path(self.website_path, images_path, brand_path)).lower()

        self.save_images(images, save_path)

        metadata["images_path"] = save_path

        return metadata

    def parse_page(
        self, collection_info: dict[str, Union[int, str]], page: int
    ) -> list[dict[str, str]]:
        metadata_page = []

        page_url = add_page(collection_info["url"], page)
        sneakers_urls = self.get_sneakers_urls(page_url)

        for sneakers_url in tqdm(sneakers_urls, leave=False):
            metadata = self.parse_sneakers(
                sneakers_url, collection_info, self.path_prefix
            )
            metadata_page.append(metadata)

        return metadata_page

    def parse_collection(self, collection: str) -> list[dict[str, str]]:
        metadata_collection = []

        collection_info = self.get_collection_info(collection)

        pbar = trange(1, collection_info["number_of_pages"] + 1, leave=False)
        for page in pbar:
            pbar.set_description(f"Page {page}")
            metadata_collection += self.parse_page(collection_info, page)

        csv_path = str(Path(collection, "metadata.csv"))
        metadata_path = str(Path(self.website_path, csv_path)).lower()  # todo

        self.save_metadata(metadata_collection, metadata_path, self.index_columns)

        return metadata_collection

    def parse_website(self) -> list[dict[str, str]]:
        full_metadata = []

        bar = tqdm(self.collections)
        for collection in bar:
            bar.set_description(f"Collection: {collection}")
            full_metadata += self.parse_collection(collection)

        metadata_path = str(Path(self.website_path, "metadata.csv"))
        self.save_metadata(full_metadata, metadata_path, self.index_columns)
        print(
            f"Collected {len(full_metadata)} sneakers from {self.website_name} website"
        )
        return full_metadata

    def save_images(self, images: tuple[bytes, str], path: str):
        if self.save_local:
            Path(path).mkdir(parents=True, exist_ok=True)
            images_to_storage(LocalStorage(), images, path)
        if self.save_s3:
            images_to_storage(S3Storage(), images, path)

    def save_metadata(
        self,
        metadata: dict[str, str],
        path: str,
        index_columns: str,
    ) -> None:
        """
        Saves metadata dict in .csv format in path. If .csv already exists, concats the data and
        removes duplicates by index_column.
        """
        if self.save_local:
            metadata_to_storage(LocalStorage(), metadata, path, index_columns)
        if self.save_s3:
            metadata_to_storage(S3Storage(), metadata, path, index_columns)
