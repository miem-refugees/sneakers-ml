import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union
from urllib.parse import urljoin

import aiohttp
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from tqdm import trange, tqdm
from tqdm.asyncio import tqdm as tqdm_async

from src.data.helper import add_page, get_image_extension
from src.data.local import LocalStorage
from src.data.s3 import S3Storage
from src.data.storage import StorageProcessor


class AbstractParser(ABC):
    WEBSITE_NAME: str
    COLLECTIONS_URL: str
    HOSTNAME_URL: str
    COLLECTIONS: list[str]
    INDEX_COLUMNS: list[str]

    def __init__(self, path: str, save_local: bool, save_s3: bool):
        self.path = path
        self.save_local = save_local
        self.save_s3 = save_s3

        self.headers = {"User-Agent": UserAgent().random}
        self.website_path = str(Path(self.path, self.WEBSITE_NAME))
        self.parser = "html.parser"

    async def get_soup(self, url: str) -> BeautifulSoup:
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.get(url) as resp:
                text = await resp.text()
        soup = BeautifulSoup(text, self.parser)
        return soup

    async def get_image(self, image_url: str) -> bytes:
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.get(image_url) as resp:
                return await resp.read()

    @abstractmethod
    def get_collection_info(self, soup: BeautifulSoup) -> dict[str, Union[str, int]]:
        raise NotImplemented

    @abstractmethod
    def get_sneakers_urls(self, soup: BeautifulSoup) -> set[str]:
        raise NotImplemented

    @abstractmethod
    def get_sneakers_metadata(self, soup: BeautifulSoup) -> dict[str, str]:
        raise NotImplemented

    @abstractmethod
    def get_sneakers_images_urls(self, soup: BeautifulSoup) -> list[str]:
        raise NotImplemented

    async def get_sneakers_images(self, images_urls: list[str]) -> list[tuple[bytes, str]]:
        async def download_image(image_url: str) -> tuple[bytes, str]:
            image_binary = await self.get_image(image_url)
            image_ext = get_image_extension(image_url)
            return image_binary, image_ext

        images = await asyncio.gather(*[download_image(image_url) for image_url in images_urls])
        return images

    async def parse_sneakers(self, url: str, collection_info: dict[str, Union[int, str]]) -> dict[str, str]:
        """
        Parses metadata and images of one pair of sneakers
        """
        for attempt in range(2):
            try:
                soup = await self.get_soup(url)
                metadata = self.get_sneakers_metadata(soup)
                images_urls = self.get_sneakers_images_urls(soup)
                images = await self.get_sneakers_images(images_urls)

                metadata["collection_name"] = collection_info["name"]
                metadata["collection_url"] = collection_info["url"]
                metadata["url"] = url

                model_path = str(Path(metadata["collection_name"], metadata["brand"], metadata["title"]))
                save_path = str(Path(self.website_path, model_path)).lower()

                self.save_images(images, save_path)
                metadata["images_path"] = save_path

                if attempt == 1:
                    print("RETRY: OK")

                return metadata
            except Exception as e:
                print(e, url)

        print("RETRY: FAIL")
        return {}

    async def parse_page(self, collection_info: dict[str, Union[int, str]], page: int) -> list[dict[str, str]]:
        page_url = add_page(collection_info["url"], page)
        soup = await self.get_soup(page_url)
        sneakers_urls = self.get_sneakers_urls(soup)

        metadata_page = await tqdm_async.gather(
            *[self.parse_sneakers(sneakers_url, collection_info) for sneakers_url in sneakers_urls], leave=False)
        return list(filter(None, metadata_page))

    async def parse_collection(self, collection: str) -> list[dict[str, str]]:
        metadata_collection = []

        collection_url = urljoin(self.COLLECTIONS_URL, collection)
        soup = await self.get_soup(collection_url)
        collection_info = self.get_collection_info(soup)

        collection_info["name"] = collection
        collection_info["url"] = collection_url

        pbar = trange(1, collection_info["number_of_pages"] + 1, leave=False)
        for page in pbar:
            pbar.set_description(f"Page {page}")
            metadata_collection += await self.parse_page(collection_info, page)

        return metadata_collection

    async def parse_website(self) -> list[dict[str, str]]:
        full_metadata = []

        bar = tqdm(self.COLLECTIONS)
        for collection in bar:
            bar.set_description(f"Collection: {collection}")
            full_metadata += await self.parse_collection(collection)

        metadata_path = str(Path(self.website_path, "metadata.csv"))
        self.save_metadata(full_metadata, metadata_path, self.INDEX_COLUMNS)
        print(f"Collected {len(full_metadata)} sneakers from {self.WEBSITE_NAME} website")
        return full_metadata

    def save_images(self, images: list[tuple[bytes, str]], directory: str) -> None:
        if self.save_local:
            StorageProcessor(LocalStorage()).images_to_storage(images, directory)
        if self.save_s3:
            StorageProcessor(S3Storage()).images_to_storage(images, directory)

    def save_metadata(self, metadata: list[dict[str, str]], path: str, index_columns: list[str]) -> None:
        if self.save_local:
            StorageProcessor(LocalStorage()).metadata_to_storage(metadata, path, index_columns)
        if self.save_s3:
            StorageProcessor(S3Storage()).metadata_to_storage(metadata, path, index_columns)
