import asyncio
import re
from typing import Union
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from src.data.base_parser import AbstractParser


class SneakerbaasParser(AbstractParser):
    WEBSITE_NAME = "sneakerbaas"
    COLLECTIONS_URL = "https://www.sneakerbaas.com/collections/sneakers/"
    HOSTNAME_URL = "https://www.sneakerbaas.com/"
    COLLECTIONS = ["category-kids", "category-unisex", "category-women", "category-men"]
    INDEX_COLUMNS = ["url", "collection_name"]

    def get_collection_info(self, soup: BeautifulSoup) -> dict[str, Union[str, int]]:
        pagination = soup.find(class_=re.compile("(?<!\S)pagination(?!\S)"))
        info = {"number_of_pages": int(pagination.find_all("span")[-2].a.text)}
        return info

    def get_sneakers_urls(self, soup: BeautifulSoup) -> set[str]:
        products_section = soup.find_all(href=re.compile("/collections/sneakers/products"))
        sneakers_urls = [urljoin(self.HOSTNAME_URL, item["href"]) for item in products_section]
        return set(sneakers_urls)

    def get_sneakers_metadata(self, soup: BeautifulSoup) -> dict[str, str]:
        metadata = {}

        content_section = soup.find(name="div", class_="page-row-content")
        metadata_section = content_section.div.find_all(name="meta")
        title_section = soup.find(name="main", id="MainContent").find_all(name="span")

        unused_metadata_keys = ["url", "image", "name"]

        for meta in metadata_section[1:]:
            if meta.has_attr("itemprop") and meta["itemprop"] not in unused_metadata_keys:
                key = self.get_slug((meta["itemprop"]))
                metadata[key] = self.fix_html(meta["content"])

        metadata["title"] = self.fix_html(title_section[2].text)
        metadata["slug"] = self.get_slug(metadata["title"])
        metadata["brand_slug"] = self.get_slug(metadata["brand"])

        return metadata

    def get_sneakers_images_urls(self, soup: BeautifulSoup) -> list[str]:
        images_urls = []
        images_section = soup.find_all(name="div", class_="swiper-slide product-image")
        for section in images_section:
            image_section = section.find("a", {"data-fancybox": "productGallery"})
            image_url = self.add_https(self.remove_query(self.remove_params(image_section["href"])))
            images_urls.append(image_url)
        return images_urls


async def main():
    await SneakerbaasParser(path="data/raw_new", save_local=True, save_s3=False).parse_website()


if __name__ == "__main__":
    asyncio.run(main())
