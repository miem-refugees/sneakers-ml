import re
from typing import Union
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from base_parser import AbstractParser
from helper import (add_https, remove_query, remove_params, fix_string, fix_html_text, )


class SneakerbaasParser(AbstractParser):
    WEBSITE_NAME = "sneakerbaas"
    COLLECTIONS_URL = "https://www.sneakerbaas.com/collections/sneakers/"
    HOSTNAME_URL = "https://www.sneakerbaas.com/"
    COLLECTIONS = ["category-kids", "category-unisex", "category-women", "category-men", ]
    INDEX_COLUMNS = ["url", "collection_name"]

    def get_collection_info(self, collection: str) -> dict[str, Union[str, int]]:
        info = {"name": collection, "url": urljoin(self.COLLECTIONS_URL, collection)}
        r = requests.get(info["url"], headers=self.headers)
        soup = BeautifulSoup(r.text, "html.parser")

        products = soup.find(class_=re.compile("collection-size")).text.strip()
        pagination = soup.find(class_=re.compile("(?<!\S)pagination(?!\S)"))

        info["number_of_products"] = int(re.search(r"\d+", products).group())
        info["number_of_pages"] = int(pagination.find_all("span")[-2].a.text)

        return info

    def get_sneakers_urls(self, page_url: str) -> set[str]:
        r = requests.get(page_url, headers=self.headers)
        soup = BeautifulSoup(r.text, "html.parser")

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
            if (meta.has_attr("itemprop") and meta["itemprop"] not in unused_metadata_keys):
                key = fix_string(meta["itemprop"])
                metadata[key] = fix_html_text(meta["content"])

        # format metadata as it is used as folder names
        metadata["brand"] = fix_string(metadata["brand"])
        metadata["title"] = fix_string(title_section[2].text)

        return metadata

    def get_sneakers_images_urls(self, soup: BeautifulSoup) -> list[str]:
        images_urls = []
        images_section = soup.find_all(name="div", class_="swiper-slide product-image")
        for section in images_section:
            image_section = section.find("a", {"data-fancybox": "productGallery"})
            image_url = add_https(remove_query(remove_params(image_section["href"])))
            images_urls.append(image_url)
        return images_urls


if __name__ == "__main__":
    SneakerbaasParser(path="data/raw", save_local=True, save_s3=False, ).parse_website()
