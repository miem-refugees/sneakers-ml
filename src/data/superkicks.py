import itertools
from typing import Union
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from base_parser import AbstractParser
from helper import add_https, remove_query, remove_params, fix_string, fix_html_text


class SuperkicksParser(AbstractParser):
    WEBSITE_NAME = "superkicks"
    COLLECTIONS_URL = "https://www.superkicks.in/collections/"
    HOSTNAME_URL = "https://www.superkicks.in/"
    COLLECTIONS = [f"{item[0]}-{item[1]}" for item in itertools.product(["men", "women"],
                                                                        ["sneakers", "basketball-sneakers",
                                                                         "classics-sneakers",
                                                                         "skateboard-sneakers", ], )]
    INDEX_COLUMNS = ["url", "collection_name"]

    def get_collection_info(self, collection: str) -> dict[str, Union[str, int]]:
        info = {"name": collection, "url": urljoin(self.COLLECTIONS_URL, collection)}

        r = requests.get(info["url"], headers=self.headers)
        soup = BeautifulSoup(r.text, "html.parser")

        pagination = soup.find(name="nav", class_="pagination").ul.find_all(name="li")

        info["number_of_pages"] = int(pagination[-2].a.text)

        return info

    def get_sneakers_urls(self, page_url: str) -> set[str]:
        r = requests.get(page_url, headers=self.headers)
        soup = BeautifulSoup(r.text, "html.parser")

        products_section = soup.find_all(name="div", class_="card__information product-card2")
        sneakers_urls = [urljoin(self.HOSTNAME_URL, item.a["href"]) for item in products_section]

        return set(sneakers_urls)

    def get_sneakers_metadata(self, soup: BeautifulSoup) -> dict[str, str]:
        metadata = {}

        brand_section = soup.find(name="p", class_="product__text")
        title_section = soup.find(name="div", class_="product__title").h1
        price_section = soup.find(name="span", class_="price-item price-item--regular")
        description_section = soup.find(name="div", class_="product__description")

        metadata["brand"] = fix_string(brand_section.text)
        metadata["title"] = fix_string(title_section.text)
        metadata["price"] = fix_html_text(price_section.text)

        for span in soup.find_all(name="span", class_="product_description-name"):
            key = span.contents[0].replace(" :", "").lower().replace(" ", "_")
            metadata[key] = fix_html_text(span.span.span.text)

        metadata["description"] = fix_html_text(description_section.text)

        return metadata

    def get_sneakers_images_urls(self, soup: BeautifulSoup) -> list[str]:
        images_urls = []

        images_section = soup.find(name="div", class_="product-media-modal__content")

        for raw_image_url in images_section.find_all(name="img"):
            image_url = add_https(remove_query(remove_params(raw_image_url["src"])))
            images_urls.append(image_url)

        return images_urls


if __name__ == "__main__":
    SuperkicksParser(path="data/raw", save_local=True, save_s3=False).parse_website()
