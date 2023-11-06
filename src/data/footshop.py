import json
import re
from typing import Union

from bs4 import BeautifulSoup

from src.data.base_parser_sync import AbstractParser
from src.data.helper import add_https, remove_query, remove_params, fix_string, fix_html_text


class FootshopParser(AbstractParser):
    WEBSITE_NAME = "footshop"
    COLLECTIONS_URL = "https://www.footshop.eu/en/"
    HOSTNAME_URL = "https://www.footshop.eu/"
    COLLECTIONS = ["5-mens-shoes", "6-womens-shoes", "55-kids-sneakers-and-shoes"]
    INDEX_COLUMNS = ["url", "collection_name"]

    def get_collection_info(self, soup: BeautifulSoup) -> dict[str, Union[str, int]]:
        pagination = soup.findAll(class_=re.compile("PaginationLink_item"))
        info = {"number_of_pages": int(pagination[-2].text)}
        return info

    def get_sneakers_urls(self, soup: BeautifulSoup) -> set[str]:
        products_section = soup.findAll(name="div", itemprop="itemListElement")
        sneakers_urls = [item.find(itemprop="url")["content"] for item in products_section]
        return set(sneakers_urls)

    def get_sneakers_metadata(self, soup: BeautifulSoup) -> dict[str, str]:
        metadata = {}

        properties_section = soup.findAll(class_=re.compile("^ProductProperties_"))[2]
        brand_section = properties_section.div.a
        title_section = properties_section.find(class_=re.compile("Headline_wrapper_"))
        color_section = properties_section.find(class_=re.compile("Headline_wrapper_"))

        meta_section = soup.find(name="div", class_=re.compile("Product_productProperties"))
        pricecurrency_section = meta_section.find(name="meta", itemprop="priceCurrency")
        price_section = soup.find(name="strong", class_=re.compile("Properties_priceValue"))

        # format metadata as it is used as folder names
        metadata["brand"] = fix_string(brand_section["title"])
        metadata["title"] = fix_string(title_section.h1.text)
        metadata["color"] = fix_html_text(color_section.small.text)

        metadata["pricecurrency"] = fix_html_text(pricecurrency_section["content"])
        metadata["price"] = fix_html_text(price_section.text)

        return metadata

    def get_sneakers_images_urls(self, soup: BeautifulSoup) -> list[str]:
        images_urls = []

        script = soup.find(name="script", type="application/json", attrs={"data-hypernova-key": "ProductDetail"}, )
        script = script.text.replace("-->", "").replace("<!--", "")[1:-1]

        script_cut = script[script.find("product_data") - 1: script.find("last_image") - 2]

        script_json = json.loads("{" + script_cut + "}}")

        for image in script_json["product_data"]["images"]["other"]:
            image_url = add_https(remove_query(remove_params(image["image"])))
            images_urls.append(image_url)

        return images_urls


if __name__ == "__main__":
    FootshopParser(path="data/raw", save_local=True, save_s3=True).parse_website()
