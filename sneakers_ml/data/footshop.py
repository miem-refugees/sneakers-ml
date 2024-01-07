import asyncio
import json
import re
from pathlib import Path
from typing import Union

from bs4 import BeautifulSoup
from tqdm import tqdm

from sneakers_ml.data.base_parser import AbstractParser


class FootshopParser(AbstractParser):
    WEBSITE_NAME = "footshop"
    COLLECTIONS_URL = "https://www.footshop.eu/en/"
    HOSTNAME_URL = "https://www.footshop.eu/"
    COLLECTIONS = ["5-mens-shoes", "6-womens-shoes", "55-kids-sneakers-and-shoes"]
    INDEX_COLUMNS = ["url", "collection_name"]

    def get_collection_info(self, soup: BeautifulSoup) -> dict[str, Union[str, int]]:
        try:
            pagination = soup.findAll(class_=re.compile("PaginationLink_item"))[-2].text
        except Exception as e:
            tqdm.write(f"Pagination - {e}")
            pagination = 1
        info = {"number_of_pages": int(pagination)}
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

        metadata["brand"] = self.fix_html(brand_section["title"])
        metadata["title"] = self.fix_html(title_section.h1.text)
        metadata["color"] = self.fix_html(color_section.small.text)
        metadata["slug"] = self.get_slug(metadata["title"])
        metadata["brand_slug"] = self.get_slug(metadata["brand"])

        metadata["pricecurrency"] = self.fix_html(pricecurrency_section["content"])
        metadata["price"] = self.fix_html(price_section.text)

        return metadata

    def get_sneakers_images_urls(self, soup: BeautifulSoup) -> list[str]:
        images_urls = []

        script = soup.find(
            name="script",
            type="application/json",
            attrs={"data-hypernova-key": "ProductDetail"},
        )
        script = script.text.replace("-->", "").replace("<!--", "")[1:-1]

        script_cut = script[script.find("product_data") - 1 : script.find("last_image") - 2]

        script_json = json.loads("{" + script_cut + "}}")

        for image in script_json["product_data"]["images"]["other"]:
            image_url = self.fix_image_url(image["image"])
            images_urls.append(image_url)

        return images_urls


async def main():
    await FootshopParser(path=Path("data") / "raw", save_local=True, save_s3=False).parse_website()


if __name__ == "__main__":
    asyncio.run(main())
