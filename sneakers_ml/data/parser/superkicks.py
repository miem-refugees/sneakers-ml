import asyncio
import itertools
import json
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from tqdm import tqdm

from sneakers_ml.data.parser.base_parser import AbstractParser


class SuperkicksParser(AbstractParser):
    WEBSITE_NAME = "superkicks"
    COLLECTIONS_URL = "https://www.superkicks.in/collections/"
    HOSTNAME_URL = "https://www.superkicks.in/"
    COLLECTIONS = [
        f"{item[0]}-{item[1]}"
        for item in itertools.product(
            ["men", "women"], ["sneakers", "basketball-sneakers", "classics-sneakers", "skateboard-sneakers"]
        )
    ]
    INDEX_COLUMNS = ["url", "collection_name"]

    def get_collection_info(self, soup: BeautifulSoup) -> dict[str, str]:
        try:
            pagination = soup.find(name="nav", class_="pagination").ul.find_all(name="li")
            pagination = pagination[-2].a.text
        except Exception as e:
            tqdm.write(f"Pagination - {e}")
            pagination = 1
        info = {"number_of_pages": str(int(pagination))}
        return info

    def get_sneakers_urls(self, soup: BeautifulSoup) -> set[str]:
        products_section = soup.find_all(name="div", class_="card__information product-card2")
        sneakers_urls = [urljoin(self.HOSTNAME_URL, item.a["href"]) for item in products_section]
        return set(sneakers_urls)

    def get_sneakers_metadata(self, soup: BeautifulSoup) -> dict[str, str]:
        metadata = {}

        brand_section = soup.find(class_="product__text") or soup.find(class_="product__vendor")
        title_section = soup.find(class_="product__title").h1 or soup.find(class_="product__title")
        price_section = soup.find(name="span", class_="price-item price-item--regular")
        description_section = soup.find(name="div", class_="product__description")

        metadata["brand"] = self.fix_html(brand_section.text)
        metadata["title"] = self.fix_html(title_section.text)
        metadata["price"] = self.fix_html(price_section.text)
        metadata["description"] = self.fix_html(description_section.text)

        metadata["slug"] = self.get_slug(metadata["title"])
        metadata["brand_slug"] = self.get_slug(metadata["brand"])

        for span in soup.find_all(name="span", class_="product_description-name"):
            key = self.get_slug(span.contents[0].replace(" :", "").replace("_", " "))
            metadata[key] = self.fix_html(span.span.span.text)

        return metadata

    def get_sneakers_images_urls(self, soup: BeautifulSoup) -> list[str]:
        script = soup.findAll("script")[3].text.split("\n")[2]
        json_script = json.loads(script[script.find("{") : -1])
        images_urls = [self.fix_image_url(url) for url in json_script["images"]]
        return images_urls


async def main() -> None:
    await SuperkicksParser(path="data/raw", save_local=True, save_s3=False).parse_website()


if __name__ == "__main__":
    asyncio.run(main())
