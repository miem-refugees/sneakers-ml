from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
import re
from tqdm.auto import tqdm, trange
from typing import Union
from pathlib import Path
import json

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

WEBSITE_NAME = "footshop"
COLLECTIONS_URL = "https://www.footshop.eu/en/"
HOSTNAME_URL = "https://www.footshop.eu/"
COLLECTIONS = [
    "5-mens-shoes",
    "6-womens-shoes",
    "55-kids-sneakers-and-shoes",
]
INDEX_COLUMN = ["url", "collection_name"]


def get_collection_info(collection: str) -> dict[str, Union[str, int]]:
    info = {"name": collection}
    info["url"] = urljoin(COLLECTIONS_URL, collection)

    r = requests.get(info["url"], headers=HEADERS)
    soup = BeautifulSoup(r.text, "html.parser")

    pagination = soup.findAll(class_=re.compile("PaginationLink_item"))

    info["number_of_pages"] = int(pagination[-2].text)

    return info


def get_sneakers_urls(page_url: str) -> set[str]:
    r = requests.get(page_url, headers=HEADERS)
    soup = BeautifulSoup(r.text, "html.parser")

    products_section = soup.findAll(name="div", itemprop="itemListElement")
    sneakers_urls = [item.find(itemprop="url")["content"] for item in products_section]

    return set(sneakers_urls)


def get_sneakers_metadata(soup: BeautifulSoup) -> dict[str, str]:
    metadata = {}

    properties_section = soup.findAll(class_=re.compile("^ProductProperties_"))[2]
    brand_section = properties_section.div.a
    title_section = properties_section.find(class_=re.compile("Headline_wrapper_"))
    color_section = properties_section.find(class_=re.compile("Headline_wrapper_"))

    meta_section = soup.find(name="div", class_=re.compile("Product_productProperties"))
    # description_section = meta_section.find(name="meta", itemprop="description")
    pricecurrency_section = meta_section.find(name="meta", itemprop="priceCurrency")
    price_section = soup.find(name="strong", class_=re.compile("Properties_priceValue"))

    # format metadata as it is used as folder names
    metadata["brand"] = fix_string(brand_section["title"])
    metadata["title"] = fix_string(title_section.h1.text)
    metadata["color"] = fix_html_text(color_section.small.text)

    # metadata["description"] = fix_html_text(description_section["content"])
    metadata["pricecurrency"] = fix_html_text(pricecurrency_section["content"])
    metadata["price"] = fix_html_text(price_section.text)

    return metadata


def get_sneakers_images(soup: BeautifulSoup) -> list[tuple[bytes, str]]:
    images = []

    script_section = (
        soup.find(
            name="script",
            type="application/json",
            attrs={"data-hypernova-key": "ProductDetail"},
        )
        .text.replace("-->", "")
        .replace("<!--", "")[1:-1]
    )

    script_cut = script_section[
        script_section.find("product_data") - 1 : script_section.find("last_image") - 2
    ]

    script_json = json.loads("{" + script_cut + "}}")

    for image in script_json["product_data"]["images"]["other"]:
        image_url = add_https(remove_query(remove_params(image["image"])))
        image_binary = requests.get(image_url).content
        image_ext = get_image_extension(image_url)
        images.append((image_binary, image_ext))

    return images


def parse_sneakers(
    url: str, collection_info: dict[str, Union[int, str]], dir: str, s3: bool
) -> dict[str, str]:
    r = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(r.text, "html.parser")

    metadata = get_sneakers_metadata(soup)
    images = get_sneakers_images(soup)

    metadata["collection_name"] = collection_info["name"]
    metadata["collection_url"] = collection_info["url"]
    metadata["url"] = url

    website_dir = str(Path(dir, WEBSITE_NAME))
    images_dir = str(Path(metadata["collection_name"], "images"))
    brand_dir = str(Path(metadata["brand"], metadata["title"]))
    save_dir = str(Path(website_dir, images_dir, brand_dir))

    images_dir, s3_dir = save_images(images, save_dir.lower(), s3)

    metadata["images_dir"] = images_dir
    metadata["s3_dir"] = s3_dir

    return metadata


def parse_sneakerbaas_page(
    dir: str, collection_info: dict[str, Union[int, str]], page: int, s3: bool
) -> list[dict[str, str]]:
    metadata_page = []
    page_url = add_page(collection_info["url"], page)
    sneakers_urls = get_sneakers_urls(page_url)

    for sneakers_url in tqdm(sneakers_urls, leave=False):
        metadata = parse_sneakers(sneakers_url, collection_info, dir, s3)
        metadata_page.append(metadata)

    return metadata_page


def parse_sneakerbaas_collection(
    dir: str, collection: str, s3: bool
) -> list[dict[str, str]]:
    metadata_collection = []
    collection_info = get_collection_info(collection)

    pbar = trange(1, collection_info["number_of_pages"] + 1, leave=False)
    for page in pbar:
        pbar.set_description(f"Page {page}")
        metadata_collection += parse_sneakerbaas_page(dir, collection_info, page, s3)

    website_dir = str(Path(dir, WEBSITE_NAME))
    csv_path = str(Path(collection, "metadata.csv"))
    metadata_path = str(Path(website_dir, csv_path.lower()))

    save_metadata(metadata_collection, metadata_path, INDEX_COLUMN, s3)

    return metadata_collection


def parse_sneakerbaas(dir: str, s3: bool) -> None:
    full_metadata = []

    bar = tqdm(COLLECTIONS)
    for collection in bar:
        bar.set_description(f"Collection: {collection}")
        full_metadata += parse_sneakerbaas_collection(dir, collection, s3)

    website_dir = str(Path(dir, WEBSITE_NAME))
    metadata_path = str(Path(website_dir, "metadata.csv"))
    save_metadata(full_metadata, metadata_path, INDEX_COLUMN, s3)
    print(f"Collected {len(full_metadata)} sneakers from {WEBSITE_NAME} website")


if __name__ == "__main__":
    parse_sneakerbaas(dir=str(Path("data", "raw")), s3=False)
