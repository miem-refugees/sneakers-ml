from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
import re
import os
from tqdm.auto import tqdm, trange
from typing import Union

from helper import (
    add_https,
    add_page,
    get_image_extension,
    remove_query,
    remove_params,
    fix_path_for_s3,
    fix_html_text,
    save_images,
    save_metadata,
    HEADERS,
)

WEBSITE_NAME = "sneakerbaas"
COLLECTIONS_URL = "https://www.sneakerbaas.com/collections/sneakers/"
HOSTNAME_URL = "https://www.sneakerbaas.com/"
COLLECTIONS = [
    "category-kids",
    "category-unisex",
    "category-women",
    "category-men",
]
INDEX_COLUMN = ["url", "collection_name"]


def get_collection_info(collection: str) -> dict[str, Union[str, int]]:
    info = {"name": collection}
    info["url"] = urljoin(COLLECTIONS_URL, collection)

    r = requests.get(info["url"], headers=HEADERS)
    soup = BeautifulSoup(r.text, "html.parser")

    products = soup.find(class_=re.compile("collection-size")).text.strip()
    pagination = soup.find(class_=re.compile("(?<!\S)pagination(?!\S)"))

    info["number_of_products"] = int(re.search(r"\d+", products).group())
    info["number_of_pages"] = int(pagination.find_all("span")[-2].a.text)

    return info


def get_sneakers_urls(page_url: str) -> set[str]:
    r = requests.get(page_url, headers=HEADERS)
    soup = BeautifulSoup(r.text, "html.parser")

    products_section = soup.find_all(href=re.compile("/collections/sneakers/products"))
    sneakers_urls = [urljoin(HOSTNAME_URL, item["href"]) for item in products_section]

    return set(sneakers_urls)


def get_sneakers_metadata(soup: BeautifulSoup) -> dict[str, str]:
    metadata = {}

    content_section = soup.find(name="div", class_="page-row-content")
    metadata_section = content_section.div.find_all(name="meta")
    title_section = soup.find(name="main", id="MainContent").find_all(name="span")

    unused_metadata_keys = ["url", "image", "name"]

    for meta in metadata_section[1:]:
        if meta.has_attr("itemprop") and meta["itemprop"] not in unused_metadata_keys:
            key = meta["itemprop"].lower().strip()
            metadata[key] = fix_html_text(meta["content"])

    # format metadata as it is used as folder names
    metadata["brand"] = fix_path_for_s3(metadata["brand"])
    metadata["title"] = fix_path_for_s3(title_section[2].text.strip())

    return metadata


def get_sneakers_images(soup: BeautifulSoup) -> list[tuple[bytes, str]]:
    images = []

    images_section = soup.find_all(name="div", class_="swiper-slide product-image")

    for section in images_section:
        image_section = section.find("a", {"data-fancybox": "productGallery"})
        image_url = add_https(remove_query(remove_params(image_section["href"])))
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

    website_dir = os.path.join(dir, WEBSITE_NAME)
    images_dir = os.path.join(metadata["collection_name"], "images")
    brand_dir = os.path.join(metadata["brand"], metadata["title"])
    save_dir = os.path.join(website_dir, images_dir, brand_dir)

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

    website_dir = os.path.join(dir, WEBSITE_NAME)
    csv_path = os.path.join(collection, "metadata.csv")
    metadata_path = os.path.join(website_dir, csv_path.lower())

    save_metadata(metadata_collection, metadata_path, INDEX_COLUMN, s3)

    return metadata_collection


def parse_sneakerbaas(dir: str, s3: bool) -> None:
    full_metadata = []

    bar = tqdm(COLLECTIONS)
    for collection in bar:
        bar.set_description(f"Collection: {collection}")
        full_metadata += parse_sneakerbaas_collection(dir, collection, s3)

    save_metadata(full_metadata, dir, INDEX_COLUMN, s3)
    print(f"Collected {len(full_metadata)} sneakers from {WEBSITE_NAME} website")


if __name__ == "__main__":
    parse_sneakerbaas(dir="data", s3=False)
