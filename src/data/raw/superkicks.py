from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
import os
from tqdm.auto import tqdm, trange
import itertools
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


WEBSITE_NAME = "superkicks"
COLLECTIONS_URL = "https://www.superkicks.in/collections/"
HOSTNAME_URL = "https://www.superkicks.in/"
COLLECTIONS = [
    f"{item[0]}-{item[1]}"
    for item in itertools.product(
        ["men", "women"],
        ["sneakers", "basketball-sneakers", "classics-sneakers", "skateboard-sneakers"],
    )
]
INDEX_COLUMN = ["url", "collection_name"]


def get_collection_info(collection: str) -> dict[str, Union[str, int]]:
    info = {"name": collection}
    info["url"] = urljoin(COLLECTIONS_URL, collection)

    r = requests.get(info["url"], headers=HEADERS)
    soup = BeautifulSoup(r.text, "html.parser")

    pagination = soup.find(name="nav", class_="pagination").ul.find_all(name="li")

    info["number_of_pages"] = int(pagination[-2].a.text)

    return info


def get_sneakers_urls(page_url: str) -> set[str]:
    r = requests.get(page_url, headers=HEADERS)
    soup = BeautifulSoup(r.text, "html.parser")

    products_section = soup.find_all(
        name="div", class_="card__information product-card2"
    )
    sneakers_urls = [urljoin(HOSTNAME_URL, item.a["href"]) for item in products_section]

    return set(sneakers_urls)


def get_sneakers_metadata(soup: BeautifulSoup) -> dict[str, str]:
    metadata = {}

    brand_section = soup.find(name="p", class_="product__text")
    title_section = soup.find(name="div", class_="product__title").h1
    price_section = soup.find(name="span", class_="price-item price-item--regular")
    description_section = soup.find(name="div", class_="product__description")

    metadata["brand"] = fix_path_for_s3(brand_section.text.strip())
    metadata["title"] = fix_path_for_s3(title_section.text.strip())
    metadata["price"] = price_section.text.strip()

    for span in soup.find_all(name="span", class_="product_description-name"):
        key = span.contents[0].replace(" :", "").lower().replace(" ", "_")
        metadata[key] = span.span.span.text.strip()

    metadata["description"] = description_section.text.strip().replace("\n", " ")

    return metadata


def get_sneakers_images(soup: BeautifulSoup) -> list[tuple[bytes, str]]:
    images = []

    images_section = soup.find(name="div", class_="product-media-modal__content")

    for raw_image_url in images_section.find_all(name="img"):
        image_url = add_https(remove_query(remove_params(raw_image_url["src"])))
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


def parse_superkicks_page(
    dir: str, collection_info: dict[str, Union[int, str]], page: int, s3: bool
) -> list[dict[str, str]]:
    metadata_page = []
    page_url = add_page(collection_info["url"], page)
    sneakers_urls = get_sneakers_urls(page_url)

    for sneakers_url in tqdm(sneakers_urls, leave=False):
        metadata = parse_sneakers(sneakers_url, collection_info, dir, s3)
        metadata_page.append(metadata)

    return metadata_page


def parse_superkicks_collection(
    dir: str, collection: str, s3: bool
) -> list[dict[str, str]]:
    metadata_collection = []
    collection_info = get_collection_info(collection)

    pbar = trange(1, collection_info["number_of_pages"] + 1, leave=False)
    for page in pbar:
        pbar.set_description(f"Page {page}")
        metadata_collection += parse_superkicks_page(dir, collection_info, page, s3)

    website_dir = os.path.join(dir, WEBSITE_NAME)
    csv_path = os.path.join(collection, "metadata.csv")
    metadata_path = os.path.join(website_dir, csv_path.lower())

    save_metadata(metadata_collection, metadata_path, INDEX_COLUMN, s3)

    return metadata_collection


def parse_superkicks(dir: str, s3: bool) -> None:
    full_metadata = []

    bar = tqdm(COLLECTIONS)
    for collection in bar:
        bar.set_description(f"Collection: {collection}")
        full_metadata += parse_superkicks_collection(dir, collection, s3)

    save_metadata(full_metadata, dir, INDEX_COLUMN, s3)
    print(f"Collected {len(full_metadata)} sneakers from {WEBSITE_NAME} website")


if __name__ == "__main__":
    parse_superkicks(dir="data", s3=False)
