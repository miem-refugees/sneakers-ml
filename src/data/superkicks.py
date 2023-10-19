from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
import os
from tqdm.auto import tqdm, trange
import pandas as pd
import itertools

from helper import (
    add_https_to_url,
    add_page_to_url,
    get_image_extension,
    get_max_file_name,
    remove_query_from_url,
    remove_params_from_url,
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


def get_collection_info(collection):
    info = {"name": collection}
    info["url"] = urljoin(COLLECTIONS_URL, collection)

    r = requests.get(info["url"], headers=HEADERS)
    soup = BeautifulSoup(r.text, "html.parser")

    info["number_of_pages"] = int(
        soup.find(name="nav", class_="pagination").ul.find_all(name="li")[-2].a.text
    )

    return info


def get_sneakers_urls(page_url):
    r = requests.get(page_url, headers=HEADERS)
    soup = BeautifulSoup(r.text, "html.parser")

    return set(
        [
            urljoin(HOSTNAME_URL, item.a["href"])
            for item in soup.find_all(
                name="div", class_="card__information product-card2"
            )
        ]
    )


def get_sneakers_metadata(soup):
    metadata = {}

    metadata["brand"] = soup.find(name="p", class_="product__text").text.strip()
    metadata["title"] = soup.find(name="div", class_="product__title").h1.text.strip()
    metadata["price"] = soup.find(
        name="span", class_="price-item price-item--regular"
    ).text.strip()

    for span in soup.find_all(name="span", class_="product_description-name"):
        key = span.contents[0].replace(" :", "").lower().replace(" ", "_")
        metadata[key] = span.span.span.text.strip()

    metadata["description"] = (
        soup.find_all(name="div", class_="product__description")[0]
        .text.strip()
        .replace("\n", " ")
    )

    return metadata


def get_sneakers_images(soup):
    images = []
    images_section = soup.find(name="div", class_="product-media-modal__content")

    for raw_image_url in images_section.find_all(name="img"):
        image_url = add_https_to_url(
            remove_query_from_url(remove_params_from_url(raw_image_url["src"]))
        )
        image_binary = requests.get(image_url).content
        image_ext = get_image_extension(image_url)
        images.append((image_binary, image_ext))

    return images


def save_sneakers_images_local(images, path, collection, brand, title):
    dir = os.path.join(
        path,
        WEBSITE_NAME,
        collection,
        "photos",
        brand,
        title,
    )
    os.makedirs(dir, exist_ok=True)

    i = get_max_file_name(dir)
    for image_binary, image_ext in images:
        i += 1
        with open(os.path.join(dir, str(i) + image_ext), "wb") as f:
            f.write(image_binary)

    return dir


def parse_sneakers(url, collection_info, path, s3=False):
    r = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(r.text, "html.parser")

    metadata = get_sneakers_metadata(soup)
    images = get_sneakers_images(soup)

    metadata["collection_name"] = collection_info["name"]
    metadata["collection_url"] = collection_info["url"]
    metadata["url"] = url

    photos_path = save_sneakers_images_local(
        images,
        path,
        metadata["collection_name"],
        metadata["brand"],
        metadata["title"],
    )  # or S3

    metadata["photos_path"] = photos_path

    return metadata


def save_metadata(metadata, path, collection="", s3=False):
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(path, WEBSITE_NAME, collection, "metadata.csv"), index=False)


def parse_superkicks(path="data", s3=False):
    print(f"{len(COLLECTIONS)} collections found")

    full_metadata = []

    bar = tqdm(COLLECTIONS)
    for collection in bar:
        metadata_collection = []
        collection_info = get_collection_info(collection)

        bar.set_description(
            f"Collection: {collection} | {collection_info['number_of_pages']}"
            f" pages | {collection_info['url']}"
        )

        pbar = trange(1, collection_info["number_of_pages"] + 1, leave=False)
        for page in pbar:
            page_url = add_page_to_url(collection_info["url"], page)
            pbar.set_description(f"Page {page} | {page_url}")
            sneakers_urls = get_sneakers_urls(page_url)
            for sneakers_url in tqdm(sneakers_urls, leave=False):
                metadata = parse_sneakers(sneakers_url, collection_info, path="data")
                metadata_collection.append(metadata)

        save_metadata(metadata_collection, path, collection)

        full_metadata += metadata_collection

    save_metadata(full_metadata, path)
    print(f"Collected {len(full_metadata)} sneakers from {WEBSITE_NAME} website")


if __name__ == "__main__":
    parse_superkicks()
