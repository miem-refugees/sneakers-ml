from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse
import os
from tqdm.auto import tqdm, trange
import pandas as pd
import itertools
import boto3

from helper import (
    add_https_to_url,
    add_page_to_url,
    get_image_extension,
    get_max_file_name,
    remove_query_from_url,
    remove_params_from_url,
    get_parent_path,
    fix_string_for_s3,
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
BUCKET = "sneakers-ml"


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

    metadata["brand"] = fix_string_for_s3(
        soup.find(name="p", class_="product__text").text.strip()
    )
    metadata["title"] = fix_string_for_s3(
        soup.find(name="div", class_="product__title").h1.text.strip()
    )
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


def save_sneakers_images(images, path, collection, brand, title, s3):
    dir = os.path.join(
        path,
        WEBSITE_NAME,
        collection,
        "photos",
        brand,
        title,
    )
    os.makedirs(dir, exist_ok=True)
    s3_dir = ""

    i = get_max_file_name(dir)
    for image_binary, image_ext in images:
        i += 1
        image_path = os.path.join(dir, str(i) + image_ext)
        with open(image_path, "wb") as f:
            f.write(image_binary)

        if s3:
            s3_dir = upload_to_s3(image_path, image_path)

    return dir, s3_dir


def parse_sneakers(url, collection_info, path, s3):
    r = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(r.text, "html.parser")

    metadata = get_sneakers_metadata(soup)
    images = get_sneakers_images(soup)

    metadata["collection_name"] = collection_info["name"]
    metadata["collection_url"] = collection_info["url"]
    metadata["url"] = url

    photos_path, s3_path = save_sneakers_images(
        images,
        path,
        metadata["collection_name"],
        metadata["brand"],
        metadata["title"],
        s3,
    )

    metadata["photos_path"] = photos_path
    metadata["s3_path"] = s3_path

    return metadata


def save_metadata(metadata, path, collection, s3):
    dir = os.path.join(path, WEBSITE_NAME, collection, "metadata.csv")
    df = pd.DataFrame(metadata)
    df.to_csv(dir, index=False)

    if s3:
        upload_to_s3(dir, dir)


def upload_to_s3(file_to_upload, s3_path):
    s3 = boto3.resource(
        service_name="s3", endpoint_url="https://storage.yandexcloud.net"
    )
    s3.Bucket(BUCKET).upload_file(file_to_upload, s3_path)
    return (
        urlparse("")
        ._replace(scheme="s3", netloc=BUCKET, path=get_parent_path(s3_path))
        .geturl()
    )


def parse_superkicks(path, s3, old_urls=None):
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
            sneakers_urls = get_sneakers_urls(page_url)
            pbar.set_description(f"Page {page} | {page_url}")
            for sneakers_url in tqdm(sneakers_urls, leave=False):
                metadata = parse_sneakers(sneakers_url, collection_info, path, s3)
                metadata_collection.append(metadata)

        save_metadata(metadata_collection, path, collection, s3)

        full_metadata += metadata_collection

    save_metadata(full_metadata, path, "", s3)
    print(f"Collected {len(full_metadata)} sneakers from {WEBSITE_NAME} website")


if __name__ == "__main__":
    parse_superkicks(path="data", s3=False)
