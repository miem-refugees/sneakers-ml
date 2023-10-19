from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse
import re
import os
from tqdm.auto import tqdm, trange
import pandas as pd
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

WEBSITE_NAME = "sneakerbaas"
COLLECTIONS_URL = "https://www.sneakerbaas.com/collections/sneakers/"
HOSTNAME_URL = "https://www.sneakerbaas.com/"
COLLECTIONS = [
    "category-kids",
    "category-unisex",
    "category-women",
    "category-men",
]
BUCKET = "sneakers-ml"


def get_collection_info(collection):
    info = {"name": collection}
    info["url"] = urljoin(COLLECTIONS_URL, collection)

    r = requests.get(info["url"], headers=HEADERS)
    soup = BeautifulSoup(r.text, "html.parser")

    products_string = soup.find(class_=re.compile("collection-size")).text.strip()

    info["number_of_products"] = int(re.search(r"\d+", products_string).group())
    info["number_of_pages"] = int(
        soup.find_all(class_=re.compile("(?<!\S)pagination(?!\S)"))[0]
        .find_all("span")[-2]
        .a.text
    )
    return info


def get_sneakers_urls(page_url):
    r = requests.get(page_url, headers=HEADERS)
    soup = BeautifulSoup(r.text, "html.parser")

    return set(
        [
            urljoin(HOSTNAME_URL, item["href"])
            for item in soup.find_all(href=re.compile("/collections/sneakers/products"))
        ]
    )


def get_sneakers_metadata(soup):
    metadata = {}

    meta_html = soup.find(name="div", class_="page-row-content").div.find_all(
        name="meta"
    )

    unused_keys = ["url", "image", "name"]
    for meta in meta_html[1:]:
        if meta.has_attr("itemprop") and meta["itemprop"] not in unused_keys:
            key = meta["itemprop"].lower().strip()
            metadata[key] = (
                meta["content"].replace("\xa0", " ").strip().replace("\n", " ")
            )

    # format metadata brand
    metadata["brand"] = fix_string_for_s3(metadata["brand"].lower())

    metadata["title"] = fix_string_for_s3(
        (soup.find(name="main", id="MainContent").find_all(name="span")[2].text.strip())
    )

    return metadata


def get_sneakers_images(soup):
    images = []
    images_section = soup.find_all(name="div", class_="swiper-slide product-image")

    for product_image in images_section:
        raw_image_html = product_image.find("a", {"data-fancybox": "productGallery"})
        raw_image_url = remove_query_from_url(
            remove_params_from_url(raw_image_html["href"])
        )
        image_url = add_https_to_url(raw_image_url)
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


def parse_sneakerbaas(path, s3, old_urls=None):
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
    parse_sneakerbaas(path="data", s3=False)
