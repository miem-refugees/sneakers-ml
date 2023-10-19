from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
import re
import os
from tqdm.auto import tqdm, trange
import pandas as pd

from .helper import (
    add_https_to_url,
    add_page_to_url,
    get_image_extension,
    get_max_file_name,
    remove_query_from_url,
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


def get_collection_info(collection=""):
    info = {"url": urljoin(COLLECTIONS_URL, collection)}
    r = requests.get(info["url"], headers=HEADERS)
    soup = BeautifulSoup(r.text, "html.parser")

    products_string = soup.find_all(class_=re.compile("collection-size"))[
        0
    ].text.strip()
    info["number_of_products"] = int(re.search(r"\d+", products_string).group())
    info["number_of_pages"] = int(
        soup.find_all(class_=re.compile("(?<!\S)pagination(?!\S)"))[0]
        .find_all("span")[-2]
        .a.text
    )
    return info


def get_sneakers_urls(url, sneakers_url_path="/collections/sneakers/products"):
    r = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(r.text, "html.parser")

    return set(
        [
            HOSTNAME_URL + item["href"]
            for item in soup.find_all(href=re.compile(sneakers_url_path))
        ]
    )


def get_sneakers_metadata(url, soup, collection):
    # metadata
    metadata_keys = ["brand", "description", "priceCurrency", "price"]
    meta_html = soup.find_all(name="div", class_="page-row-content")[0].div.find_all(
        name="meta"
    )
    metadata = {"url": url, "collection": collection}
    for meta in meta_html:
        if meta.has_attr("itemprop"):
            if meta["itemprop"] in metadata_keys:
                metadata[meta["itemprop"]] = meta["content"].replace("\xa0", " ")

    # default_fields = ["Description", "Colors", "Stijlcode"]
    # for i, item in enumerate(metadata["description"].split("- ")[1:]):
    #     metadata[default_fields[i]] = item
    # metadata.pop("description")

    # format metadata brand
    metadata["brand"] = metadata["brand"].lower()

    # title
    metadata["title"] = (
        soup.find(name="main", id="MainContent").find_all(name="span")[2].text
    )
    return metadata


def get_sneakers_images(soup):
    images = []
    images_section = soup.find_all(name="div", class_="swiper-slide product-image")
    for product_image in images_section:
        raw_image_url = product_image.find("a", {"data-fancybox": "productGallery"})[
            "href"
        ]
        image_url = add_https_to_url(remove_query_from_url(raw_image_url))
        image_binary = requests.get(image_url).content
        image_ext = get_image_extension(image_url)
        images.append((image_binary, image_ext))
    return images


def save_sneakers_images_local(images, metadata, path="data"):
    dir = os.path.join(
        path,
        WEBSITE_NAME,
        metadata["collection"],
        "photos",
        metadata["brand"],
        metadata["title"],
    )
    os.makedirs(dir, exist_ok=True)

    i = get_max_file_name(dir)
    for image_binary, image_ext in images:
        i += 1
        with open(os.path.join(dir, str(i) + image_ext), "wb") as f:
            f.write(image_binary)

    return dir


def parse_sneakers(url, collection, path="data"):
    r = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(r.text, "html.parser")
    metadata = get_sneakers_metadata(url, soup, collection)
    images = get_sneakers_images(soup)
    photos_path = save_sneakers_images_local(images, metadata, path)  # or S3
    metadata["photos_path"] = photos_path
    return metadata


def parse_sneakerbaas(path="data"):
    full_collection = get_collection_info()
    print(
        f"{WEBSITE_NAME} website: {full_collection['number_of_products']} sneakers found"
    )
    print(f"{len(COLLECTIONS)} collections found")

    full_metadata = []

    for collection in COLLECTIONS:
        metadata_collection = []
        collection_info = get_collection_info(collection)
        print(
            f"Parsing collection: {collection}, found {collection_info['number_of_pages']} pages, "
            f"{collection_info['number_of_products']} products, {collection_info['url']}"
        )
        for page in trange(1, collection_info["number_of_pages"] + 1):
            page_url = add_page_to_url(collection_info["url"], page)
            print(page_url)
            sneakers_urls = get_sneakers_urls(page_url)
            for sneakers_url in tqdm(sneakers_urls):
                metadata = parse_sneakers(sneakers_url, collection)
                metadata_collection.append(metadata)

        print(
            f"Collected {len(metadata_collection)} sneakers out of {collection_info['number_of_products']} in "
            f"{collection} collection"
        )
        df = pd.DataFrame(metadata_collection)
        df.to_csv(
            os.path.join(path, WEBSITE_NAME, collection, "metadata.csv"), index=False
        )

        full_metadata += metadata_collection

    df = pd.DataFrame(full_metadata)
    df.to_csv(os.path.join(path, WEBSITE_NAME, "metadata.csv"), index=False)
    print(
        f"Collected {len(full_metadata)} sneakers out of {full_collection['number_of_products']} in "
        f"{WEBSITE_NAME} website"
    )


if __name__ == "__main__":
    parse_sneakerbaas()
