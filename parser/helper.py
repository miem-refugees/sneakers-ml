from fake_useragent import UserAgent
from urllib.parse import urlparse, urlunparse, urlsplit


def get_home_url(url):
    parsed_url = urlsplit(url)
    home_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    return home_url


def remove_query_from_url(url):
    parsed_url = urlparse(url)
    parsed_url = parsed_url._replace(query="")
    modified_url = urlunparse(parsed_url)
    return modified_url


def add_https_to_url(url):
    if not url.startswith("http://") and not url.startswith("https://"):
        corrected_url = "https://" + url
    else:
        corrected_url = url
    return corrected_url


def get_image_extension(url):
    parsed_url = urlparse(add_https_to_url(url))
    return "." + parsed_url.path.split("/")[-1].split(".")[1]
