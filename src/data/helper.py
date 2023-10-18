from fake_useragent import UserAgent
from urllib.parse import urlparse, urlsplit
from pathlib import Path

HEADERS = {"User-Agent": UserAgent().random}


def get_hostname_url(url):
    parsed_url = urlsplit(url)
    return f"{parsed_url.scheme}://{parsed_url.netloc}"


def remove_query_from_url(url):
    return urlparse(url)._replace(query="").geturl()


def add_https_to_url(url):
    return urlparse(url)._replace(scheme="https").geturl()


def get_image_extension(url):
    return "." + urlparse(url).path.split("/")[-1].split(".")[1]


def get_max_file_name(folder):
    path_iter = Path(folder).iterdir()
    if any(path_iter):
        return int(
            max(
                (Path(fn).stem for fn in Path(folder).iterdir()),
                key=lambda fn: int(Path(fn).stem),
            )
        )
    else:
        return -1


def add_page_to_url(url, page_number):
    return urlparse(url)._replace(query=f"page={page_number}").geturl()
