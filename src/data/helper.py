from pathlib import Path
from urllib.parse import urlparse, urlsplit


def get_parent(path: str) -> str:
    return str(Path(path).parent)


def get_hostname_url(url: str) -> str:
    parsed_url = urlsplit(url)
    return f"{parsed_url.scheme}://{parsed_url.netloc}"


def remove_query(url: str) -> str:
    return urlparse(url)._replace(query="").geturl()


def remove_params(url: str) -> str:
    return urlparse(url)._replace(params="").geturl()


def add_https(url: str) -> str:
    return urlparse(url)._replace(scheme="https").geturl()


def get_image_extension(url: str) -> str:
    return Path(urlparse(url).path).suffix


def add_page(url: str, page_number: int) -> str:
    """
    Adds page number to "url" link as a query.
    Example page: 3, url: https://www.sneakerbaas.com/collections/sneakers
    Returns: https://www.sneakerbaas.com/collections/sneakers?page=3
    """
    return urlparse(url)._replace(query=f"page={page_number}").geturl()


def fix_string(path: str) -> str:
    """
    Removes ", ', \, / symbols, which cause errors on s3.
    """
    return fix_html_text(path.replace('"', "").replace("'", "").replace("/", "").replace("\\", "").lower())


def fix_html_text(text: str) -> str:
    """
    Strips, removes newline symbols and "\\xa0" symbol which frequently appear in html texts.
    """
    return text.replace("\xa0", " ").strip().replace("\n", " ")


def split_dir_filename_ext(path):
    path_obj = Path(path)
    directory = path_obj.parent
    filename = path_obj.stem
    file_extension = path_obj.suffix
    return str(directory), str(filename), str(file_extension)

# def form_s3_url(path: str) -> str:
#     """
#     Returns s3 path in "s3://bucket/path" format.
#     """
#     s3_url = urlparse("")._replace(scheme="s3", netloc=BUCKET, path=get_parent(path))
#     return s3_url.geturl()
