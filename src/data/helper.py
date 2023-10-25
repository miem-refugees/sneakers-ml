from fake_useragent import UserAgent
from urllib.parse import urlparse, urlsplit
from pathlib import Path
import boto3
import pandas as pd

HEADERS = {"User-Agent": UserAgent().random}
BUCKET = "sneakers-ml"
YANDEX_URL = "https://storage.yandexcloud.net"


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


def get_max_file_name(dir: str, s3: bool = False) -> int:
    """
    Returns the max integer file number in the "dir" folder.
    Example files: 1.png, 2.png, 3.png. Returned value: 3.
    Returns -1 if the folder is empty.
    """
    filenames = get_filenames(dir, s3)

    if filenames:
        without_ext = [int(Path(fn).stem) for fn in filenames]
        return max(without_ext)
    else:
        return -1


def add_page(url: str, page_number: int) -> str:
    """
    Adds page number to "url" link as a query.
    Example page: 3, url: https://www.sneakerbaas.com/collections/sneakers
    Returns: https://www.sneakerbaas.com/collections/sneakers?page=3
    """
    return urlparse(url)._replace(query=f"page={page_number}").geturl()


def fix_path_for_s3(path: str) -> str:
    """
    Removes ", ', \, / symbols, which cause errors on s3.
    """
    return path.replace('"', "").replace("'", "").replace("/", "").replace("\\", "")


def get_filenames(dir: str, s3: bool = False) -> list[str]:
    """
    Returns all filenames in a "dir" folder, either local or s3.
    """
    if s3:
        s3_res = boto3.resource(service_name="s3", endpoint_url=YANDEX_URL)
        s3_bucket = s3_res.Bucket(BUCKET)
        s3_objects = s3_bucket.objects.filter(Prefix=dir).all()
        return [Path(f.key).name for f in s3_objects]
    else:
        return [str(file) for file in Path(dir).iterdir() if file.is_file()]


def image_exists(dir: str, image_binary: bytes) -> bool:
    """
    Checks if the "image_binary" image is already present in the "dir" folder.
    """
    images = get_filenames(dir)

    for image in images:
        if open(image, "rb").read() == image_binary:
            return True

    return False


def form_s3_url(path: str) -> str:
    """
    Returns s3 path in "s3://bucket/path" format.
    """
    s3_url = urlparse("")._replace(scheme="s3", netloc=BUCKET, path=get_parent(path))
    return s3_url.geturl()


def upload_local_s3(local_dir: str, s3_dir: str) -> str:
    """
    Uploads file from "local_dir" dir to "s3_dir" dir and returns
    s3 dir in "s3://bucket/s3_dir" format
    """
    s3 = boto3.resource(service_name="s3", endpoint_url=YANDEX_URL)
    s3.Bucket(BUCKET).upload_file(local_dir, s3_dir)
    return form_s3_url(s3_dir)


def fix_html_text(text: str) -> str:
    """
    Strips, removes newline symbols and "\\xa0" symbol which frequently appear in html texts.
    """
    return text.replace("\xa0", " ").strip().replace("\n", " ")


def save_images(
    images: tuple[bytes, str], dir: str, s3: bool = False
) -> tuple[str, str]:
    """
    Save images to dir, upload to s3 if required.
    """
    Path(dir).mkdir(parents=True, exist_ok=True)
    s3_dir = ""

    current_max_file_name = get_max_file_name(dir)

    for image_binary, image_ext in images:
        if not image_exists(dir, image_binary):
            current_max_file_name += 1
            image_dir = str(Path(dir, str(current_max_file_name) + image_ext))
            with open(image_dir, "wb") as f:
                f.write(image_binary)

            if s3:
                s3_dir = upload_local_s3(image_dir, image_dir)

    return dir, s3_dir


def save_metadata(
    metadata: dict[str, str], path: str, index_column: str, s3: bool = False
) -> None:
    """
    Saves metadata dict in .csv format in path. If .csv already exists, concats the data and
    removes duplicates by index_column. Uploads to s3 if required.
    """
    df = pd.DataFrame(metadata)

    if Path(path).is_file():
        old_df = pd.read_csv(path)
        df = pd.concat([old_df, df])
        df = df.drop_duplicates(subset=index_column, keep="first").reset_index(drop=True)

    df.to_csv(path, index=False)

    if s3:
        upload_local_s3(path, path)
