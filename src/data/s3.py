import io
import boto3
from pathlib import Path

from src.data.base import AbstractStorage


class S3Storage(AbstractStorage):
    YANDEX_S3 = "https://storage.yandexcloud.net"
    SNEAKERS_BUCKET = "sneakers-ml"

    def __init__(
        self, bucket_name: str = SNEAKERS_BUCKET, endpoint_url: str = YANDEX_S3
    ):
        self.bucket_name = bucket_name
        self.s3: boto3.ServiceResource = boto3.resource(
            service_name="s3", endpoint_url=endpoint_url
        )
        self.bucket = self.s3.Bucket(bucket_name)

    def upload_binary(self, binary_data: bytes, s3_path: str) -> None:
        """
        Uploads Python binary to s3_path location
        """
        object = self.s3.Object(self.bucket_name, s3_path)
        object.put(Body=binary_data)

    def upload_file(self, local_path: str, s3_path: str) -> None:
        """
        Uploads file from "local_path" dir to "s3_path" dir and returns
        s3 dir in "s3://bucket/s3_dir" format
        """
        self.bucket.upload_file(local_path, s3_path)

    def download_file(self, s3_path: str) -> bytes:
        """
        Downloads file by path
        """
        binary_io = io.BytesIO()
        self.bucket.download_fileobj(s3_path, binary_io)
        return binary_io.getvalue()

    def delete_file(self, s3_path: str) -> None:
        """
        Deletes file. Do not raise or return error if nothing deleted
        """
        self.s3.Object(self.bucket_name, s3_path).delete()

    def get_all_files(self, path_prefix: str) -> list[str]:
        """
        Returns all filenames in path
        """
        s3_objects = self.bucket.objects.filter(Prefix=path_prefix).all()
        return [Path(f.key).name for f in s3_objects]
