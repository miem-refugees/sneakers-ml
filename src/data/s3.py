import logging
import tempfile
import boto3

from src.data.base import AbstractStorage


class S3Storage(AbstractStorage):
    def __init__(self, bucket_name: str, endpoint_url: str):
        self.s3 = boto3.resource(service_name="s3", endpoint_url=endpoint_url)
        self.s3_client = boto3.client('s3')
        self.bucket_name = bucket_name
        self.bucket = self.s3.Bucket(bucket_name)

        self.logger = logging.getLogger(self.__class__.__name__)

    def upload_file(self, local_path: str, s3_path: str):
        """
        Uploads file from "local_path" dir to "s3_path" dir and returns
        s3 dir in "s3://bucket/s3_dir" format
        """
        self.bucket.upload_file(local_path, s3_path)

    def download_file(self, s3_path: str):
        """
        Downloads file to temporary cache and returns data
        :param s3_path:
        :return:
        """
        with tempfile.TemporaryFile() as data:
            self.s3_client.download_fileobj(self.bucket_name, s3_path, data)
            return data
