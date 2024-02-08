from io import BytesIO

import boto3
import numpy as np
from PIL import Image


class S3ImageUtility:
    def __init__(self, bucket, region_name="ru-central-1"):
        self.bucket = bucket
        self.region_name = region_name

    def read_image_from_s3(self, name: str):
        s3 = boto3.resource("s3", region_name=self.region_name)
        bucket = s3.Bucket(self.bucket)
        object = bucket.Object(name)
        response = object.get()
        file_stream = response["Body"]
        im = Image.open(file_stream)
        return np.array(im)

    def write_image_to_s3(self, image: Image, name: str):
        s3 = boto3.resource("s3", self.region_name)
        bucket = s3.Bucket(self.bucket)
        object = bucket.Object(name)
        file_stream = BytesIO()
        image.save(file_stream, format="jpeg")
        object.put(Body=file_stream.getvalue())
