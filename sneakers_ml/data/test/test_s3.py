import uuid
from tempfile import NamedTemporaryFile

import pytest

from sneakers_ml.data.storage.base import AbstractStorage
from sneakers_ml.data.storage.s3 import S3Storage


@pytest.mark.skip("need to fix this test")
class TestS3:
    def setup(self):
        self.storage: AbstractStorage = S3Storage()
        self.file = NamedTemporaryFile()
        self.secret_code = uuid.uuid4().bytes
        with open(self.file.name, "wb") as f:
            f.write(self.secret_code)

        self.file_name = self.file.name.split("/")[-1]
        self.s3_path = f"tmp/{self.file_name}"

    def teardown(self):
        self.file.close()
        self.storage.delete_file(self.s3_path)

        assert len(self.storage.get_all_filenames(self.s3_path)) == 0

    def test_s3_flow(self):
        # upload
        self.storage.upload_file(self.file.name, self.s3_path)

        # check exist
        all_files = self.storage.get_all_filenames(self.s3_path)
        assert self.file_name in all_files

        # check download
        actual = self.storage.download_file(self.s3_path)
        assert actual == self.secret_code
