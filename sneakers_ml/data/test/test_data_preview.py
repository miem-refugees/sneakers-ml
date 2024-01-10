import pandas as pd

from sneakers_ml.data.data_preview import ImagePreview
from sneakers_ml.data.test.conftest import DummyStorage


class TestImagePreview:
    def setup(self):
        self.storage = DummyStorage()
        self.img_preview = ImagePreview(self.storage)

    def test_preview(self):
        img_path_col = "img_path"
        sample = {"name": "New Balance 547", img_path_col: "src/data/test/static/newbalance574.jpg"}
        self.storage.upload_file(sample[img_path_col], sample[img_path_col])

        frame = pd.DataFrame([sample, sample])  # to check consistence for all rows

        result = self.img_preview.preview(frame, img_path_col)
        assert result is not None
        print(result)
