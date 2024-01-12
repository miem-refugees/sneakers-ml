from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm.auto import tqdm

from sneakers_ml.data.merger.column import ColumnPreprocessor
from sneakers_ml.data.merger.dataframe import DataFramePreprocessor
from sneakers_ml.data.storage.image import get_images_count, get_images_formats, get_images_suffixes
from sneakers_ml.data.storage.local import LocalStorage
from sneakers_ml.data.storage.storage import StorageProcessor


class Merger:
    def __init__(self, metadata_path: str, ignore: Sequence[str]) -> None:
        tqdm.pandas()

        self.metadata_path = Path(metadata_path)
        discovered_datasets = list(self.metadata_path.iterdir())

        self.datasets = {Path(source).stem: pd.read_csv(source) for source in discovered_datasets}

        for ignored_dataset in ignore:
            self.datasets.pop(ignored_dataset)

        self.processor = StorageProcessor(LocalStorage())
        self.df_processor = DataFramePreprocessor(self.datasets)

        logger.info(f"Read {len(self.datasets)} datasets {list(self.datasets.keys())}")

    def get_datasets(self) -> dict[str, pd.DataFrame]:
        return self.datasets

    def get_preprocessed_datasets(self) -> dict[str, pd.DataFrame]:
        return self.df_processor.get_preprocessed_datasets()

    def get_full_dataframe(self) -> pd.DataFrame:
        formatted_datasets = self.get_preprocessed_datasets()
        concatted_datasets = pd.concat(list(formatted_datasets.values()), ignore_index=True)

        full_aggregations = {
            "brand_merge": list,
            "images_path": list,
            "title": list,
            "title_without_color": list,
            "brand": list,
            "collection_name": list,
            "color": list,
            "price": list,
            "pricecurrency": list,
            "url": list,
            "website": list,
        }

        full_dataframe = (
            concatted_datasets.groupby("title_merge").agg(full_aggregations).reset_index().sort_values(by="title_merge")
        )

        full_dataframe["images_flattened"] = full_dataframe["images_path"].apply(ColumnPreprocessor.flatten_list)

        logger.info(f"Full dataframe columns: {full_dataframe.columns.to_numpy()}")
        logger.info(f"Full dataframe shape: {full_dataframe.shape}")
        logger.info(f"{concatted_datasets.shape[0]} -> {full_dataframe.shape[0]}")

        return full_dataframe

    def get_main_dataframe(self) -> pd.DataFrame:
        formatted_datasets = self.get_preprocessed_datasets()
        concatted_datasets = pd.concat(list(formatted_datasets.values()), ignore_index=True)

        main_aggregations = {"brand_merge": lambda x: x.mode()[0], "images_path": ColumnPreprocessor.flatten_list}

        main_dataframe = (
            concatted_datasets.groupby("title_merge").agg(main_aggregations).reset_index().sort_values(by="title_merge")
        )

        main_dataframe["brand_merge"] = main_dataframe["brand_merge"].apply(
            lambda x: ColumnPreprocessor.BRANDS_MAPPING[x] if x in ColumnPreprocessor.BRANDS_MAPPING else x
        )

        logger.info(f"Main dataframe columns: {main_dataframe.columns.to_numpy()}")
        logger.info(f"Main dataframe shape: {main_dataframe.shape}")
        logger.info(f"{concatted_datasets.shape[0]} -> {main_dataframe.shape[0]}")

        return main_dataframe

    def _copy_images(self, row: pd.Series, images_column: str, merge_column: str, path: Path) -> pd.Series:
        images = row[images_column]
        path = path / row[merge_column]
        return pd.Series([self.processor.images_to_directory(images, str(path)), str(path)])  # type: ignore[list-item]

    def _merge_images(self, dataframe: pd.DataFrame, merge_column: str, path: Path) -> pd.DataFrame:
        dataframe[["unique_images_count", "images"]] = dataframe.progress_apply(
            lambda x: self._copy_images(x, "images_path", merge_column, path), axis=1
        )  # type: ignore[operator]
        dataframe = dataframe.drop("images_path", axis=1)

        images_count = get_images_count(str(path))
        images_suffixes = get_images_suffixes(str(path))
        images_formats = get_images_formats(str(path))

        logger.info(f"{merge_column} dataset columns: {dataframe.columns.to_numpy()}")
        logger.info(f"{merge_column} dataset shape: {dataframe.shape}")
        logger.info(f"{merge_column} dataset images count: {images_count}")
        logger.info(f"{merge_column} dataset images suffixes: {set(images_suffixes)}")
        logger.info(f"{merge_column} dataset images formats: {set(images_formats)}")

        return dataframe

    def get_brand_classification_dataset(self, save_path: str) -> pd.DataFrame:
        path = Path(save_path)
        main_dataframe = self.get_main_dataframe()

        brands_dataset = (
            main_dataframe.groupby("brand_merge").agg({"images_path": ColumnPreprocessor.flatten_list}).reset_index()
        )
        brands_path = path / "images" / "by-brands"
        brands_dataset = self._merge_images(brands_dataset, "brand_merge", brands_path)
        brands_dataset.to_csv(path / "metadata" / "brands_dataset.csv", index=False)

        return brands_dataset

    def get_model_classification_dataset(self, save_path: str) -> pd.DataFrame:
        path = Path(save_path)
        main_dataframe = self.get_main_dataframe()

        models_path = path / "images" / "by-models"
        models_dataset = self._merge_images(main_dataframe, "title_merge", models_path)
        models_dataset.to_csv(path / "metadata" / "models_dataset.csv", index=False)

        return models_dataset

    def save_all(self, save_path: str) -> dict[str, pd.DataFrame]:
        path = Path(save_path)
        path.mkdir(parents=True, exist_ok=True)

        main_dataframe = self.get_main_dataframe()
        main_dataframe.to_csv(path / "metadata" / "main_dataset.csv", index=False)

        full_dataframe = self.get_full_dataframe()
        full_dataframe.to_csv(path / "metadata" / "full_dataset.csv", index=False)

        brands_classification_dataset = self.get_brand_classification_dataset(save_path)
        model_classification_dataset = self.get_model_classification_dataset(save_path)

        logger.info("SUCCESS")

        return {
            "main_dataset": main_dataframe,
            "full_dataset": full_dataframe,
            "brands_dataset": brands_classification_dataset,
            "models_dataset": model_classification_dataset,
        }


if __name__ == "__main__":
    Merger("data/raw/metadata", ignore=("kickscrew", "highsnobiety", "footshop")).save_all("data/merged")
