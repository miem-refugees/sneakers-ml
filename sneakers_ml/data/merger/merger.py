from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm.auto import tqdm

from sneakers_ml.data.image import get_images_count, get_images_formats, get_images_suffixes
from sneakers_ml.data.local import LocalStorage
from sneakers_ml.data.merger.column import ColumnPreprocessor
from sneakers_ml.data.merger.dataframe import DataFramePreprocessor
from sneakers_ml.data.storage import StorageProcessor

tqdm.pandas()


class Merger:
    def __init__(self, metadata_path: str) -> None:
        self.metadata_path = Path(metadata_path)
        discovered_datasets = list(self.metadata_path.iterdir())
        self.datasets = {Path(source).stem: pd.read_csv(source) for source in discovered_datasets}
        logger.info(f"Read {len(self.datasets)} datasets {list(self.datasets.keys())}")

        self.processor = StorageProcessor(LocalStorage())
        self.df_processor = DataFramePreprocessor(self.datasets)

    def get_datasets(self) -> dict[str, pd.DataFrame]:
        return self.datasets

    def get_preprocessed_datasets(self) -> dict[str, pd.DataFrame]:
        return self.df_processor.get_preprocessed_datasets()

    def get_full_merged_dataset(self, concatted_datasets: pd.DataFrame) -> pd.DataFrame:
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

        full_merged_dataset = (
            concatted_datasets.groupby("title_merge").agg(full_aggregations).reset_index().sort_values(by="title_merge")
        )

        full_merged_dataset["images_flattened"] = full_merged_dataset["images_path"].apply(
            ColumnPreprocessor.flatten_list
        )

        logger.info(f"Full merged dataset columns: {full_merged_dataset.columns.to_numpy()}")
        logger.info(f"Full merged dataset shape: {full_merged_dataset.shape}")
        return full_merged_dataset

    def get_main_merged_dataset(self, concatted_datasets: pd.DataFrame) -> pd.DataFrame:
        main_aggregations = {
            "brand_merge": lambda x: x.value_counts().index[0],
            "images_path": ColumnPreprocessor.flatten_list,
            "price": "first",
            "pricecurrency": "first",
            "color": ColumnPreprocessor.flatten_list,
            "website": list,
        }

        main_merged_dataset = (
            concatted_datasets.groupby("title_merge").agg(main_aggregations).reset_index().sort_values(by="title_merge")
        )

        main_merged_dataset["brand_merge"] = main_merged_dataset["brand_merge"].apply(
            lambda x: ColumnPreprocessor.BRANDS_MAPPING[x] if x in ColumnPreprocessor.BRANDS_MAPPING else x,
        )

        logger.info(f"Main dataset columns: {main_merged_dataset.columns.to_numpy()}")
        logger.info(f"Main dataset shape: {main_merged_dataset.shape}")

        return main_merged_dataset

    def get_merged_datasets(
        self, path: str, extra_symbols_columns: tuple = ("title", "brand")
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        formatted_datasets = self.get_preprocessed_datasets()

        logger.info(f"Extra symbols columns {extra_symbols_columns}")
        for dataset_name, symbols in ColumnPreprocessor.check_extra_symbols(
            formatted_datasets, extra_symbols_columns
        ).items():
            logger.info(f"{dataset_name}: {symbols}")

        concatted_datasets = pd.concat(list(formatted_datasets.values()), ignore_index=True)

        full_dataset = self.get_full_merged_dataset(concatted_datasets)
        main_dataset = self.get_main_merged_dataset(concatted_datasets)

        logger.info(f"{concatted_datasets.shape[0]} -> {full_dataset.shape[0]}")

        if path:
            save_path = Path(path)
            save_path.mkdir(parents=True, exist_ok=True)

            full_path = save_path / "full_dataset.csv"
            main_path = save_path / "main_dataset.csv"

            full_dataset.to_csv(full_path, index=False)
            main_dataset.to_csv(main_path, index=False)

        return main_dataset, full_dataset

    def _apply_images_merge(self, x: dict, path: str, merge_column_name: str, images_column_name: str) -> pd.Series:
        path = Path(path) / x[merge_column_name]
        return pd.Series([self.processor.images_to_directory(x[images_column_name], path), str(path)])

    def _get_merged_images_dataset(self, df: pd.DataFrame, merge_column_name: str, path: str) -> pd.DataFrame:
        dataframe = df.copy()
        dataframe[["unique_images_count", "images"]] = dataframe.progress_apply(
            lambda x: self._apply_images_merge(x, path, merge_column_name, "images_path"),
            axis=1,
        )
        dataframe = dataframe.drop("images_path", axis=1)

        images_count = get_images_count(path)
        images_suffixes = get_images_suffixes(path)
        images_formats = get_images_formats(path)

        logger.info(f"{merge_column_name} dataset columns: {dataframe.columns.to_numpy()}")
        logger.info(f"{merge_column_name} dataset shape: {dataframe.shape}")
        logger.info(f"{merge_column_name} dataset images count: {images_count}")
        logger.info(f"{merge_column_name} dataset images suffixes: {set(images_suffixes)}")
        logger.info(f"{merge_column_name} dataset images formats: {set(images_formats)}")

        return dataframe

    def merge_images(self, main_dataset: pd.DataFrame, path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        save_path = Path(path)

        main_dataset = main_dataset.drop(["price", "pricecurrency", "color", "website"], axis=1)

        brands_dataset = (
            main_dataset.groupby("brand_merge").agg({"images_path": ColumnPreprocessor.flatten_list}).reset_index()
        )
        brands_path = str(save_path / "images" / "by-brands")
        brands_dataset = self._get_merged_images_dataset(brands_dataset, "brand_merge", brands_path)

        models_dataset = main_dataset.copy()
        models_path = str(save_path / "images" / "by-models")
        models_dataset = self._get_merged_images_dataset(models_dataset, "title_merge", models_path)

        brands_dataset.to_csv(save_path / "metadata" / "brands_dataset.csv", index=False)
        models_dataset.to_csv(save_path / "metadata" / "models_dataset.csv", index=False)

        return brands_dataset, models_dataset

    def merge(self, path: str) -> dict[str, pd.DataFrame]:
        save_path = Path(path)
        main_dataset, full_dataset = self.get_merged_datasets(path=str(save_path / "metadata"))
        brands_dataset, models_dataset = self.merge_images(main_dataset, str(save_path))
        logger.info("ALL MERGED")
        return {
            "main_dataset": main_dataset,
            "full_dataset": full_dataset,
            "brands_dataset": brands_dataset,
            "models_dataset": models_dataset,
        }


if __name__ == "__main__":
    Merger("data/raw/metadata").merge("data/merged")
