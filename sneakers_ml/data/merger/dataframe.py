import pandas as pd
from loguru import logger
from pandarallel import pandarallel

from sneakers_ml.data.merger.column import ColumnPreprocessor

pandarallel.initialize(progress_bar=False)


class DataFramePreprocessor:
    def __init__(self, datasets: [str, pd.DataFrame]) -> None:
        self.datasets = {name: dataset.copy() for name, dataset in datasets.items()}
        self.brands = ColumnPreprocessor.get_all_brands(datasets)
        logger.info(f"Generated {len(self.brands)} brands")
        self.colors = ColumnPreprocessor.get_colors(ColumnPreprocessor.COLOR_WORDS_PATH)
        logger.info(f"Read {len(self.colors)} colors")

        for name in self.datasets:
            self.datasets[name]["website"] = name
            self.datasets[name].columns = self.datasets[name].columns.str.lower()

    def _preprocess_superkicks(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = dataframe.drop(
            [
                "product-dimensions",
                "collection_url",
                "generic-name",
                "weight",
                "imported-by",
                "manufacturer",
                "unit-of-measurement",
                "marketed-by",
                "article-code",
                "country-of-origin",
                "slug",
                "brand_slug",
            ],
            axis=1,
        )
        dataframe = dataframe.drop_duplicates(subset=["title", "collection_name", "url"])

        dataframe["pricecurrency"] = "INR"
        dataframe["price"] = dataframe["price"].apply(lambda x: float(x.replace("₹", "").replace(",", "")))

        dataframe[["title_without_color", "color"]] = dataframe["title"].apply(
            lambda x: pd.Series(ColumnPreprocessor.split_title_and_color(x))
        )
        dataframe = self._create_merge_columns(dataframe)

        dataframe = dataframe.drop(["description"], axis=1)

        logger.info(f"{dataframe['website'][0]} columns: {dataframe.columns.to_numpy()}")
        logger.info(f"{dataframe['website'][0]} shape: {dataframe.shape}")

        return dataframe

    def _preprocess_sneakerbaas(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = dataframe.drop(["collection_url", "slug", "brand_slug"], axis=1)
        dataframe = dataframe.drop_duplicates(subset=["title", "collection_name", "url"])

        dataframe["description_color"] = dataframe["description"].apply(ColumnPreprocessor.sneakerbaas_get_color)

        dataframe[["title_without_color", "color"]] = dataframe["title"].apply(
            lambda x: pd.Series(ColumnPreprocessor.split_title_and_color(x))
        )
        dataframe = self._create_merge_columns(dataframe)

        dataframe = dataframe.drop(["description", "description_color"], axis=1)

        logger.info(f"{dataframe['website'][0]} columns: {dataframe.columns.to_numpy()}")
        logger.info(f"{dataframe['website'][0]} shape: {dataframe.shape}")

        return dataframe

    def _preprocess_footshop(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = dataframe.drop(["collection_url", "slug", "brand_slug"], axis=1)
        dataframe = dataframe.drop_duplicates(subset=["title", "collection_name", "url"])

        dataframe["price"] = dataframe["price"].apply(lambda x: float(x.replace("€", "").replace("$", "")))

        dataframe["title_without_color"] = dataframe["title"].apply(
            lambda x: ColumnPreprocessor.apply_replacements(x, ColumnPreprocessor.DEFAULT_REPLACEMENTS)
        )

        dataframe["color"] = dataframe["color"].apply(ColumnPreprocessor.split_colors)

        dataframe = self._create_merge_columns(dataframe)

        logger.info(f"{dataframe['website'][0]} columns: {dataframe.columns.to_numpy()}")
        logger.info(f"{dataframe['website'][0]} shape: {dataframe.shape}")

        return dataframe

    def _preprocess_kickscrew(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        images_columns = ["right-side-img", "left-side-img", "front-both-img"]

        dataframe["images_path"] = dataframe.apply(
            lambda x: ColumnPreprocessor.merge_images_columns(x, images_columns), axis=1
        )
        dataframe = dataframe.drop([*images_columns, "slug"], axis=1)

        dataframe = dataframe.drop_duplicates(subset=["title", "brand", "url"])

        dataframe[["title_without_color", "color"]] = dataframe["title"].apply(
            lambda x: pd.Series(ColumnPreprocessor.split_title_and_color(x))
        )
        dataframe = self._create_merge_columns(dataframe)

        logger.info(f"{dataframe['website'][0]} columns: {dataframe.columns.to_numpy()}")
        logger.info(f"{dataframe['website'][0]} shape: {dataframe.shape}")

        return dataframe

    def _create_merge_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe["title_merge"] = dataframe["title_without_color"].parallel_apply(
            lambda x: ColumnPreprocessor.preprocess_title(x, self.brands, self.colors),
        )

        dataframe["brand_merge"] = dataframe["brand"].apply(ColumnPreprocessor.preprocess_text)
        return dataframe

    def get_preprocessed_datasets(self) -> dict[str, pd.DataFrame]:
        return {
            "superkicks": self._preprocess_superkicks(self.datasets["superkicks"]),
            "sneakerbaas": self._preprocess_sneakerbaas(self.datasets["sneakerbaas"]),
        }
