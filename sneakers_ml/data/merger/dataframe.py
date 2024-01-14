import pandas as pd
from loguru import logger
from pandarallel import pandarallel

from sneakers_ml.data.merger.column import ColumnPreprocessor


class DataFramePreprocessor:
    def __init__(self, datasets: dict[str, pd.DataFrame]) -> None:
        pandarallel.initialize(progress_bar=False)

        self.datasets = {name: dataset.copy() for name, dataset in datasets.items()}
        for name in self.datasets:
            self.datasets[name]["website"] = name
            self.datasets[name].columns = self.datasets[name].columns.str.lower()

        self.brands = ColumnPreprocessor.get_all_brands(datasets)
        self.colors = ColumnPreprocessor.get_colors(ColumnPreprocessor.COLOR_WORDS_PATH)
        self.preprocessed = False

        logger.info(f"Read {len(self.colors)} colors")
        logger.info(f"Generated {len(self.brands)} brands")

    def preprocess_superkicks(self, superkicks_dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = superkicks_dataframe.copy()
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

    def preprocess_sneakerbaas(self, raw_dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = raw_dataframe.copy()
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

    def preprocess_footshop(self, raw_dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = raw_dataframe.copy()
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

    def preprocess_kickscrew(self, raw_dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = raw_dataframe.copy()
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
            lambda x: ColumnPreprocessor.preprocess_title(x, self.brands, self.colors)
        )

        dataframe["brand_merge"] = dataframe["brand"].apply(ColumnPreprocessor.preprocess_text)
        return dataframe

    def _log_extra_symbols(self) -> None:
        extra_symbols_columns = ("title", "brand")
        logger.info(f"Extra symbols columns {extra_symbols_columns}")
        extra_symbols = ColumnPreprocessor.check_extra_symbols(self.datasets, extra_symbols_columns)
        for dataset_name, symbols in extra_symbols.items():
            logger.info(f"{dataset_name}: {symbols}")

    def preprocess_datasets(self) -> None:
        self.datasets = {name: getattr(self, f"preprocess_{name}")(self.datasets[name]) for name in self.datasets}
        self._log_extra_symbols()
        self.preprocessed = True

    def get_preprocessed_datasets(self) -> dict[str, pd.DataFrame]:
        if not self.preprocessed:
            self.preprocess_datasets()
        return {name: dataset.copy() for name, dataset in self.datasets.items()}
