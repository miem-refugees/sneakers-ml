import re
from itertools import permutations
from pathlib import Path
from string import ascii_letters, digits
from typing import Union

import numpy as np
import pandas as pd
from loguru import logger
from pandarallel import pandarallel
from tqdm.auto import tqdm

from sneakers_ml.data.image import get_images_count, get_images_formats, get_images_suffixes
from sneakers_ml.data.local import LocalStorage
from sneakers_ml.data.storage import StorageProcessor

pandarallel.initialize(progress_bar=False)
tqdm.pandas()


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

    def merge_images(
        self,
        main_dataset: pd.DataFrame,
        path: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
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


class ColumnPreprocessor:
    COLOR_WORDS_PATH = "data/merged/metadata/other/color_words.txt"
    ALLOWED_SYMBOLS = ascii_letters + digits + " "

    DEFAULT_REPLACEMENTS = {
        "ê": "e",
        "ä": "a",
        "é": "e",
        "ç": "c",
        "ô": "o",
        "ü": "u",
        "&amp;": "&",
        "β": "beta",
        "ß": "beta",
        "–": "-",
        "‘": "'",
        "’": "'",
        "”": '"',
        "“": '"',
        "\\'\\'": '"',
        '""': '"',
        "''": "'",
    }
    WHITESPACE_REPLACEMENTS = dict.fromkeys({"-", "_", "\\", "/", "&"}, " ")

    ADDITIONAL_BRANDS = {
        "andersson",
        "wales bonner",
        "crosc",
        "ader",
        "ami",
        "andre saraiva",
        "april skateboards",
        "asphaltgold",
        "auralee",
        "awake",
        "beams",
        "bianca chandon",
        "billie eilish",
    }

    BRANDS_MAPPING = {
        "vans vault": "vans",
        "saucony originals": "saucony",
        "salomon advanced": "salomon",
        "reebok classics": "reebok",
        "puma sportstyle": "puma",
        "nike skateboarding": "nike",
        "clarks originals": "clarks",
        "adidas performance": "adidas",
        "adidas originals": "adidas",
    }

    @classmethod
    def get_all_brands(cls, datasets: dict[str, pd.DataFrame]) -> list[str]:
        brands_raw: set[str] = set()
        for dataset in datasets.values():
            brands_raw = brands_raw.union(dataset["brand"].to_list())

        brands_raw = brands_raw.union(cls.ADDITIONAL_BRANDS)

        brands_solo = {cls.preprocess_text(text) for text in brands_raw}

        collabs_x = {f"{pair[0]} x {pair[1]}" for pair in permutations(brands_solo, 2)}
        collabs_whitespace = {f"{pair[0]} {pair[1]}" for pair in permutations(brands_solo, 2)}

        brands = list(set.union(collabs_x, collabs_whitespace, brands_solo))
        brands.sort(key=lambda x: (len(x), x), reverse=True)
        return brands

    @staticmethod
    def split_colors(text: str) -> list[str]:
        color_replacements = {"&": "/", " / ": "/", "/ ": "/", " /": "/"}
        for key, value in color_replacements.items():
            text = text.replace(key, value)
        colors = text.lower().split("/")
        return colors

    @classmethod
    def split_title_and_color(cls, title: str) -> tuple[str, str]:
        title = cls.apply_replacements(title, cls.DEFAULT_REPLACEMENTS)
        for quote in ["'", '"']:
            end_quote = title.rfind(quote)
            second_last_quote = title.rfind(quote, 0, end_quote - 1) if end_quote != -1 else -1

            if second_last_quote != -1:
                title_part = title[:second_last_quote].strip()
                color_part = title[second_last_quote + 1 : end_quote].strip()
                return title_part, color_part

        return title, ""

    @staticmethod
    def apply_replacements(text: str, replacements: dict[str, str]) -> str:
        for key, value in replacements.items():
            text = text.replace(key.lower(), value.lower())
            text = text.replace(key.upper(), value.upper())
        return text

    @classmethod
    def preprocess_text(cls, text: str) -> str:
        text = text.lower()

        text = cls.apply_replacements(text, cls.DEFAULT_REPLACEMENTS)
        text = cls.apply_replacements(text, cls.WHITESPACE_REPLACEMENTS)
        text = cls.remove_extra_symbols(text)
        text = cls.remove_extra_whitespaces(text)

        return text.strip()

    @classmethod
    def preprocess_title(cls, title: str, brands: list[str], colors: list) -> str:
        title = cls.preprocess_text(title)

        for brand in brands:
            title = title.removeprefix(brand)

        for color in colors:
            title = title.removesuffix(color)

        title = title.removeprefix(" x ")  # some collabs are only half-resolved

        return title.strip()

    @classmethod
    def remove_extra_symbols(cls, input_string: str) -> str:
        return "".join(char for char in input_string if char in cls.ALLOWED_SYMBOLS).strip()

    @staticmethod
    def remove_extra_whitespaces(text: str) -> str:
        return " ".join(text.split())

    @classmethod
    def get_extra_symbols(cls, df: pd.DataFrame, column: str = "title") -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for text in df[column].tolist():
            for symbol in text:
                if not any(
                    symbol.lower() in d
                    for d in (
                        cls.ALLOWED_SYMBOLS,
                        cls.DEFAULT_REPLACEMENTS,
                        cls.WHITESPACE_REPLACEMENTS,
                    )
                ):
                    if symbol not in out:
                        out[symbol] = []
                    out[symbol].append(text)

        return out

    @classmethod
    def check_extra_symbols(
        cls,
        datasets: dict[str, pd.DataFrame],
        columns: tuple[str],
    ) -> dict[str, set[str]]:
        extra_symbols: dict[str, set[str]] = {dataset_name: set() for dataset_name in datasets}
        for dataset_name, dataset in datasets.items():
            for column in columns:
                extra_symbols[dataset_name] |= set(cls.get_extra_symbols(dataset, column).keys())

        return extra_symbols

    @staticmethod
    def flatten_list(images_list: list) -> list[str]:
        return np.unique([item for item in np.hstack(images_list) if not pd.isna(item)]).tolist()

    @staticmethod
    def get_colors(path: str) -> list[str]:
        with Path(path).open("r") as file:
            colors = list({word.strip().lower() for word in file.readlines()})
        colors.sort(key=lambda x: (len(x), x), reverse=True)
        with Path(path).open("w") as file:
            file.write("\n".join(colors))
        return colors

    @staticmethod
    def merge_images_columns(columns: dict, images_columns: list[str]) -> list[str]:
        return [columns[images_column] for images_column in images_columns]

    @staticmethod
    def sneakerbaas_get_color(input_string: str) -> Union[list[str], float]:
        if pd.notna(input_string):
            match = re.search(
                r"(Colour|Colours|Colors|Color|Kleur): (.*?)(?:-|$)",
                input_string,
                re.IGNORECASE,
            )
            if match:
                colors = match.group(2).strip()
                colors = colors.replace("/", " ").lower().split()
                return list(dict.fromkeys(colors))
        return np.nan


if __name__ == "__main__":
    Merger("data/raw/metadata").merge("data/merged")
