import os
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

from src.data.image import get_images_count, get_images_formats, get_images_suffixes
from src.data.local import LocalStorage
from src.data.storage import StorageProcessor

pandarallel.initialize(progress_bar=False)
tqdm.pandas()


class Merger:
    COLOR_WORDS_PATH = "data/merged/metadata/other/color_words.txt"

    ALLOWED_SYMBOLS = ascii_letters + digits + " "

    DEFAULT_REPLACEMENTS = {"ê": "e", "ä": "a", "é": "e", "ç": "c", "ô": "o", "ü": "u", "&amp;": "&", "β": "beta",
                            "ß": "beta", "–": "-", '‘': "'", '’': "'", '”': '"', '“': '"', "\\'\\'": '"', '""': '"',
                            "''": "'"}
    WHITESPACE_REPLACEMENTS = dict.fromkeys({"-", "_", "\\", "/", "&"}, " ")

    ADDITIONAL_BRANDS = {"andersson", "wales bonner", "crosc", "beams", "ader", "ami", "andre saraiva",
                         "april skateboards", "asphaltgold", "auralee", "awake", "beams", "bianca chandon",
                         "billie eilish"}

    BRANDS_MAPPING = {"vans vault": "vans", "saucony originals": "saucony", "salomon advanced": "salomon",
                      "reebok classics": "reebok", "puma sportstyle": "puma", "nike skateboarding": "nike",
                      "clarks originals": "clarks", "adidas performance": "adidas", "adidas originals": "adidas"}

    def __init__(self, metadata_path=Path("data", "raw", "metadata")) -> None:
        discovered_datasets = os.listdir(metadata_path)
        self.datasets = {Path(source).stem: pd.read_csv(Path(metadata_path, source)) for source in discovered_datasets}
        logger.info(f"Read {len(self.datasets)} datasets {list(self.datasets.keys())}")

        for name in self.datasets:
            self.datasets[name]["website"] = name
            self.datasets[name].columns = [x.lower() for x in self.datasets[name].columns]

        self.brands = self.get_all_brands()
        logger.info(f"Generated {len(self.brands)} brands")
        self.colors = self.get_colors(self.COLOR_WORDS_PATH)
        logger.info(f"Read {len(self.colors)} colors")

        self.processor = StorageProcessor(LocalStorage())

    def get_all_brands(self) -> list[str]:
        brands_raw = []
        for name, dataset in self.datasets.items():
            brands_raw += dataset["brand"].to_list()

        brands_raw = set(brands_raw)
        brands_raw = set.union(brands_raw, self.ADDITIONAL_BRANDS)

        brands_solo = {self.preprocess_text(text) for text in brands_raw}

        collabs_x = {f"{pair[0]} x {pair[1]}" for pair in permutations(brands_solo, 2)}
        collabs_whitespace = {f"{pair[0]} {pair[1]}" for pair in permutations(brands_solo, 2)}

        brands = list(set.union(collabs_x, collabs_whitespace, brands_solo))
        brands.sort(key=lambda x: (len(x), x), reverse=True)
        return brands

    def get_datasets(self) -> dict[str, pd.DataFrame]:
        return {"superkicks": self.datasets["superkicks"], "sneakerbaas": self.datasets["sneakerbaas"],
                "footshop": self.datasets["footshop"], "kickscrew": self.datasets["kickscrew"]}

    def get_preprocessed_datasets(self) -> dict[str, pd.DataFrame]:
        return {"superkicks": self.preprocess_superkicks(self.datasets["superkicks"]),
                "sneakerbaas": self.preprocess_sneakerbaas(self.datasets["sneakerbaas"])}

        # "footshop": self.preprocess_footshop(self.datasets["footshop"]),  # "kickscrew": self.preprocess_kickscrew(self.datasets["kickscrew"])}

    def preprocess_superkicks(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(["product-dimensions", "collection_url", "generic-name", "weight", "imported-by", "manufacturer",
                      "unit-of-measurement", "marketed-by", "article-code", "country-of-origin", "slug", "brand_slug"],
                     axis=1)
        df = df.drop_duplicates(subset=["title", "collection_name", "url"])

        df["pricecurrency"] = "INR"
        df["price"] = df["price"].apply(lambda x: float(x.replace("₹", "").replace(",", "")))

        df[["title_without_color", "color"]] = df["title"].apply(lambda x: pd.Series(self.split_title_and_color(x)))
        df = self.create_merge_columns(df)

        df = df.drop(["description"], axis=1)

        logger.info(f"{df['website'][0]} columns: {df.columns.values}")
        logger.info(f"{df['website'][0]} shape: {df.shape}")

        return df

    def preprocess_sneakerbaas(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(["collection_url", "slug", "brand_slug"], axis=1)
        df = df.drop_duplicates(subset=["title", "collection_name", "url"])

        def get_color(input_string: str) -> Union[list[str], None]:
            if pd.notnull(input_string):
                match = re.search(r'(Colour|Colours|Colors|Color|Kleur): (.*?)(?:-|$)', input_string, re.IGNORECASE)
                if match:
                    colors = match.group(2).strip()
                    colors = colors.replace("/", " ").lower().split()
                    return list(dict.fromkeys(colors))

        df["description_color"] = df["description"].apply(get_color)

        df[["title_without_color", "color"]] = df["title"].apply(lambda x: pd.Series(self.split_title_and_color(x)))
        df = self.create_merge_columns(df)

        df = df.drop(["description", "description_color"], axis=1)

        logger.info(f"{df['website'][0]} columns: {df.columns.values}")
        logger.info(f"{df['website'][0]} shape: {df.shape}")

        return df

    def preprocess_footshop(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(["collection_url", "slug", "brand_slug"], axis=1)
        df = df.drop_duplicates(subset=["title", "collection_name", "url"])

        df["price"] = df["price"].apply(lambda x: float(x.replace("€", "").replace("$", "")))

        df["title_without_color"] = df["title"].apply(lambda x: self.apply_replacements(x, self.DEFAULT_REPLACEMENTS))

        df["color"] = df["color"].apply(self.split_colors)

        df = self.create_merge_columns(df)

        logger.info(f"{df['website'][0]} columns: {df.columns.values}")
        logger.info(f"{df['website'][0]} shape: {df.shape}")

        return df

    def preprocess_kickscrew(self, df: pd.DataFrame) -> pd.DataFrame:
        images_columns = ["right-side-img", "left-side-img", "front-both-img"]

        df["images_path"] = df.apply(lambda x: self.merge_images_columns(x, images_columns), axis=1)
        df = df.drop(images_columns + ["slug"], axis=1)

        df = df.drop_duplicates(subset=["title", "brand", "url"])

        df[["title_without_color", "color"]] = df["title"].apply(lambda x: pd.Series(self.split_title_and_color(x)))
        df = self.create_merge_columns(df)

        logger.info(f"{df['website'][0]} columns: {df.columns.values}")
        logger.info(f"{df['website'][0]} shape: {df.shape}")

        return df

    def create_merge_columns(self, df):
        df["title_merge"] = df.parallel_apply(
            lambda x: self.preprocess_title(x["title_without_color"], self.brands, self.colors), axis=1)

        df["brand_merge"] = df["brand"].apply(self.preprocess_text)
        return df

    def get_full_merged_dataset(self, concatted_datasets: pd.DataFrame) -> pd.DataFrame:
        full_aggregations = {"brand_merge": list, "images_path": list, "title": list, "title_without_color": list,
                             "brand": list, "collection_name": list, "color": list, "price": list,
                             "pricecurrency": list, "url": list, "website": list}

        full_merged_dataset = concatted_datasets.groupby("title_merge").agg(
            full_aggregations).reset_index().sort_values(by="title_merge")

        full_merged_dataset["images_flattened"] = full_merged_dataset["images_path"].apply(self.flatten_images_list)

        logger.info(f"Full merged dataset columns: {full_merged_dataset.columns.values}")
        logger.info(f"Full merged dataset shape: {full_merged_dataset.shape}")
        return full_merged_dataset

    def get_main_merged_dataset(self, concatted_datasets: pd.DataFrame) -> pd.DataFrame:

        main_aggregations = {"brand_merge": lambda x: x.value_counts().index[0],
                             "images_path": self.flatten_images_list, "price": "first", "pricecurrency": "first",
                             "color": self.flatten_images_list, "website": list}

        main_merged_dataset = concatted_datasets.groupby("title_merge").agg(
            main_aggregations).reset_index().sort_values(by="title_merge")

        main_merged_dataset["brand_merge"] = main_merged_dataset["brand_merge"].apply(
            lambda x: self.BRANDS_MAPPING[x] if x in self.BRANDS_MAPPING else x)

        logger.info(f"Main dataset columns: {main_merged_dataset.columns.values}")
        logger.info(f"Main dataset shape: {main_merged_dataset.shape}")

        return main_merged_dataset

    def get_merged_datasets(self, save_path: Union[str, Path] = None,
                            extra_symbols_columns: tuple[str] = ("title", "brand")) -> tuple[
        pd.DataFrame, pd.DataFrame]:

        formatted_datasets = self.get_preprocessed_datasets()

        logger.info(f"Extra symbols columns {extra_symbols_columns}")
        for dataset_name, symbols in self.check_extra_symbols(formatted_datasets, extra_symbols_columns).items():
            logger.info(f"{dataset_name}: {symbols}")

        concatted_datasets = pd.concat(list(formatted_datasets.values()), ignore_index=True)

        full_dataset = self.get_full_merged_dataset(concatted_datasets)
        main_dataset = self.get_main_merged_dataset(concatted_datasets)

        logger.info(f"{concatted_datasets.shape[0]} -> {full_dataset.shape[0]}")

        if save_path:
            path = Path(save_path)
            path.mkdir(parents=True, exist_ok=True)

            full_path = save_path / "full_dataset.csv"
            main_path = save_path / "main_dataset.csv"

            full_dataset.to_csv(full_path, index=False)
            main_dataset.to_csv(main_path, index=False)

        return main_dataset, full_dataset

    def _apply_images_merge(self, x: dict, path: Union[str, Path], merge_column_name: str,
                            images_column_name: str) -> pd.Series:
        path = Path(path) / x[merge_column_name]
        return pd.Series([self.processor.images_to_directory(x[images_column_name], path), str(path)])

    def _get_merged_images_dataset(self, df: pd.DataFrame, merge_column_name: str, path: Union[str, Path]):
        path = Path(path)
        df[["unique_images_count", "images"]] = df.progress_apply(
            lambda x: self._apply_images_merge(x, path, merge_column_name, "images_path"), axis=1)
        df = df.drop("images_path", axis=1)

        images_count = get_images_count(path)
        images_suffixes = get_images_suffixes(path)
        images_formats = get_images_formats(path)

        logger.info(f"{merge_column_name} dataset columns: {df.columns.values}")
        logger.info(f"{merge_column_name} dataset shape: {df.shape}")
        logger.info(f"{merge_column_name} dataset images count: {images_count}")
        logger.info(f"{merge_column_name} dataset images suffixes: {set(images_suffixes)}")
        logger.info(f"{merge_column_name} dataset images formats: {set(images_formats)}")

        return df

    def merge_images(self, main_dataset: pd.DataFrame, save_path: Union[str, Path]) -> tuple[
        pd.DataFrame, pd.DataFrame]:
        save_path = Path(save_path)

        main_dataset = main_dataset.drop(["price", "pricecurrency", "color"], axis=1)

        brands_dataset = main_dataset.groupby("brand_merge").agg(
            {"images_path": self.flatten_images_list}).reset_index()
        brands_path = save_path / "images" / "by-brands"
        brands_dataset = self._get_merged_images_dataset(brands_dataset, "brand_merge", brands_path)

        models_dataset = main_dataset.copy()
        models_path = save_path / "images" / "by-models"
        models_dataset = self._get_merged_images_dataset(models_dataset, "title_merge", models_path)

        brands_dataset.to_csv(save_path / "metadata" / "brands_dataset.csv", index=False)
        models_dataset.to_csv(save_path / "metadata" / "models_dataset.csv", index=False)

        return brands_dataset, models_dataset

    def merge(self, path: Union[str, Path] = Path("data", "merged")) -> dict[str, pd.DataFrame]:
        path = Path(path)
        main_dataset, full_dataset = self.get_merged_datasets(save_path=path / "metadata")
        brands_dataset, models_dataset = self.merge_images(main_dataset, path)
        logger.info(f"ALL MERGED")
        return {"main_dataset": main_dataset, "full_dataset": full_dataset, "brands_dataset": brands_dataset,
                "models_dataset": models_dataset}

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
                color_part = title[second_last_quote + 1:end_quote].strip()
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
    def preprocess_title(cls, title: str, brands: list, colors: list) -> str:
        title = cls.preprocess_text(title)

        for brand in brands:
            title = title.removeprefix(brand)

        for color in colors:
            title = title.removesuffix(color)

        title = title.removeprefix(" x ")  # some collabs are only half-resolved

        return title.strip()

    @classmethod
    def remove_extra_symbols(cls, input_string: str) -> str:
        return ''.join(char for char in input_string if char in cls.ALLOWED_SYMBOLS).strip()

    @staticmethod
    def remove_extra_whitespaces(text: str) -> str:
        return " ".join(text.split())

    @classmethod
    def get_extra_symbols(cls, df: pd.DataFrame, column="title") -> dict[str, list[str]]:
        out = {}
        for text in df[column].tolist():
            for symbol in text:
                if not any(symbol.lower() in d for d in
                           (cls.ALLOWED_SYMBOLS, cls.DEFAULT_REPLACEMENTS, cls.WHITESPACE_REPLACEMENTS)):
                    if symbol not in out:
                        out[symbol] = []
                    out[symbol].append(text)

        return out

    @classmethod
    def check_extra_symbols(cls, datasets: dict[str, pd.DataFrame], columns: Union[list[str], tuple[str]]) -> dict[
        str, set[str]]:
        extra_symbols = {dataset_name: set() for dataset_name in datasets}
        for dataset_name, dataset in datasets.items():
            for column in columns:
                extra_symbols[dataset_name] |= set(cls.get_extra_symbols(dataset, column).keys())

        return extra_symbols

    @staticmethod
    def flatten_images_list(images_list: list) -> list[str]:
        return np.unique([item for item in np.hstack(images_list) if not pd.isnull(item)]).tolist()

    @staticmethod
    def get_colors(path: str) -> list[str]:
        colors = list({word.strip().lower() for word in open(path, "r").readlines()})
        colors.sort(key=lambda x: (len(x), x), reverse=True)
        open(path, "w").write("\n".join(colors))
        return colors

    @staticmethod
    def merge_images_columns(columns: dict, images_columns: list[str]) -> list[str]:
        return [columns[images_column] for images_column in images_columns]


if __name__ == "__main__":
    Merger().merge()
