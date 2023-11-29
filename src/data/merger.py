import os
import re
from itertools import permutations
from pathlib import Path
from string import ascii_letters, digits
from typing import Union

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from tqdm.auto import tqdm

from src.data.local import LocalStorage
from src.data.storage import StorageProcessor

pandarallel.initialize(progress_bar=False)
tqdm.pandas()


class Merger:
    COLOR_WORDS_PATH = "data/merged/metadata/color_words.txt"

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

        for name in self.datasets:
            self.datasets[name]["website"] = name
            self.datasets[name].columns = [x.lower() for x in self.datasets[name].columns]

        self.brands = self.get_all_brands()
        self.colors = self.get_colors(self.COLOR_WORDS_PATH)

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
                "sneakerbaas": self.preprocess_sneakerbaas(self.datasets["sneakerbaas"]),
                "footshop": self.preprocess_footshop(self.datasets["footshop"]),
                "kickscrew": self.preprocess_kickscrew(self.datasets["kickscrew"])}

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

        print(df["website"][0] + " columns", df.columns, "size", df.shape)

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

        print(df["website"][0] + " columns", df.columns, "size", df.shape)

        return df

    def preprocess_footshop(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(["collection_url", "slug", "brand_slug"], axis=1)
        df = df.drop_duplicates(subset=["title", "collection_name", "url"])

        df["price"] = df["price"].apply(lambda x: float(x.replace("€", "").replace("$", "")))

        df["title_without_color"] = df["title"].apply(lambda x: self.apply_replacements(x, self.DEFAULT_REPLACEMENTS))

        df["color"] = df["color"].apply(self.split_colors)

        df = self.create_merge_columns(df)

        print(df["website"][0] + " columns", df.columns, "size", df.shape)
        return df

    def preprocess_kickscrew(self, df: pd.DataFrame) -> pd.DataFrame:
        images_columns = ["right-side-img", "left-side-img", "front-both-img"]

        df["images_path"] = df.apply(lambda x: self.merge_images_columns(x, images_columns), axis=1)
        df = df.drop(images_columns + ["slug"], axis=1)

        df = df.drop_duplicates(subset=["title", "brand", "url"])

        df[["title_without_color", "color"]] = df["title"].apply(lambda x: pd.Series(self.split_title_and_color(x)))
        df = self.create_merge_columns(df)

        print(df["website"][0] + " columns", df.columns, "size", df.shape)

        return df

    def create_merge_columns(self, df):
        df["title_merge"] = df.parallel_apply(
            lambda x: self.preprocess_title(x["title_without_color"], self.brands, self.colors), axis=1)

        df["brand_merge"] = df["brand"].apply(self.preprocess_text)
        return df

    def get_merged_dataset(self, path: Union[str, Path] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        formatted_datasets = self.get_preprocessed_datasets()

        self.check_extra_symbols(formatted_datasets, ["title", "brand"])

        concat_dataset = pd.concat(list(formatted_datasets.values()), ignore_index=True)

        full_aggregations = {"brand_merge": list, "images_path": list, "title": list, "title_without_color": list,
                             "brand": list, "collection_name": list, "color": list, "price": list,
                             "pricecurrency": list, "url": list, "website": list}

        full_merged_dataset = concat_dataset.groupby("title_merge").agg(full_aggregations).reset_index().sort_values(
            by="title_merge")

        essential_aggregations = {"brand_merge": lambda x: x.value_counts().index[0],
                                  "images_path": self.flatten_images_list}

        essential_merged_dataset = concat_dataset.groupby("title_merge").agg(
            essential_aggregations).reset_index().sort_values(by="title_merge")

        essential_merged_dataset["brand_merge"] = essential_merged_dataset["brand_merge"].apply(
            lambda x: self.BRANDS_MAPPING[x] if x in self.BRANDS_MAPPING else x)

        danya_aggregations = {"brand": "first", "collection_name": list, "color": "first", "images_path": list,
                              "price": "first", "pricecurrency": "first", "url": list, "website": list}

        danya_dataset = concat_dataset.groupby("title_merge").agg(danya_aggregations).reset_index().sort_values(
            by="title_merge")

        print(f"{concat_dataset.shape[0]} -> {full_merged_dataset.shape[0]}")

        if path:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            essential_path = path / "dataset_essential.csv"
            full_merged_path = path / "dataset.csv"
            danya_path = path / "danya_eda.csv"
            danya_dataset.to_csv(danya_path, index=False)
            essential_merged_dataset.to_csv(essential_path, index=False)
            full_merged_dataset.to_csv(full_merged_path, index=False)

        return essential_merged_dataset, full_merged_dataset

    def merge_images(self, essentials: pd.DataFrame, path: Path) -> None:

        save_path = Path("data", "merged", "metadata")

        essentials_brands_dataset = essentials.groupby("brand_merge").agg(
            {"images_path": self.flatten_images_list}).reset_index()

        essentials_brands_dataset["unique_images_count"] = essentials_brands_dataset.progress_apply(
            lambda x: self.processor.images_to_directory(x["images_path"], path / "by-brands" / x["brand_merge"]),
            axis=1)

        essentials_models_dataset = essentials.copy()

        essentials_models_dataset["unique_images_count"] = essentials_models_dataset.progress_apply(
            lambda x: self.processor.images_to_directory(x["images_path"], path / "by-models" / x["title_merge"]),
            axis=1)

        essentials_brands_dataset.to_csv(save_path / "essential_brands.csv", index=False)
        essentials_models_dataset.to_csv(save_path / "essential_models.csv", index=False)

    def merge(self, path=Path("data", "merged", "metadata")) -> tuple[pd.DataFrame, pd.DataFrame]:
        essentials, merged_dataset = self.get_merged_dataset(path)
        self.merge_images(essentials, Path("data") / "merged" / "images")
        return essentials, merged_dataset

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
    def get_extra_symbols(cls, df: pd.DataFrame, column="title"):
        out = dict()
        for text in df[column].tolist():
            for symbol in text:
                if not any(symbol.lower() in d for d in
                           (cls.ALLOWED_SYMBOLS, cls.DEFAULT_REPLACEMENTS, cls.WHITESPACE_REPLACEMENTS)):
                    if symbol not in out:
                        out[symbol] = []
                    out[symbol].append(text)

        return out

    @classmethod
    def check_extra_symbols(cls, datasets: dict[str, pd.DataFrame], columns: list[str]):
        extra_symbols = {dataset_name: [] for dataset_name in datasets}
        for dataset_name, dataset in datasets.items():
            for column in columns:
                extra_symbols[dataset_name] += list(cls.get_extra_symbols(dataset, column).keys())

        print(extra_symbols)

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
