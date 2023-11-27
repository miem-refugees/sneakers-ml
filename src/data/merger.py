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
    COLOR_WORDS_PATH = "notebooks/merger/color_words.txt"

    ALLOWED_SYMBOLS = ascii_letters + digits + " "

    DEFAULT_REPLACEMENTS = {"ê": "e", "ä": "a", "é": "e", "ç": "c", "ô": "o", "ü": "u", "&amp;": "&", "β": "beta",
                            "ß": "beta", "–": "-", '‘': "'", '’': "'", '”': '"', '“': '"', "\\'\\'": '"', '""': '"',
                            "''": "'"}
    WHITESPACE_REPLACEMENTS = dict.fromkeys({"-", "_", "\\", "/", "&"}, " ")

    ADDITIONAL_BRANDS = {"andersson", "wales bonner", "crosc", "beams", "ader", "ami", "andre saraiva",
                         "april skateboards", "asphaltgold", "auralee", "awake", "beams", "bianca chandon",
                         "billie eilish"}

    def __init__(self, metadata_path=Path("data", "raw", "metadata")) -> None:
        discovered_datasets = os.listdir(metadata_path)
        self.datasets = {Path(source).stem: pd.read_csv(Path(metadata_path, source)) for source in discovered_datasets}

        for name in self.datasets:
            self.datasets[name]["website"] = name
            self.datasets[name].columns = [x.lower() for x in self.datasets[name].columns]

        self.brands = self.get_all_brands()
        self.color_words = self.get_color_words(self.COLOR_WORDS_PATH)

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
        brands.sort(key=len, reverse=True)
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

        df["title_merge"] = df.parallel_apply(lambda x: self.preprocess_title(x["title_without_color"], self.brands),
                                              axis=1)

        df["brand_merge"] = df["brand"].apply(self.preprocess_text)

        df = df.drop(["description"], axis=1)

        print(df["website"][0] + " columns", df.columns, "size", df.shape)

        return df

    def preprocess_sneakerbaas(self, df):
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

        df["title_merge"] = df.parallel_apply(lambda x: self.preprocess_title(x["title_without_color"], self.brands),
                                              axis=1)

        df["brand_merge"] = df["brand"].apply(self.preprocess_text)

        df = df.drop(["description", "description_color"], axis=1)

        print(df["website"][0] + " columns", df.columns, "size", df.shape)

        return df

    def preprocess_footshop(self, df):
        df = df.drop(["collection_url", "slug", "brand_slug"], axis=1)
        df = df.drop_duplicates(subset=["title", "collection_name", "url"])

        df["price"] = df["price"].apply(lambda x: float(x.replace("€", "").replace("$", "")))

        df["title_without_color"] = df["title"].apply(lambda x: self.apply_replacements(x, self.DEFAULT_REPLACEMENTS))

        df["color"] = df["color"].apply(self.split_colors)

        df["title_merge"] = df.parallel_apply(lambda x: self.preprocess_title(x["title_without_color"], self.brands),
                                              axis=1)

        df["brand_merge"] = df["brand"].apply(self.preprocess_text)

        print(df["website"][0] + " columns", df.columns, "size", df.shape)
        return df

    def preprocess_kickscrew(self, df):
        df["images_path"] = df.apply(
            lambda columns: [columns["right-side-img"], columns["left-side-img"], columns["front-both-img"]], axis=1)
        df = df.drop(["right-side-img", "left-side-img", "front-both-img", "slug"], axis=1)

        df = df.drop_duplicates(subset=["title", "brand", "url"])

        df[["title_without_color", "color"]] = df["title"].apply(lambda x: pd.Series(self.split_title_and_color(x)))

        df["title_merge"] = df.parallel_apply(lambda x: self.preprocess_title(x["title_without_color"], self.brands),
                                              axis=1)

        df["brand_merge"] = df["brand"].apply(self.preprocess_text)

        print(df["website"][0] + " columns", df.columns, "size", df.shape)

        return df

    def get_merged_dataset(self, path: Union[str, Path] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        formatted_datasets = self.get_preprocessed_datasets()

        self.check_extra_symbols(formatted_datasets, ["title", "brand"])

        concat_dataset = pd.concat(list(formatted_datasets.values()), ignore_index=True)

        aggregations = {"brand_merge": lambda x: x.value_counts().index[0], "images_path": list, "title": list,
                        "title_without_color": list, "brand": list, "collection_name": list, "color": list,
                        "price": list, "pricecurrency": list, "url": list, "website": list}

        merged_dataset = concat_dataset.groupby("title_merge").agg(aggregations).reset_index().sort_values(
            by="title_merge")

        merged_dataset["images_path_unclean"] = merged_dataset["images_path"]
        merged_dataset["images_path"] = merged_dataset["images_path"].apply(self.flatten_images_list)

        print(f"{concat_dataset.shape[0]} -> {merged_dataset.shape[0]}")

        essentials = merged_dataset[["title_merge", "brand_merge", "images_path"]]

        if path:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            essentials_path = path / "dataset_essential.csv"
            merged_dataset_path = path / "dataset.csv"
            essentials.to_csv(essentials_path, index=False)
            merged_dataset.to_csv(merged_dataset_path, index=False)

        return essentials, merged_dataset

    def merge_images(self, essentials: pd.DataFrame, path) -> None:

        essentials.groupby("brand_merge").agg({"images_path": Merger.flatten_images_list}).reset_index().progress_apply(
            lambda x: self.processor.images_to_destination(x["images_path"], path / "by-brands" / x["brand_merge"]),
            axis=1)

        essentials.progress_apply(
            lambda x: self.processor.images_to_destination(x["images_path"], path / "by-models" / x["title_merge"]),
            axis=1)

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

    def preprocess_title(self, text: str, brands: list) -> str:
        text = self.preprocess_text(text)

        for brand in brands:
            text = text.removeprefix(brand)

        for color in self.color_words:
            text = text.removesuffix(color)

        text = text.removeprefix(" x ")  # some collabs are only half-resolved

        return text.strip()

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
        return list(set([item for item in np.hstack(images_list) if not pd.isnull(item)]))

    @staticmethod
    def get_color_words(path: str) -> list[str]:
        color_words = list({word.strip().lower() for word in open(path, "r").readlines()})
        color_words.sort(key=len, reverse=True)
        open(path, "w").write("\n".join(color_words))
        return color_words


if __name__ == "__main__":
    Merger().merge()
