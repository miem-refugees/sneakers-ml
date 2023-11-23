import os
import re
from itertools import permutations
from pathlib import Path
from string import ascii_letters, digits
from typing import Union

import numpy as np
import pandas as pd
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=False)

color_path = "notebooks/merger/color_words.txt"
color_words = list({word.strip().lower() for word in open(color_path, "r").readlines()})
color_words.sort(key=len, reverse=True)

allowed_symbols = ascii_letters + digits + " "

default_replacements = {"ê": "e", "ä": "a", "é": "e", "ç": "c", "ô": "o", "ü": "u", "&amp;": "&", "β": "beta",
                        "ß": "beta", "–": "-", '‘': "'", '’': "'", '”': '"', '“': '"', "\\'\\'": '"', '""': '"',
                        "''": "'"}
whitespace_replacements = dict.fromkeys({"-", "_", "\\", "/", "&"}, " ")

additional_brands = {"andersson", "wales bonner", "crosc", "beams", "ader", "ami", "andre saraiva", "april skateboards",
                     "asphaltgold", "auralee", "awake", "beams", "bianca chandon", "billie eilish"}


class Merger:
    def __init__(self, metadata_path: str = "data/raw/metadata") -> None:
        discovered_datasets = os.listdir(metadata_path)
        self.datasets = {Path(source).stem: pd.read_csv(str(Path(metadata_path, source))) for source in
                         discovered_datasets}

        for name in self.datasets:
            self.datasets[name]["website"] = name
            self.datasets[name].columns = [x.lower() for x in self.datasets[name].columns]

        self.brands = self.get_all_brands()

    def get_all_brands(self) -> list[str]:
        brands_raw = []
        for name, dataset in self.datasets.items():
            brands_raw += dataset["brand"].to_list()

        brands_raw = set(brands_raw)
        brands_raw = set.union(brands_raw, additional_brands)

        brands_solo = {self.format_merge(text) for text in brands_raw}

        collabs_x = {f"{pair[0]} x {pair[1]}" for pair in permutations(brands_solo, 2)}
        collabs = {f"{pair[0]} {pair[1]}" for pair in permutations(brands_solo, 2)}

        brands = list(set.union(collabs_x, collabs, brands_solo))
        brands.sort(key=len, reverse=True)
        return brands

    def get_datasets(self) -> dict[str, pd.DataFrame]:
        return {"superkicks": self.datasets["superkicks"], "sneakerbaas": self.datasets["sneakerbaas"],
                "footshop": self.datasets["footshop"], "kickscrew": self.datasets["kickscrew"]}

    def get_formatted(self) -> dict[str, pd.DataFrame]:
        return {"superkicks": self.format_superkicks(self.datasets["superkicks"]),
                "sneakerbaas": self.format_sneakerbaas(self.datasets["sneakerbaas"]),
                "footshop": self.format_footshop(self.datasets["footshop"]),}
                # "kickscrew": self.format_kickscrew(self.datasets["kickscrew"])}

    def format_superkicks(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(["product-dimensions", "collection_url", "generic-name", "weight", "imported-by", "manufacturer",
                      "unit-of-measurement", "marketed-by", "article-code", "country-of-origin", "slug", "brand_slug"],
                     axis=1)
        df = df.drop_duplicates(subset=["title", "collection_name", "url"])

        df["pricecurrency"] = "INR"
        df["price"] = df["price"].apply(lambda x: float(x.replace("₹", "").replace(",", "")))

        df[["title_without_color", "color"]] = df["title"].apply(lambda x: pd.Series(self.split_title_and_color(x)))

        df["title_merge"] = df.parallel_apply(lambda x: self.format_merge_title(x["title_without_color"], self.brands),
                                              axis=1)

        df["brand_merge"] = df["brand"].apply(self.format_merge)

        df = df.drop(["description"], axis=1)

        print(df["website"][0] + " columns", df.columns, "size", df.shape)

        return df

    def format_sneakerbaas(self, df):
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

        df["title_merge"] = df.parallel_apply(lambda x: self.format_merge_title(x["title_without_color"], self.brands),
                                              axis=1)

        df["brand_merge"] = df["brand"].apply(self.format_merge)

        df = df.drop(["description", "description_color"], axis=1)

        print(df["website"][0] + " columns", df.columns, "size", df.shape)

        return df

    def format_footshop(self, df):
        df = df.drop(["collection_url", "slug", "brand_slug"], axis=1)
        df = df.drop_duplicates(subset=["title", "collection_name", "url"])

        df["price"] = df["price"].apply(lambda x: float(x.replace("€", "").replace("$", "")))

        df["title_without_color"] = df["title"].apply(lambda x: self.apply_replacements(x, default_replacements))

        df["color"] = df["color"].apply(self.split_color)

        df["title_merge"] = df.parallel_apply(lambda x: self.format_merge_title(x["title_without_color"], self.brands),
                                              axis=1)

        df["brand_merge"] = df["brand"].apply(self.format_merge)

        print(df["website"][0] + " columns", df.columns, "size", df.shape)
        return df

    def format_kickscrew(self, df):
        df["images_path"] = df.apply(
            lambda columns: [columns["right-side-img"], columns["left-side-img"], columns["front-both-img"]], axis=1)
        df = df.drop(["right-side-img", "left-side-img", "front-both-img", "slug"], axis=1)

        df = df.drop_duplicates(subset=["title", "brand", "url"])

        df[["title_without_color", "color"]] = df["title"].apply(lambda x: pd.Series(self.split_title_and_color(x)))

        df["title_merge"] = df.parallel_apply(lambda x: self.format_merge_title(x["title_without_color"], self.brands),
                                              axis=1)

        df["brand_merge"] = df["brand"].apply(self.format_merge)

        print(df["website"][0] + " columns", df.columns, "size", df.shape)

        return df

    def get_merged_dataset(self, path=None):
        formatted_datasets = self.get_formatted()

        self.check_extra_symbols(formatted_datasets, ["title", "brand"])

        concat_dataset = pd.concat(list(formatted_datasets.values()), ignore_index=True)

        merged_dataset = concat_dataset.groupby("title_merge").agg(
            {"brand_merge": lambda x: x.value_counts().index[0], "images_path": list, "title": list,
             "title_without_color": list, "brand": list, "collection_name": list, "color": list, "price": list,
             "pricecurrency": list, "url": list, "website": list}).reset_index().sort_values(by="title_merge")

        merged_dataset["images_path_unclean"] = merged_dataset["images_path"]
        merged_dataset["images_path"] = merged_dataset["images_path"].apply(
            lambda x: set([item for item in np.hstack(x) if not pd.isnull(item)]))

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

    @staticmethod
    def split_color(text):
        color_replacements = {"&": "/", " / ": "/", "/ ": "/", " /": "/"}
        for key, value in color_replacements.items():
            text = text.replace(key, value)
        colors = text.lower().split("/")
        return colors

    @classmethod
    def split_title_and_color(cls, title: str):
        title = cls.apply_replacements(title, default_replacements)
        for quote in ["'", '"']:
            end_quote = title.rfind(quote)
            second_last_quote = title.rfind(quote, 0, end_quote - 1) if end_quote != -1 else -1

            if second_last_quote != -1:
                title_part = title[:second_last_quote].strip()
                color_part = title[second_last_quote + 1:end_quote].strip()
                return title_part, color_part

        return title, ""

    @staticmethod
    def apply_replacements(text, replacements):
        for key, value in replacements.items():
            text = text.replace(key.lower(), value.lower())
            text = text.replace(key.upper(), value.upper())
        return text

    @classmethod
    def format_merge(cls, text):
        text = text.lower()

        text = cls.apply_replacements(text, default_replacements)
        text = cls.apply_replacements(text, whitespace_replacements)
        text = cls.remove_extra_symbols(text)
        text = cls.remove_extra_whitespaces(text)

        return text.strip()

    @classmethod
    def format_merge_title(cls, text, brands):
        text = cls.format_merge(text)

        for brand in brands:
            text = text.removeprefix(brand)

        for color in color_words:
            text = text.removesuffix(color)

        text = text.removeprefix(" x ")  # some collabs are only half-removed

        return text.strip()

    @staticmethod
    def remove_extra_symbols(input_string: str) -> str:
        return ''.join(char for char in input_string if char in allowed_symbols).strip()

    @staticmethod
    def remove_extra_whitespaces(text):
        return " ".join(text.split())

    @staticmethod
    def get_extra_symbols(df: pd.DataFrame, column="title"):
        out = dict()
        for text in df[column].tolist():
            for symbol in text:
                if not any(
                        symbol.lower() in d for d in (allowed_symbols, default_replacements, whitespace_replacements)):
                    if symbol not in out:
                        out[symbol] = []
                    out[symbol].append(text)

        return out

    @classmethod
    def check_extra_symbols(cls, datasets: dict, columns):
        extra_symbols = {dataset_name: [] for dataset_name in datasets}
        for dataset_name, dataset in datasets.items():
            for column in columns:
                extra_symbols[dataset_name] += list(cls.get_extra_symbols(dataset, column).keys())

        print(extra_symbols)
