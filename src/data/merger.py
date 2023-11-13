import os
import re
from abc import ABC
from pathlib import Path
from string import ascii_letters, digits
from typing import Union

import pandas as pd

# color_path = "notebooks/merger/color_words.txt"
# color_words = set([word.strip().lower() for word in open(color_path, "r").readlines()])

allowed_symbols = ascii_letters + digits + " "
not_allowed_extra_symbols = ['‘', '’', '“', '”', '™', '®', "'", '"', "~"]
allowed_extra_symbols = {'#', '&', '*', '+', ',', '-', '.', ':', '?', '_', 'ß', 'ç', 'é', 'ê', 'ü', 'β', "%", "!", "+",
                         "=", '(', ')'}


class Merger:
    def __init__(self, metadata_path="data/raw/metadata"):
        discovered_datasets = os.listdir(metadata_path)
        try:
            self.datasets = {Path(source).stem: pd.read_csv(str(Path(metadata_path, source))) for source in
                             discovered_datasets}
            assert len(self.datasets) > 0
        except FileNotFoundError as err:
            print(f"Some dataset could not be resolved:")
            raise

    def get_datasets(self):
        return {"superkicks": self.datasets["superkicks"], "sneakerbaas": self.datasets["sneakerbaas"],
                "footshop": self.datasets["footshop"], "highsnobiety": self.datasets["highsnobiety"],
                "kickscrew": self.datasets["kickscrew"]}

    def get_formatted(self):
        return {"superkicks": self.format_superkicks(self.datasets["superkicks"]),
                "sneakerbaas": self.format_sneakerbaas(self.datasets["sneakerbaas"]),
                "footshop": self.format_footshop(self.datasets["footshop"]),
                "highsnobiety": self.format_highsnobiety(self.datasets["highsnobiety"]),
                "kickscrew": self.format_kickscrew(self.datasets["kickscrew"])}

    @staticmethod
    def format_superkicks(df):
        df.columns = [x.lower() for x in df.columns]

        df = df.drop(["product-dimensions", "collection_url", "generic-name", "weight", "imported-by", "manufacturer",
                      "unit-of-measurement", "marketed-by", "article-code", "country-of-origin"], axis=1)
        df = df.drop_duplicates(subset=["title", "collection_name", "url"])

        df["pricecurrency"] = "INR"
        df["price"] = df["price"].apply(lambda x: float(x.replace("₹", "").replace(",", "")))

        df["title_clean"] = df["title"]

        # self.df = self.df.drop("description", axis=1, errors="ignore")  # мб пригодится потом
        return df

    @staticmethod
    def format_sneakerbaas(df):

        df.columns = [x.lower() for x in df.columns]

        df = df.drop("collection_url", axis=1)
        df = df.drop_duplicates(subset=["title", "collection_name", "url"])

        def get_color(input_string: str) -> Union[list[str], None]:
            if pd.notnull(input_string):
                match = re.search(r'(Colour|Colours|Colors|Color|Kleur): (.*?)(?:-|$)', input_string, re.IGNORECASE)
                if match:
                    colors = match.group(2).strip()
                    colors = colors.replace("/", " ").lower().split()
                    return list(dict.fromkeys(colors))

        df["description_color"] = df["description"].apply(get_color)
        return df

    @staticmethod
    def format_footshop(df):
        df = df.drop(["collection_url"], axis=1)
        df = df.drop_duplicates(subset=["title", "collection_name", "url"])

        df["price"] = df["price"].apply(lambda x: float(x.replace("€", "").replace("$", "")))

        def get_color(text):
            colors = text.replace("&", "/").replace("/ ", "/").lower().split("/")
            return colors

        df["color"] = df["color"].apply(get_color)

        return df

    @staticmethod
    def format_kickscrew(df):
        df = df.drop("slug", axis=1)
        df["images_path"] = df.apply(
            lambda columns: [columns["right-side-img"], columns["left-side-img"], columns["front-both-img"]], axis=1)
        df = df.drop_duplicates(subset=["title", "brand", "url"])
        df = df.drop(["right-side-img", "left-side-img", "front-both-img"], axis=1)
        return df

    @staticmethod
    def format_highsnobiety(df):
        df = df.drop("slug", axis=1)
        df["images_path"] = df.apply(
            lambda columns: [columns["right-side-img"], columns["left-side-img"], columns["front-both-img"]], axis=1)
        df = df.drop_duplicates(subset=["title", "brand", "url"])
        df = df.drop(["right-side-img", "left-side-img", "front-both-img"], axis=1)
        return df

    @classmethod
    def format_title(cls, text):
        text = text.replace("/", " ")
        text = text.replace("|", " ")
        text = text.replace("–", "-")
        text = text.replace("&amp;", "&")

        text = cls.remove_extra_symbols(text)

        text = cls.remove_extra_whitespaces(text)
        text = text.lower()

        text = cls.remove_color_words(text)

        return text

    @staticmethod
    def remove_extra_symbols(input_string: str) -> str:
        # return ''.join(char for char in input_string if char not in not_allowed_extra_symbols)
        return ''.join(char for char in input_string if char in allowed_symbols).strip()

    @staticmethod
    def remove_extra_whitespaces(text):
        if pd.notnull(text):
            return " ".join(text.split())
        else:
            return text

    # @staticmethod
    # def remove_color_words(text):
    #     # мб идти справа налево
    #     out = []
    #     text_split = text.split()
    #     for word in text_split:
    #         if word not in color_words:
    #             out.append(word)
    #     return " ".join(out)

    @staticmethod
    def check_extra_symbols(datasets: dict, column="title"):
        def get_extra_symbols(df: pd.DataFrame, column="title") -> set:
            out = set()
            for text in df[column].tolist():
                for symbol in text:
                    if symbol not in allowed_symbols and symbol not in allowed_extra_symbols:
                        out.add(symbol)
            return out

        for dataset in datasets:
            print(dataset, get_extra_symbols(datasets[dataset], column))
