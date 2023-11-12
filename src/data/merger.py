import re
from abc import ABC
from string import ascii_letters, digits
from typing import Union

import pandas as pd

color_path = "notebooks/merger/color_words.txt"
color_words = set([word.strip().lower() for word in open(color_path, "r").readlines()])

allowed_symbols = ascii_letters + digits + " "
not_allowed_extra_symbols = ['‘', '’', '“', '”', '™', '®', "'", '"', "~"]
allowed_extra_symbols = {'#', '&', '*', '+', ',', '-', '.', ':', '?', '_', 'ß', 'ç', 'é', 'ê', 'ü', 'β', "%", "!", "+",
                         "=", '(', ')'}


class AbstractFormatter(ABC):
    website_name: str

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def _format_columns(self):
        self.df["title_old"] = self.df["title"]  # backup title
        self.df.columns = [x.lower() for x in self.df.columns]

    def _format_title(self):
        self.df["title"] = self.df["title"].apply(self.format_title)

    def _format_brand(self):
        self.df["brand"] = self.df["brand"].apply(self.remove_extra_whitespaces)

    def _format_description(self):
        self.df["description"] = self.df["description"].apply(self.remove_extra_whitespaces)

    def _format_price(self):
        pass

    def _format_color(self):
        pass

    def format(self):
        self.df["website"] = self.website_name
        self._format_columns()
        self._format_title()
        self._format_brand()
        self._format_description()
        self._format_price()
        self._format_color()
        self.df = self.df.drop("description", axis=1, errors="ignore")  # мб пригодится потом
        return self.df

    @classmethod
    def format_title(cls,text):
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

    @staticmethod
    def remove_color_words(text):
        # мб идти справа налево
        out = []
        text_split = text.split()
        for word in text_split:
            if word not in color_words:
                out.append(word)
        return " ".join(out)

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


class SuperkicksFormatter(AbstractFormatter):
    website_name = "superkicks"

    def _format_columns(self):
        super()._format_columns()
        self.df = self.df.drop(
            ["product_dimensions", "collection_url", "generic_name", "weight", "imported_by", "manufacturer",
             "unit_of_measurement", "marketed_by", "article_code", "country_of_origin"], axis=1)
        self.df = self.df.drop_duplicates(subset=["title", "collection_name", "url"])

    def _format_price(self):
        self.df["pricecurrency"] = "INR"
        self.df["price"] = self.df["price"].apply(lambda x: float(x.replace("₹", "").replace(",", "")))

    def _format_color(self):
        def get_color(text):
            colors = list()
            text = text.split()

            for word in text:
                if word in color_words:
                    colors.append(word)

            return list(dict.fromkeys(colors))  # preserving order

        self.df["color"] = self.df["title"].apply(get_color)


class SneakerbaasFormatter(AbstractFormatter):
    website_name = "sneakerbaas"

    def _format_columns(self):
        super()._format_columns()
        self.df = self.df.drop("collection_url", axis=1)
        self.df = self.df.drop_duplicates(subset=["title", "collection_name", "url"])

    def _format_color(self):
        def get_color(input_string: str) -> Union[list[str], None]:
            if pd.notnull(input_string):
                match = re.search(r'(Colour|Colours|Colors|Color|Kleur): (.*?)(?:-|$)', input_string, re.IGNORECASE)
                if match:
                    colors = match.group(2).strip()
                    colors = colors.replace("/", " ").lower().split()
                    return list(dict.fromkeys(colors))

        self.df["color"] = self.df["description"].apply(get_color)


class FootshopFormatter(AbstractFormatter):
    website_name = "footshop"

    def _format_columns(self):
        super()._format_columns()
        self.df = self.df.drop(["collection_url"], axis=1)
        self.df = self.df.drop_duplicates(subset=["title", "collection_name", "url"])

    def _format_price(self):
        self.df["price"] = self.df["price"].apply(lambda x: float(x.replace("€", "").replace("$", "")))

    def _format_description(self):
        return

    def _format_color(self):
        def get_color(text):
            colors = text.replace("&", "/").replace("/ ", "/").lower().split("/")
            return colors

        self.df["color"] = self.df["color"].apply(get_color)


class KickscrewFormatter(AbstractFormatter):
    website_name = "kickscrew"

    def _format_columns(self):
        super()._format_columns()
        self.df = self.df.drop("slug", axis=1)
        self.df["images_path"] = self.df.apply(
            lambda columns: [columns["right-side-img"], columns["left-side-img"], columns["front-both-img"]], axis=1)
        self.df = self.df.drop_duplicates(subset=["title", "brand", "url"])
        self.df = self.df.drop(["right-side-img", "left-side-img", "front-both-img"], axis=1)

    def _format_description(self):
        return


class HighsnobietyFormatter(AbstractFormatter):
    website_name = "highsnobiety"

    def _format_columns(self):
        super()._format_columns()
        self.df = self.df.drop("slug", axis=1)
        self.df["images_path"] = self.df.apply(
            lambda columns: [columns["right-side-img"], columns["left-side-img"], columns["front-both-img"]], axis=1)
        self.df = self.df.drop_duplicates(subset=["title", "brand", "url"])
        self.df = self.df.drop(["right-side-img", "left-side-img", "front-both-img"], axis=1)
