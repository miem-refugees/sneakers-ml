import re
from itertools import permutations
from pathlib import Path
from string import ascii_letters, digits
from typing import Union

import numpy as np
import pandas as pd


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
                    for d in (cls.ALLOWED_SYMBOLS, cls.DEFAULT_REPLACEMENTS, cls.WHITESPACE_REPLACEMENTS)
                ):
                    if symbol not in out:
                        out[symbol] = []
                    out[symbol].append(text)

        return out

    @classmethod
    def check_extra_symbols(cls, datasets: dict[str, pd.DataFrame], columns: tuple[str, ...]) -> dict[str, set[str]]:
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
            match = re.search(r"(Colour|Colours|Colors|Color|Kleur): (.*?)(?:-|$)", input_string, re.IGNORECASE)
            if match:
                colors = match.group(2).strip()
                colors = colors.replace("/", " ").lower().split()
                return list(dict.fromkeys(colors))
        return np.nan
