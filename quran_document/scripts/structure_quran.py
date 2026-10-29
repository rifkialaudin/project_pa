#!/usr/bin/env python3
"""
Build a structured Quran document dataset from two Tanzil SQL dumps:
- Uthmani Arabic text
- Indonesian translation

The output is suitable as the next input to the embedding/indexing stage.

Expected source schema in both SQL files:
    (`index`, `sura`, `aya`, `text`)

Example:
    python structure_quran.py \
        --arabic data/raw/quran-uthmani.sql \
        --translation data/raw/id.indonesian.sql \
        --output data/processed/quran_documents.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


# Surah names used for human-readable references.
SURAH_NAMES: Dict[int, str] = {
    1: "Al-Fatihah",
    2: "Al-Baqarah",
    3: "Ali 'Imran",
    4: "An-Nisa",
    5: "Al-Ma'idah",
    6: "Al-An'am",
    7: "Al-A'raf",
    8: "Al-Anfal",
    9: "At-Taubah",
    10: "Yunus",
    11: "Hud",
    12: "Yusuf",
    13: "Ar-Ra'd",
    14: "Ibrahim",
    15: "Al-Hijr",
    16: "An-Nahl",
    17: "Al-Isra",
    18: "Al-Kahf",
    19: "Maryam",
    20: "Taha",
    21: "Al-Anbiya",
    22: "Al-Hajj",
    23: "Al-Mu'minun",
    24: "An-Nur",
    25: "Al-Furqan",
    26: "Asy-Syu'ara",
    27: "An-Naml",
    28: "Al-Qasas",
    29: "Al-Ankabut",
    30: "Ar-Rum",
    31: "Luqman",
    32: "As-Sajdah",
    33: "Al-Ahzab",
    34: "Saba",
    35: "Fatir",
    36: "Ya-Sin",
    37: "As-Saffat",
    38: "Sad",
    39: "Az-Zumar",
    40: "Ghafir",
    41: "Fussilat",
    42: "Asy-Syura",
    43: "Az-Zukhruf",
    44: "Ad-Dukhan",
    45: "Al-Jasiyah",
    46: "Al-Ahqaf",
    47: "Muhammad",
    48: "Al-Fath",
    49: "Al-Hujurat",
    50: "Qaf",
    51: "Az-Zariyat",
    52: "At-Tur",
    53: "An-Najm",
    54: "Al-Qamar",
    55: "Ar-Rahman",
    56: "Al-Waqi'ah",
    57: "Al-Hadid",
    58: "Al-Mujadilah",
    59: "Al-Hasyr",
    60: "Al-Mumtahanah",
    61: "As-Saff",
    62: "Al-Jumu'ah",
    63: "Al-Munafiqun",
    64: "At-Tagabun",
    65: "At-Talaq",
    66: "At-Tahrim",
    67: "Al-Mulk",
    68: "Al-Qalam",
    69: "Al-Haqqah",
    70: "Al-Ma'arij",
    71: "Nuh",
    72: "Al-Jinn",
    73: "Al-Muzzammil",
    74: "Al-Muddassir",
    75: "Al-Qiyamah",
    76: "Al-Insan",
    77: "Al-Mursalat",
    78: "An-Naba",
    79: "An-Nazi'at",
    80: "Abasa",
    81: "At-Takwir",
    82: "Al-Infitar",
    83: "Al-Mutaffifin",
    84: "Al-Insyiqaq",
    85: "Al-Buruj",
    86: "At-Tariq",
    87: "Al-A'la",
    88: "Al-Gasyiyah",
    89: "Al-Fajr",
    90: "Al-Balad",
    91: "Asy-Syams",
    92: "Al-Lail",
    93: "Ad-Duha",
    94: "Asy-Syarh",
    95: "At-Tin",
    96: "Al-Alaq",
    97: "Al-Qadr",
    98: "Al-Bayyinah",
    99: "Az-Zalzalah",
    100: "Al-Adiyat",
    101: "Al-Qari'ah",
    102: "At-Takasur",
    103: "Al-Asr",
    104: "Al-Humazah",
    105: "Al-Fil",
    106: "Quraisy",
    107: "Al-Ma'un",
    108: "Al-Kausar",
    109: "Al-Kafirun",
    110: "An-Nasr",
    111: "Al-Lahab",
    112: "Al-Ikhlas",
    113: "Al-Falaq",
    114: "An-Nas",
}

# One data row in the SQL files is written on one line in the supplied dumps.
# The regex deliberately accepts escaped SQL quotes/backslashes inside text.
ROW_PATTERN = re.compile(
    r"\(\s*(?P<index>\d+)\s*,\s*"
    r"(?P<sura>\d+)\s*,\s*"
    r"(?P<aya>\d+)\s*,\s*"
    r"'(?P<text>(?:\\.|[^'\\])*)'\s*\)"
)


def decode_sql_string(value: str) -> str:
    """Decode common MySQL SQL-string escapes without changing normal Unicode text."""
    # The supplied Tanzil dumps mainly contain plain text. We only decode SQL
    # escape sequences so the actual text value is reconstructed correctly.
    return (
        value.replace(r"\\", "\\")
        .replace(r"\'", "'")
        .replace(r'\"', '"')
        .replace(r"\n", "\n")
        .replace(r"\r", "\r")
        .replace(r"\t", "\t")
    )


def normalize_text(text: str) -> str:
    """Minimal normalization: trim surrounding whitespace only.

    We intentionally do not lowercase, stem, remove stopwords, strip Arabic
    diacritics, or otherwise modify the Quran text.
    """
    return text.strip()


def parse_sql_rows(sql_path: Path) -> Dict[Tuple[int, int], dict]:
    """Parse Quran rows from a Tanzil SQL dump.

    Returns:
        {(sura, aya): {"index": int, "sura": int, "aya": int, "text": str}}
    """
    if not sql_path.is_file():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")

    rows: Dict[Tuple[int, int], dict] = {}
    matched_rows = 0

    # utf-8-sig safely handles a BOM if one exists.
    with sql_path.open("r", encoding="utf-8-sig", newline="") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.lstrip().startswith("("):
                continue

            match = ROW_PATTERN.search(line)
            if not match:
                # Not every line beginning with '(' has to be a data row, but
                # recording it helps diagnose unexpected input formats.
                continue

            sura = int(match.group("sura"))
            aya = int(match.group("aya"))
            index = int(match.group("index"))
            text = normalize_text(decode_sql_string(match.group("text")))
            key = (sura, aya)

            if key in rows:
                raise ValueError(
                    f"Duplicate Quran key {key} found in {sql_path} at line {line_number}."
                )

            rows[key] = {
                "index": index,
                "sura": sura,
                "aya": aya,
                "text": text,
            }
            matched_rows += 1

    if matched_rows == 0:
        raise ValueError(
            f"No data rows were parsed from {sql_path}. "
            "Please check whether the SQL format matches the expected Tanzil dump."
        )

    return rows


def make_document(
    arabic: dict,
    translation: dict,
) -> dict:
    """Create one normalized Quran document from an Arabic/translation pair."""
    sura = arabic["sura"]
    aya = arabic["aya"]
    surah_name = SURAH_NAMES.get(sura)

    if surah_name is None:
        raise ValueError(f"Unknown surah number: {sura}")

    reference = f"QS. {surah_name}: {aya}"
    indonesian_text = translation["text"]

    return {
        "document_id": f"quran_{sura:03d}_{aya:03d}",
        "source_type": "quran",
        "reference": reference,
        "arabic_text": arabic["text"],
        "indonesian_text": indonesian_text,
        "embedding_text": f"{reference}. {indonesian_text}",
        "metadata": {
            "surah_number": sura,
            "surah_name": surah_name,
            "ayah_number": aya,
        },
    }


def validate_documents(
    arabic_rows: Dict[Tuple[int, int], dict],
    translation_rows: Dict[Tuple[int, int], dict],
    documents: List[dict],
) -> dict:
    """Validate pairing and required fields, returning a summary."""
    arabic_keys = set(arabic_rows)
    translation_keys = set(translation_rows)

    missing_translation = sorted(arabic_keys - translation_keys)
    missing_arabic = sorted(translation_keys - arabic_keys)

    empty_arabic = [
        (doc["metadata"]["surah_number"], doc["metadata"]["ayah_number"])
        for doc in documents
        if not doc["arabic_text"]
    ]
    empty_translation = [
        (doc["metadata"]["surah_number"], doc["metadata"]["ayah_number"])
        for doc in documents
        if not doc["indonesian_text"]
    ]
    duplicate_document_ids = [
        document_id
        for document_id in _find_duplicates(doc["document_id"] for doc in documents)
    ]

    if missing_translation or missing_arabic:
        raise ValueError(
            "Arabic and Indonesian datasets do not contain the same (sura, aya) keys. "
            f"Missing translation: {len(missing_translation)}; "
            f"missing Arabic: {len(missing_arabic)}."
        )

    if empty_arabic:
        raise ValueError(f"Found {len(empty_arabic)} documents with empty Arabic text.")

    if empty_translation:
        raise ValueError(
            f"Found {len(empty_translation)} documents with empty Indonesian text."
        )

    if duplicate_document_ids:
        raise ValueError(
            f"Found duplicate document_id values: {duplicate_document_ids[:10]}"
        )

    return {
        "arabic_count": len(arabic_rows),
        "translation_count": len(translation_rows),
        "document_count": len(documents),
        "surah_count": len({doc["metadata"]["surah_number"] for doc in documents}),
        "missing_translation_count": len(missing_translation),
        "missing_arabic_count": len(missing_arabic),
        "empty_arabic_count": len(empty_arabic),
        "empty_translation_count": len(empty_translation),
        "duplicate_document_id_count": len(duplicate_document_ids),
    }


def _find_duplicates(values: Iterable[str]) -> List[str]:
    seen = set()
    duplicates = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        else:
            seen.add(value)
    return sorted(duplicates)


def build_documents(
    arabic_path: Path,
    translation_path: Path,
) -> tuple[List[dict], dict]:
    arabic_rows = parse_sql_rows(arabic_path)
    translation_rows = parse_sql_rows(translation_path)

    common_keys = sorted(set(arabic_rows) & set(translation_rows))
    documents = [make_document(arabic_rows[key], translation_rows[key]) for key in common_keys]

    validation = validate_documents(arabic_rows, translation_rows, documents)
    return documents, validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Structure Tanzil Quran Arabic + Indonesian SQL data into JSON documents."
    )
    parser.add_argument(
        "--arabic",
        type=Path,
        default=Path("data/raw/quran-uthmani.sql"),
        help="Path to quran-uthmani.sql",
    )
    parser.add_argument(
        "--translation",
        type=Path,
        default=Path("data/raw/id.indonesian.sql"),
        help="Path to id.indonesian.sql",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/quran_documents.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation (default: 2)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        documents, validation = build_documents(args.arabic, args.translation)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(documents, file, ensure_ascii=False, indent=args.indent)
            file.write("\n")

        print("Document structuring berhasil.")
        print(f"Arabic rows       : {validation['arabic_count']}")
        print(f"Translation rows  : {validation['translation_count']}")
        print(f"Documents         : {validation['document_count']}")
        print(f"Surahs            : {validation['surah_count']}")
        print(f"Output            : {args.output}")
        print("Validation        : PASSED")
        return 0

    except (FileNotFoundError, ValueError, UnicodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
