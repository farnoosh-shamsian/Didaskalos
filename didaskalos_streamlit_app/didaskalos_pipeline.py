from __future__ import annotations

import os
import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from markdown import markdown as markdown_to_html


CASE_MAP = {"n": "nominative", "g": "genitive", "d": "dative", "a": "accusative", "v": "vocative"}
TENSE_MAP = {
    "p": "present",
    "i": "imperfect",
    "f": "future",
    "a": "aorist",
    "r": "perfect",
    "l": "pluperfect",
    "t": "future perfect",
}
MOOD_MAP = {"i": "indicative", "s": "subjunctive", "o": "optative", "m": "imperative", "n": "infinitive", "p": "participle"}
VOICE_MAP = {"a": "active", "m": "middle", "p": "passive", "e": "middle/passive"}
SIMPLE_POS_LABELS = {
    "d": "adverb",
    "r": "preposition",
    "g": "particle",
    "c": "conjunction",
    "i": "interjection",
}
POS_CATEGORY_MAP = {
    "v": "verb",
    "l": "article",
    "p": "pronoun",
    **SIMPLE_POS_LABELS,
}


GREEK_MARK_RE = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]")


def clean_text(element):
    return " ".join(element.itertext()).split() if element is not None else []


def list_treebanks(folder: str | Path) -> pd.DataFrame:
    folder = Path(folder)
    rows = []

    if not folder.exists():
        return pd.DataFrame(columns=["file", "title", "author"])

    for xml_file in sorted(folder.glob("*.xml")):
        root = ET.parse(xml_file).getroot()
        title = " ".join(clean_text(root.find(".//title"))) or None
        author = " ".join(clean_text(root.find(".//author"))) or None

        rows.append(
            {
                "file": xml_file.name,
                "title": title,
                "author": author,
            }
        )

    return pd.DataFrame(rows)


def parse_treebank_xml(file_path: str | Path) -> pd.DataFrame:
    tree = ET.parse(file_path)
    root = tree.getroot()

    end_punct = {".", "?", ";", "!", ":"}
    data = []
    sentence_counter = 1
    token_index = 0
    current_sentence_id = f"tb_{sentence_counter}"

    for sentence in root.findall(".//sentence"):
        document_id = sentence.get("document_id")
        subdoc = sentence.get("subdoc")

        for word in sentence.findall("word"):
            token_index += 1
            form = word.get("form") or ""

            data.append(
                {
                    "sentence_id": current_sentence_id,
                    "document_id": document_id,
                    "subdoc": subdoc,
                    "word_id": word.get("id"),
                    "token_index": token_index,
                    "form": form,
                    "lemma": word.get("lemma"),
                    "postag": word.get("postag"),
                    "relation": word.get("relation"),
                    "head": word.get("head"),
                }
            )

            if form in end_punct:
                sentence_counter += 1
                current_sentence_id = f"tb_{sentence_counter}"

    return pd.DataFrame(data)


def _decode(code_map, ch: str) -> str:
    return "unknown" if ch == "-" else code_map.get(ch, ch)


def parse_postag(postag: str) -> str:
    if not isinstance(postag, str) or not postag:
        return "NA"

    pos = postag[0]

    if pos in {"n", "a"} and len(postag) > 7:
        return _decode(CASE_MAP, postag[7])

    if pos == "v" and len(postag) > 5:
        return ", ".join(
            [
                _decode(TENSE_MAP, postag[3]),
                _decode(MOOD_MAP, postag[4]),
                _decode(VOICE_MAP, postag[5]),
            ]
        )

    if pos == "l":
        return "article"

    if pos == "p":
        return "pronoun"

    if pos in SIMPLE_POS_LABELS:
        return SIMPLE_POS_LABELS[pos]

    return "NA"


def parse_pos_category(postag: str) -> str:
    if not isinstance(postag, str) or not postag:
        return "other"
    return POS_CATEGORY_MAP.get(postag[0], "noun/adjective" if postag[0] in {"n", "a"} else "other")


def normalize_frequency_row_name(label: str) -> str:
    if not isinstance(label, str):
        return label

    normalized = label.strip().lower()
    normalized = normalized.replace(", ", "_").replace(",", "_")
    normalized = normalized.replace("/", "_").replace(" ", "_")
    normalized = normalized.replace("(", "").replace(")", "")
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def normalize_greek_lemma(lemma: str) -> str:
    if not isinstance(lemma, str):
        return ""
    return "".join(c for c in unicodedata.normalize("NFD", lemma.lower().strip()) if unicodedata.category(c) != "Mn")


def parse_verb_subcategory(lemma: str, postag: str | None = None) -> str:
    if postag and not str(postag).startswith("v"):
        return ""

    lemma_n = normalize_greek_lemma(lemma)
    if not lemma_n:
        return ""
    if lemma_n.endswith("μαι"):
        return "deponent"
    if lemma_n.endswith("μι"):
        return "mi"
    if lemma_n.endswith("ω"):
        return "w"
    return "irregular"


def is_greek_lemma(lemma: str) -> bool:
    return isinstance(lemma, str) and bool(GREEK_MARK_RE.search(lemma))


def build_combined_df(folder: str | Path, selected_files: list[str]) -> pd.DataFrame:
    xml_paths = [Path(folder) / filename for filename in selected_files]
    all_dfs = []

    for file_path in xml_paths:
        df = parse_treebank_xml(file_path)
        df["file"] = os.path.basename(file_path)
        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df["syllabus"] = combined_df["postag"].apply(parse_postag)
    combined_df["pos_category"] = combined_df["postag"].apply(parse_pos_category)
    combined_df["verb_subcategory"] = combined_df.apply(
        lambda row: parse_verb_subcategory(row["lemma"], row["postag"]) if row["pos_category"] == "verb" else "",
        axis=1,
    )

    return combined_df


def build_frequency_syllabus(combined_df: pd.DataFrame) -> pd.DataFrame:
    if combined_df is None or combined_df.empty:
        return pd.DataFrame(columns=["syllabus", "pos_category", "frequency", "syllabus_normalized"])

    verb_mask = (
        combined_df["pos_category"].eq("verb")
        & combined_df["verb_subcategory"].notna()
        & combined_df["verb_subcategory"].astype(str).ne("")
    )

    syllabus_with_verb_bucket = combined_df["syllabus"].where(
        ~verb_mask,
        combined_df["syllabus"] + " (" + combined_df["verb_subcategory"] + ")",
    )

    frequency_syllabus = (
        pd.DataFrame(
            {
                "syllabus": syllabus_with_verb_bucket,
                "pos_category": combined_df["pos_category"],
            }
        )
        .groupby(["syllabus", "pos_category"], dropna=False)
        .size()
        .reset_index(name="frequency")
        .sort_values("frequency", ascending=False, ignore_index=True)
    )
    frequency_syllabus["syllabus_normalized"] = frequency_syllabus["syllabus"].apply(normalize_frequency_row_name)
    return frequency_syllabus


def syllabus_to_filename(syllabus_label: str) -> str | None:
    if pd.isna(syllabus_label) or syllabus_label == "NA":
        return None
    return normalize_frequency_row_name(syllabus_label) + ".md"


def generate_textbook_markdown(
    frequency_syllabus: pd.DataFrame,
    grammar_folder: str | Path,
    lesson_count: int = 20,
) -> str:
    markdown_content = []
    markdown_content.append("# A Frequency-Based Textbook for Ancient Greek Grammar")
    markdown_content.append("")
    markdown_content.append("This syllabus organizes grammar lessons by frequency of occurrence in the selected treebanks.")
    markdown_content.append("")
    markdown_content.append("## Table of Contents")
    markdown_content.append("")

    lesson_rows = frequency_syllabus[
        frequency_syllabus["syllabus"].notna() & (frequency_syllabus["syllabus"] != "NA")
    ].head(int(lesson_count))

    lesson_data = []
    rank = 0

    for _, row in lesson_rows.iterrows():
        rank += 1
        label = row["syllabus"]
        pos_category = row.get("pos_category", "other")
        freq = row["frequency"]
        filename = syllabus_to_filename(label)

        if filename is None:
            continue

        lesson_data.append(
            {
                "rank": rank,
                "label": label,
                "pos_category": pos_category,
                "frequency": freq,
                "filename": filename,
            }
        )
        markdown_content.append(f"{rank}. [{label}](#{filename.replace('.md', '')})")

    markdown_content.append("")
    markdown_content.append("---")
    markdown_content.append("")

    grammar_folder = Path(grammar_folder)

    for lesson in lesson_data:
        markdown_content.append(f"## {lesson['rank']}. {lesson['label']}")
        markdown_content.append(f"**Part of Speech Family:** {lesson['pos_category']}")
        markdown_content.append(f"**Frequency:** {lesson['frequency']}")
        markdown_content.append("")

        lesson_path = grammar_folder / lesson["filename"]
        if lesson_path.exists():
            try:
                markdown_content.append(lesson_path.read_text(encoding="utf-8"))
            except Exception as exc:
                markdown_content.append(f"*Error reading file: {exc}*")
        else:
            markdown_content.append(f"*Module file not found: {lesson['filename']}*")

        markdown_content.append("")
        markdown_content.append("---")
        markdown_content.append("")

    return "\n".join(markdown_content)


def generate_textbook_html(
        frequency_syllabus: pd.DataFrame,
        grammar_folder: str | Path,
        lesson_count: int = 20,
        doc_title: str = "A Frequency-Based Textbook for Ancient Greek Grammar",
) -> str:
        markdown_content = generate_textbook_markdown(
                frequency_syllabus=frequency_syllabus,
                grammar_folder=grammar_folder,
                lesson_count=lesson_count,
        )
        body_html = markdown_to_html(markdown_content, extensions=["extra", "toc", "tables"])

        return f"""<!doctype html>
<html lang=\"grc\">
<head>
    <meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
    <title>{doc_title}</title>
    <style>
        body {{
            margin: 0;
            padding: 2rem;
            font-family: Arial, sans-serif;
            line-height: 1.7;
            color: #222;
            background: #fff;
        }}
        h1, h2, h3 {{
            line-height: 1.3;
        }}
        pre {{
            padding: 1rem;
            background: #f6f8fa;
            overflow-x: auto;
        }}
        code {{
            font-family: Consolas, Monaco, monospace;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 1rem 0;
        }}
        th, td {{
            border: 1px solid #ccc;
            padding: 0.5rem;
            text-align: left;
        }}
        th {{
            background: #f0f0f0;
        }}
    </style>
</head>
<body>
{body_html}
</body>
</html>"""
