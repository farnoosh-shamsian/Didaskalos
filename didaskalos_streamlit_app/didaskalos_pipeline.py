from __future__ import annotations

import os
import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from markdown import markdown as markdown_to_html


def _force_utf8_stdio() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    import sys

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_force_utf8_stdio()


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
        combined_df["syllabus"].astype(str) + " (" + combined_df["verb_subcategory"].astype(str) + ")",
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


SIMPLE_POS_LESSONS = {
    "adverb": "d",
    "preposition": "r",
    "particle": "g",
    "conjunction": "c",
    "interjection": "i",
}


POS_EXERCISE_PROMPTS = {
    "verb": {
        1: "What is the number and person of the following verbs?",
        2: "What is the number and person of the marked verbs in each sentence?",
        3: "Find one finite verb in each sentence and give its dictionary lemma.",
    },
    "noun/adjective": {
        1: "Which declension do the following words belong to?",
        2: "What is the gender, case, and number of the marked words?",
        3: "Translate each sentence and identify the marked noun or adjective.",
    },
    "article": {
        1: "What is the gender, case, and number of the marked words?",
        2: "What is the gender, case, and number of the marked articles?",
        3: "Identify the article in each sentence and give its gender, case, and number.",
    },
    "pronoun": {
        1: "What is the gender, case, and number of the following pronouns?",
        2: "What is the gender, case, and number of the marked pronouns?",
        3: "Identify each pronoun and state its gender, case, and number.",
    },
    "adverb": {
        1: "What are the following adverbs?",
        2: "Identify the marked adverb in each sentence and note its role.",
        3: "Translate each sentence and explain how the adverb affects the meaning.",
    },
    "preposition": {
        1: "What are the following prepositions?",
        2: "Identify the marked preposition in each sentence and give the case it governs.",
        3: "State the governed case for each marked preposition and identify its complement.",
    },
    "particle": {
        1: "What are the following particles?",
        2: "Identify the marked particle in each sentence and explain its effect.",
        3: "Translate each sentence and note the contribution of the particle.",
    },
    "conjunction": {
        1: "What are the following conjunctions?",
        2: "Identify the marked conjunction in each sentence and explain what it connects.",
        3: "Translate each sentence and describe the clause or phrase joined by the conjunction.",
    },
    "interjection": {
        1: "What are the following interjections?",
        2: "Identify the marked interjection in each sentence and give its force or meaning.",
        3: "Translate each sentence and explain the interjection.",
    },
}


def split_syllabus_label_and_bucket(syllabus_label: str) -> tuple[str, str | None]:
    if not isinstance(syllabus_label, str):
        return syllabus_label, None
    match = re.match(r"^(.*)\s\(([^()]*)\)$", syllabus_label.strip())
    if not match:
        return syllabus_label, None
    return match.group(1), match.group(2)


def get_exercise_prompt(lesson_pos_category: str, set_number: int) -> str:
    return POS_EXERCISE_PROMPTS.get(lesson_pos_category, POS_EXERCISE_PROMPTS["noun/adjective"]).get(
        set_number,
        "Translate each sentence.",
    )


def decode_nominal_features(postag: str) -> str:
    if not isinstance(postag, str) or len(postag) < 8:
        return "gender: unknown; number: unknown; case: unknown"

    gender_map = {"m": "masculine", "f": "feminine", "n": "neuter", "c": "common"}
    number_map = {"s": "singular", "p": "plural", "d": "dual"}
    case_map = {"n": "nominative", "g": "genitive", "d": "dative", "a": "accusative", "v": "vocative"}

    gender = gender_map.get(postag[6], "unknown")
    number = number_map.get(postag[2], "unknown")
    case_value = case_map.get(postag[7], "unknown")
    return f"gender: {gender}; number: {number}; case: {case_value}"


def decode_set1_verb_features(postag: str) -> str:
    if not isinstance(postag, str) or len(postag) < 9:
        return "POS tag explanation unavailable"

    person_map = {"1": "1st person", "2": "2nd person", "3": "3rd person", "-": "not marked"}
    number_map = {"s": "singular", "p": "plural", "d": "dual", "-": "not marked"}
    tense_map_local = {
        "p": "present",
        "i": "imperfect",
        "f": "future",
        "a": "aorist",
        "r": "perfect",
        "l": "pluperfect",
        "t": "future perfect",
        "-": "not marked",
    }
    mood_map_local = {
        "i": "indicative",
        "s": "subjunctive",
        "o": "optative",
        "m": "imperative",
        "n": "infinitive",
        "p": "participle",
        "-": "not marked",
    }
    voice_map_local = {"a": "active", "m": "middle", "p": "passive", "e": "middle/passive", "-": "not marked"}

    person_code = postag[1]
    number_code = postag[2]
    tense_code = postag[3]
    mood_code = postag[4]
    voice_code = postag[5]

    return (
        f"postag {postag} -> "
        f"person: {person_map.get(person_code, 'unknown')} ({person_code}); "
        f"number: {number_map.get(number_code, 'unknown')} ({number_code}); "
        f"tense: {tense_map_local.get(tense_code, 'unknown')} ({tense_code}); "
        f"mood: {mood_map_local.get(mood_code, 'unknown')} ({mood_code}); "
        f"voice: {voice_map_local.get(voice_code, 'unknown')} ({voice_code})"
    )


def infer_preposition_governed_case(sentence_rows: pd.DataFrame | None, preposition_token_index: int | float | None) -> str:
    if sentence_rows is None or sentence_rows.empty or preposition_token_index is None:
        return "unknown"

    order_column = "token_index" if "token_index" in sentence_rows.columns else "word_id"
    ordered_rows = sentence_rows.sort_values(order_column)
    tail_rows = ordered_rows[ordered_rows[order_column] > preposition_token_index]

    for _, candidate in tail_rows.iterrows():
        postag = candidate.get("postag")
        if not isinstance(postag, str) or len(postag) < 8:
            continue
        if postag[0] in {"n", "a", "p"}:
            return decode_nominal_features(postag).split(";")[-1].split(":", 1)[-1].strip()
        if str(candidate.get("form", "")).strip() in {".", "?", ";", "!", ":"}:
            break

    return "unknown"


def describe_topic_word(row: pd.Series, lesson_pos_category: str, sentence_rows: pd.DataFrame | None = None) -> str:
    lemma = row.get("lemma", "")
    postag = row.get("postag", "")

    if lesson_pos_category == "verb":
        return decode_set1_verb_features(postag)
    if lesson_pos_category in {"noun/adjective", "article", "pronoun"}:
        return decode_nominal_features(postag)
    if lesson_pos_category == "preposition":
        governed_case = infer_preposition_governed_case(sentence_rows, row.get("token_index"))
        return f"preposition; governs case: {governed_case}"
    if lesson_pos_category in {"adverb", "particle", "conjunction", "interjection"}:
        return f"{lesson_pos_category}; lemma: {lemma}"
    return f"lemma: {lemma}"


def get_topic_rows_for_label(syllabus_label: str, combined_df: pd.DataFrame) -> pd.DataFrame:
    base_label, verb_bucket = split_syllabus_label_and_bucket(syllabus_label)
    if verb_bucket is None:
        direct = combined_df[combined_df["syllabus"] == syllabus_label].copy()
        if not direct.empty:
            return direct
    else:
        direct = combined_df[combined_df["syllabus"] == base_label].copy()
        if not direct.empty:
            return direct[(direct["pos_category"] == "verb") & (direct["verb_subcategory"] == verb_bucket)]

    normalized_target = normalize_frequency_row_name(syllabus_label)
    normalized_series = combined_df["syllabus"].apply(normalize_frequency_row_name)

    verb_suffix_map = {
        "_w": "w",
        "_mi": "mi",
        "_deponent": "deponent",
        "_irregular": "irregular",
    }

    for suffix, raw_bucket in verb_suffix_map.items():
        if normalized_target.endswith(suffix):
            base_norm = normalized_target[: -len(suffix)]
            return combined_df[
                (normalized_series == base_norm)
                & (combined_df["pos_category"] == "verb")
                & (combined_df["verb_subcategory"].isin([raw_bucket, suffix.lstrip("_")]))
            ].copy()

    return combined_df[normalized_series == normalized_target].copy()


def filter_topic_rows_by_lesson_rules(
    syllabus_label: str,
    lesson_pos_category: str,
    topic_rows: pd.DataFrame,
) -> pd.DataFrame:
    case_lessons = {"accusative", "dative", "genitive", "nominative", "vocative"}
    if syllabus_label == "article":
        return topic_rows[topic_rows["postag"].str.startswith("l", na=False)]
    if syllabus_label in case_lessons:
        return topic_rows[topic_rows["postag"].str.startswith(("n", "a"), na=False)]
    if lesson_pos_category == "verb":
        return topic_rows[topic_rows["postag"].str.startswith("v", na=False)]
    if lesson_pos_category == "pronoun":
        return topic_rows[topic_rows["postag"].str.startswith("p", na=False)]
    if lesson_pos_category in SIMPLE_POS_LESSONS:
        prefix = SIMPLE_POS_LESSONS[lesson_pos_category]
        return topic_rows[topic_rows["postag"].str.startswith(prefix, na=False)]
    return topic_rows


def mark_topic_words_in_sentence(sentence_text: str, target_forms: set[str]) -> str:
    if not target_forms:
        return sentence_text

    marked_text = sentence_text
    for form in sorted(target_forms, key=len, reverse=True):
        if not form:
            continue
        marked_text = re.sub(rf"(?<!\\w)({re.escape(form)})(?!\\w)", r"<u>\\1</u>", marked_text)
    return marked_text


def get_topic_words(
    syllabus_label: str,
    lesson_pos_category: str,
    combined_df: pd.DataFrame,
    num_words: int = 15,
) -> pd.DataFrame:
    topic_rows = get_topic_rows_for_label(syllabus_label, combined_df)
    if topic_rows.empty:
        return pd.DataFrame()

    topic_rows = topic_rows.dropna(subset=["form", "lemma", "postag"]).copy()
    topic_rows["form"] = topic_rows["form"].astype(str).str.strip()
    topic_rows["lemma"] = topic_rows["lemma"].astype(str).str.strip()
    topic_rows["postag"] = topic_rows["postag"].astype(str).str.strip()
    topic_rows = topic_rows[(topic_rows["form"] != "") & (topic_rows["lemma"] != "") & (topic_rows["postag"] != "")]
    topic_rows = topic_rows[topic_rows["lemma"].apply(is_greek_lemma)]

    if topic_rows.empty:
        return pd.DataFrame()

    topic_rows = filter_topic_rows_by_lesson_rules(syllabus_label, lesson_pos_category, topic_rows)
    if topic_rows.empty:
        return pd.DataFrame()

    if "lemma_frequency" not in topic_rows.columns:
        local_counts = topic_rows["lemma"].value_counts()
        topic_rows["lemma_frequency"] = topic_rows["lemma"].map(local_counts)

    topic_rows["lemma_frequency"] = pd.to_numeric(topic_rows["lemma_frequency"], errors="coerce").fillna(0)
    topic_rows = topic_rows.sort_values("lemma_frequency", ascending=False)
    return topic_rows.drop_duplicates(subset=["lemma"], keep="first").head(num_words)


def assemble_sentences(df: pd.DataFrame) -> pd.DataFrame:
    attach_to_prev = {",", ".", ";", ":", "!", "?", ")", "']"}

    def join_forms(forms: list[str]) -> str:
        words = []
        for form in forms:
            token = str(form)
            if token in attach_to_prev and words:
                words[-1] += token
            else:
                words.append(token)

        text = " ".join(words)
        text = re.sub(r"\\s+([,.:;!?\\)])", r"\\1", text)
        text = re.sub(r"([\\(\\[])\\s+", r"\\1", text)
        text = re.sub(r"[\\[\\]\\d]", "", text)
        return re.sub(r"\\s+", " ", text).strip()

    rows = []
    for sent_id, group in df.groupby("sentence_id", sort=False):
        group = group.sort_values("token_index" if "token_index" in group.columns else "word_id")
        first = group.iloc[0]
        rows.append(
            {
                "sentence_id": sent_id,
                "document_id": first.get("document_id"),
                "subdoc": first.get("subdoc"),
                "file": first.get("file"),
                "sentence_text": join_forms(group["form"].tolist()),
                "word_count": len(group),
            }
        )

    return pd.DataFrame(rows)


def add_sentence_scores(sentences_df: pd.DataFrame, combined_df: pd.DataFrame) -> pd.DataFrame:
    out = sentences_df.copy()

    greek = combined_df[combined_df["lemma"].apply(is_greek_lemma)].copy()
    if "lemma_frequency" in greek.columns:
        greek["lemma_frequency"] = pd.to_numeric(greek["lemma_frequency"], errors="coerce").fillna(0.0)
    else:
        counts = greek["lemma"].value_counts()
        greek["lemma_frequency"] = greek["lemma"].map(counts).astype(float)

    if greek.empty:
        out["avg_lemma_freq"] = 0.0
    else:
        sent_avg = greek.groupby("sentence_id", as_index=False).agg(avg_lemma_freq=("lemma_frequency", "mean"))
        out = out.drop(columns=["avg_lemma_freq"], errors="ignore").merge(sent_avg, on="sentence_id", how="left")
        out["avg_lemma_freq"] = out["avg_lemma_freq"].fillna(0.0)

    def to_0_100(series: pd.Series) -> pd.Series:
        max_value = series.max()
        if pd.notna(max_value) and max_value > 0:
            return series / max_value * 100
        return pd.Series(0.0, index=series.index)

    out["lemma_frequency_score"] = pd.to_numeric(to_0_100(out["avg_lemma_freq"]), errors="coerce").fillna(0.0)
    out["lemma_difficulty_score"] = 100 - out["lemma_frequency_score"]
    out["sentence_length_score"] = pd.to_numeric(to_0_100(out["word_count"]), errors="coerce").fillna(0.0)
    out["difficulty_score"] = (out["lemma_difficulty_score"] + 2 * out["sentence_length_score"]) / 3
    return out


def get_topic_sentences(
    syllabus_label: str,
    combined_df: pd.DataFrame,
    sentences_df: pd.DataFrame,
    num_sentences: int = 40,
) -> pd.DataFrame:
    matching_rows = get_topic_rows_for_label(syllabus_label, combined_df)
    if matching_rows.empty:
        return pd.DataFrame()

    matching_sentence_indices = set(matching_rows["sentence_index"].unique())
    if not matching_sentence_indices:
        return pd.DataFrame()

    topic_sentences = sentences_df[sentences_df["sentence_index"].isin(matching_sentence_indices)].copy()
    if topic_sentences.empty:
        return pd.DataFrame()

    return topic_sentences.sort_values("difficulty_score").head(num_sentences)


def format_exercise_set1(topic_words: pd.DataFrame, lesson_pos_category: str) -> str:
    if topic_words is None or topic_words.empty:
        return ""

    prompt = get_exercise_prompt(lesson_pos_category, 1)
    lines = ["### Exercise Set 1", "", prompt, ""]
    for idx, (_, row) in enumerate(topic_words.iterrows(), 1):
        lines.append(f"{idx}. {row['form']} (lemma: {row['lemma']})")
    lines.append("")
    return "\n".join(lines)


def format_exercise_set2(exercise_sentences: pd.DataFrame, lesson_pos_category: str) -> str:
    if exercise_sentences is None or exercise_sentences.empty:
        return ""

    prompt = get_exercise_prompt(lesson_pos_category, 2)
    lines = ["### Exercise Set 2", "", prompt, ""]
    for idx, (_, row) in enumerate(exercise_sentences.iterrows(), 1):
        sentence_text = row.get("sentence_text_marked", row["sentence_text"])
        lines.append(f"{idx}. {sentence_text}")
        lines.append("")
    return "\n".join(lines)


def format_exercise_set3(exercise_sentences: pd.DataFrame, lesson_pos_category: str) -> str:
    if exercise_sentences is None or exercise_sentences.empty:
        return ""

    prompt = get_exercise_prompt(lesson_pos_category, 3)
    lines = ["### Exercise Set 3", "", prompt, ""]
    for idx, (_, row) in enumerate(exercise_sentences.iterrows(), 1):
        sentence_text = row.get("sentence_text_marked", row["sentence_text"])
        lines.append(f"{idx}. {sentence_text}")
        lines.append("")
    return "\n".join(lines)


def format_verb_exercise_set3(exercise_sentences: pd.DataFrame) -> str:
    if exercise_sentences is None or exercise_sentences.empty:
        return ""

    lines = [
        "### Exercise Set 3 (Verbs)",
        "",
        "Find one finite verb in each sentence and give its dictionary lemma.",
        "",
    ]
    for idx, (_, row) in enumerate(exercise_sentences.iterrows(), 1):
        sentence_text = row.get("sentence_text_marked", row["sentence_text"])
        lines.append(f"{idx}. {sentence_text}")
        lines.append("")
    return "\n".join(lines)


def build_exercise_set1_answer_key(
    syllabus_label: str,
    lesson_pos_category: str,
    combined_df: pd.DataFrame,
    num_words: int = 15,
) -> str:
    topic_words = get_topic_words(syllabus_label, lesson_pos_category, combined_df, num_words=num_words)
    lines = ["#### Answer Key for Exercise Set 1", ""]

    if topic_words is None or topic_words.empty:
        lines.append("*No answer key available for Exercise Set 1.*")
        lines.append("")
        return "\n".join(lines)

    for idx, (_, row) in enumerate(topic_words.iterrows(), 1):
        sentence_rows = None
        if "sentence_index" in combined_df.columns and pd.notna(row.get("sentence_index")):
            sentence_rows = combined_df[combined_df["sentence_index"] == row["sentence_index"]].copy()
        answer = describe_topic_word(row, lesson_pos_category, sentence_rows=sentence_rows)
        lines.append(f"{idx}. {row['form']} (lemma: {row['lemma']}) -> {answer}")

    lines.append("")
    return "\n".join(lines)


def generate_exercises_for_topic(
    syllabus_label: str,
    lesson_pos_category: str,
    combined_df: pd.DataFrame,
    sentences_df: pd.DataFrame,
    num_sentences: int = 40,
    sentences_per_exercise: int = 10,
) -> str:
    exercise_blocks = []

    topic_words = get_topic_words(syllabus_label, lesson_pos_category, combined_df, num_words=15)
    exercise_set1 = format_exercise_set1(topic_words, lesson_pos_category)
    if exercise_set1:
        exercise_blocks.append(exercise_set1)

    topic_sentences = get_topic_sentences(
        syllabus_label=syllabus_label,
        combined_df=combined_df,
        sentences_df=sentences_df,
        num_sentences=num_sentences,
    )

    if not topic_sentences.empty:
        topic_rows = get_topic_rows_for_label(syllabus_label, combined_df)
        topic_rows = topic_rows.dropna(subset=["form", "postag"]).copy()
        topic_rows["form"] = topic_rows["form"].astype(str).str.strip()
        topic_rows["postag"] = topic_rows["postag"].astype(str).str.strip()
        topic_rows = topic_rows[(topic_rows["form"] != "") & (topic_rows["postag"] != "")]
        topic_rows = filter_topic_rows_by_lesson_rules(syllabus_label, lesson_pos_category, topic_rows)

        sentence_form_lookup = (
            topic_rows.groupby("sentence_index")["form"].apply(lambda series: set(series.tolist())).to_dict()
            if not topic_rows.empty
            else {}
        )

        topic_sentences = topic_sentences.copy()
        topic_sentences["sentence_text_marked"] = topic_sentences.apply(
            lambda row: mark_topic_words_in_sentence(
                row["sentence_text"],
                sentence_form_lookup.get(row["sentence_index"], set()),
            ),
            axis=1,
        )

    required_total = 2 * sentences_per_exercise
    if len(topic_sentences) >= required_total:
        exercise_set2 = format_exercise_set2(topic_sentences.iloc[0:sentences_per_exercise], lesson_pos_category)
        if lesson_pos_category == "verb":
            exercise_set3 = format_verb_exercise_set3(topic_sentences.iloc[sentences_per_exercise:required_total])
        else:
            exercise_set3 = format_exercise_set3(topic_sentences.iloc[sentences_per_exercise:required_total], lesson_pos_category)

        exercise_blocks.extend([exercise_set2, exercise_set3])

    return "\n".join(exercise_blocks)


def generate_textbook_markdown(
    frequency_syllabus: pd.DataFrame,
    grammar_folder: str | Path,
    lesson_count: int = 20,
    combined_df: pd.DataFrame | None = None,
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
    working_combined_df = None
    working_sentences_df = None

    if combined_df is not None and not combined_df.empty:
        working_combined_df = combined_df.copy()

        if "lemma_frequency" not in working_combined_df.columns:
            greek_rows = working_combined_df[working_combined_df["lemma"].apply(is_greek_lemma)]
            lemma_counts = greek_rows["lemma"].value_counts()
            working_combined_df["lemma_frequency"] = working_combined_df["lemma"].map(lemma_counts).fillna(0)

        working_sentences_df = assemble_sentences(working_combined_df)
        if not working_sentences_df.empty:
            working_sentences_df["sentence_index"] = range(len(working_sentences_df))
            working_combined_df["sentence_index"] = working_combined_df.groupby("sentence_id", sort=False).ngroup()
            working_sentences_df = add_sentence_scores(working_sentences_df, working_combined_df)

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
        markdown_content.append("### Exercises")
        markdown_content.append("")

        if working_combined_df is not None and working_sentences_df is not None and not working_sentences_df.empty:
            exercises = generate_exercises_for_topic(
                lesson["label"],
                lesson["pos_category"],
                working_combined_df,
                working_sentences_df,
            )
            if exercises:
                markdown_content.append(exercises)
            else:
                markdown_content.append(f"*No exercises available for {lesson['label']}.*")

            markdown_content.append("")
            answer_key = build_exercise_set1_answer_key(
                lesson["label"],
                lesson["pos_category"],
                working_combined_df,
                num_words=15,
            )
            if answer_key:
                markdown_content.append(answer_key)
        else:
            markdown_content.append("*Exercises are unavailable because combined treebank data was not provided.*")

        markdown_content.append("")
        markdown_content.append("---")
        markdown_content.append("")

    return "\n".join(markdown_content)


def generate_textbook_html(
        frequency_syllabus: pd.DataFrame,
        grammar_folder: str | Path,
        lesson_count: int = 20,
        doc_title: str = "A Frequency-Based Textbook for Ancient Greek Grammar",
    combined_df: pd.DataFrame | None = None,
) -> str:
        markdown_content = generate_textbook_markdown(
                frequency_syllabus=frequency_syllabus,
                grammar_folder=grammar_folder,
                lesson_count=lesson_count,
        combined_df=combined_df,
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
