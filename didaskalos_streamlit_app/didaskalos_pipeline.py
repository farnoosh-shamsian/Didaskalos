from __future__ import annotations

import os
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from markdown import markdown as markdown_to_html

try:
    from i18n import DEFAULT_LANG, is_rtl, t
except ImportError:  # imported as part of a package rather than as a flat module
    from .i18n import DEFAULT_LANG, is_rtl, t

try:
    from treebank_parsers import parse_agdt_xml, parse_treebank_file
except ImportError:  # imported as part of a package rather than as a flat module
    from .treebank_parsers import parse_agdt_xml, parse_treebank_file

try:
    from work_catalog import format_citation
except ImportError:  # imported as part of a package rather than as a flat module
    from .work_catalog import format_citation

# Back-compat alias: parse_treebank_xml was the historical name for the AGDT
# parser before formats were pluggable. Kept so external callers keep working.
parse_treebank_xml = parse_agdt_xml


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

# First postag character of the word classes whose lemmas carry lexical content
# (noun, adjective, verb, adverb, pronoun). Everything else (article, particle,
# conjunction, preposition, interjection, punctuation) is treated as a function
# word for difficulty scoring and known-vocabulary coverage.
CONTENT_POS_PREFIXES = ("n", "a", "v", "d", "p")

# Sentence difficulty blends how rare the sentence's content words are on
# average, how rare its single rarest word is (the word that gates
# comprehension), and how long the sentence is.
DIFFICULTY_WEIGHT_MEAN_RARITY = 0.35
DIFFICULTY_WEIGHT_RAREST_WORD = 0.35
DIFFICULTY_WEIGHT_LENGTH = 0.30

# Stage-aware exercise selection: prefer sentences whose content lemmas are
# mostly vocabulary already introduced in earlier lessons.
KNOWN_LEMMA_COVERAGE_THRESHOLD = 0.70
KNOWN_FUNCTION_LEMMA_SEED_COUNT = 50


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


# Cached: these two pure helpers are called millions of times during a build
# (once per token, repeatedly per lesson) but over only a few thousand distinct
# lemma strings, so memoizing collapses that to one real computation per unique
# value. This is the single biggest CPU win in textbook generation.
@lru_cache(maxsize=None)
def _normalize_greek_lemma_cached(lemma: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", lemma.lower().strip()) if unicodedata.category(c) != "Mn")


def normalize_greek_lemma(lemma: str) -> str:
    if not isinstance(lemma, str):
        return ""
    return _normalize_greek_lemma_cached(lemma)


# Treebank lemmas carry Perseus homonym numbers (λέγω3, πάρειμι1, θύω1). The
# digit hides the ending that identifies a verb's conjugation class, which sent
# ~3,900 tokens of perfectly regular verbs into the "irregular" bucket, so it is
# stripped before any ending is inspected.
_HOMONYM_DIGITS_RE = re.compile(r"\d+$")


@lru_cache(maxsize=None)
def lemma_conjugation_key(lemma: str) -> str:
    return _HOMONYM_DIGITS_RE.sub("", normalize_greek_lemma(lemma))


IRREGULAR_VERB_BUCKET = "irregular"


def parse_verb_subcategory(lemma: str, postag: str | None = None) -> str:
    if postag and not str(postag).startswith("v"):
        return ""

    lemma_n = lemma_conjugation_key(lemma)
    if not lemma_n:
        return ""
    # Deponent (middle-only) lemmas are bucketed by conjugation class, the way
    # textbooks conjugate them: thematic -ομαι (βούλομαι, contract φοβέομαι)
    # follows the -ω paradigms, athematic -μαι (δύναμαι, ἐπίσταμαι, κεῖμαι)
    # follows the -μι paradigms. Deponency itself is a lexical property carried
    # separately (is_deponent_lemma), not a paradigm bucket.
    if lemma_n.endswith("ομαι"):
        return "w"
    if lemma_n.endswith("μαι") or lemma_n.endswith("μι"):
        return "mi"
    if lemma_n.endswith("ω"):
        return "w"
    return IRREGULAR_VERB_BUCKET


def is_deponent_lemma(lemma: str) -> bool:
    """Middle-only ("deponent") verb: the dictionary form ends in -μαι."""
    return lemma_conjugation_key(lemma).endswith("μαι")


@lru_cache(maxsize=None)
def _is_greek_lemma_cached(lemma: str) -> bool:
    return bool(GREEK_MARK_RE.search(lemma))


def is_greek_lemma(lemma: str) -> bool:
    return isinstance(lemma, str) and _is_greek_lemma_cached(lemma)


# ---------------------------------------------------------------------------
# Declension classification (declension-based textbook mode)
# ---------------------------------------------------------------------------

# AGDT 9-position postag indices.
POSTAG_NUMBER_INDEX = 2
POSTAG_GENDER_INDEX = 6
POSTAG_CASE_INDEX = 7

# Label text drives the lesson filename via normalize_frequency_row_name
# (e.g. "first declension feminine nouns" -> first_declension_feminine_nouns.md),
# so it must stay in sync with the declension module file names in lessons/en/.
# The short code (N1..N8, ADJ1..ADJ3) is kept only as the dict key /
# declension_code column.
NOUN_DECLENSION_LABELS = {
    "N1": "first declension feminine nouns",
    "N2": "first declension masculine nouns",
    "N3": "second declension masculine nouns",
    "N4": "second declension neuter nouns",
    "N5": "third declension consonant stem nouns",
    "N6": "third declension iota upsilon stem nouns",
    "N7": "third declension nasal liquid stem nouns",
    # N8 and ADJ3 are residual buckets, but their labels name what is in them
    # rather than calling them "other": frequency order can put a residual bucket
    # first (μέγας and πολύς alone outrank whole declension classes), and a
    # lesson titled "Other Adjectives" makes no sense as the first adjective
    # lesson a learner meets. See LESSON_PREREQUISITES for the ordering side.
    "N8": "sigma stem and irregular nouns",
}

ADJECTIVE_DECLENSION_LABELS = {
    "ADJ1": "first second declension adjectives",
    "ADJ2": "third declension adjectives",
    "ADJ3": "two ending and irregular adjectives",
}

DECLENSION_LABELS = {**NOUN_DECLENSION_LABELS, **ADJECTIVE_DECLENSION_LABELS}

# Keys are diacritic-stripped, lowercased, with final sigma normalized to σ
# (the output of _classification_key). Only lemmas whose nominative-singular
# ending points to the wrong class need to be listed here.
IRREGULAR_NOUN_LEXICON = {
    "γυνη": "N5",  # γυναικός: consonant stem despite ending in -η
    "παισ": "N5",  # παιδός: dental stem despite ending in -ις
    "ελπισ": "N5",  # ἐλπίδος
    "χαρισ": "N5",  # χάριτος
    "ορνισ": "N5",  # ὄρνιθος
    "ερισ": "N5",  # ἔριδος
    "κλεισ": "N5",  # κλειδός
    "νουσ": "N3",  # second declension contract
    "πλουσ": "N3",
    "ζευσ": "N8",
    "γραυσ": "N8",
    "γηρασ": "N8",
    "κερασ": "N8",
    "τερασ": "N8",
    "κρεασ": "N8",
    "υδωρ": "N7",
}

IRREGULAR_ADJECTIVE_LEXICON = {
    "πολυσ": "ADJ3",  # mixed 2nd/3rd declension paradigm
    "μεγασ": "ADJ3",  # mixed 2nd/3rd declension paradigm
}


def _classification_key(text: str) -> str:
    return lemma_conjugation_key(text).replace("ς", "σ")  # final sigma -> sigma


def _genitive_singular_signal(forms: list[str]) -> str | None:
    """Vote on the declension using the genitive-singular forms attested in the corpus.

    Returns one of: "d12" (-ου: 1st masc / 2nd decl), "d1" (-ης/-ας: 1st decl),
    "d3i" (-εως: 3rd decl iota stem), "d3s" (-ους: 3rd decl sigma stem),
    "d3" (-ος: other 3rd decl), or None when no genitive singular is attested.
    """
    counts: Counter[str] = Counter()
    for form in forms:
        key = _classification_key(form)
        if key.endswith("εωσ"):
            counts["d3i"] += 1
        elif key.endswith("ουσ"):
            counts["d3s"] += 1
        elif key.endswith("οσ"):
            counts["d3"] += 1
        elif key.endswith("ου"):
            counts["d12"] += 1
        elif key.endswith(("ησ", "ασ")):
            counts["d1"] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def classify_noun_declension(lemma: str, gender: str = "-", genitive_signal: str | None = None) -> str:
    """Classify a noun lemma into N1..N8.

    ``gender`` is the AGDT postag gender character ("m", "f", "n" or "-"),
    ideally the majority gender of the lemma across the corpus.
    ``genitive_signal`` is the output of _genitive_singular_signal for the lemma.
    """
    key = _classification_key(lemma)
    if not key:
        return "N8"
    if key in IRREGULAR_NOUN_LEXICON:
        return IRREGULAR_NOUN_LEXICON[key]

    third_declension_evidence = genitive_signal in {"d3", "d3s", "d3i"}

    # Unambiguously third-declension nominative endings.
    if key.endswith(("ευσ", "αυσ", "ουσ", "ω")):
        return "N8"  # βασιλεύς, ναῦς, βοῦς, πειθώ
    if key.endswith(("ην", "ων", "ηρ", "ωρ")):
        return "N7"  # ποιμήν, δαίμων, πατήρ, ῥήτωρ
    if key.endswith("ισ"):
        # πόλις (-εως, iota stem) vs. ἐλπίς (-ίδος, dental stem).
        if genitive_signal == "d3":
            return "N5"
        return "N6"
    if key.endswith(("υσ", "υ")):
        return "N6"  # ἰχθύς, ἄστυ

    # -μα, -ματος neuters are consonant (dental) stems: σῶμα, πρᾶγμα.
    if gender == "n" and key.endswith("μα"):
        return "N5"

    if gender == "f" and key.endswith(("α", "η")):
        return "N5" if third_declension_evidence else "N1"
    if gender == "m" and key.endswith(("ασ", "ησ")):
        # πολίτης (-ου, 1st decl) vs. Σωκράτης (-ους, sigma stem) vs. γίγας (-αντος).
        if genitive_signal == "d3s":
            return "N8"
        if genitive_signal == "d3":
            return "N5"
        return "N2"

    if key.endswith("οσ"):
        if gender == "n":
            return "N8"  # γένος, τεῖχος: sigma-stem neuters
        if third_declension_evidence:
            return "N5"
        return "N3"  # masc λόγος (rare feminines like ὁδός also land here)
    if gender == "n" and key.endswith("ον"):
        return "N4"

    # Remaining lemmas ending in a consonant: φύλαξ, νύξ, Ἑλλάς, χείρ, ...
    if key.endswith(("ξ", "ψ", "ρ", "ν", "σ")):
        return "N5"

    return "N8"


def classify_adjective_declension(lemma: str) -> str:
    key = _classification_key(lemma)
    if not key:
        return "ADJ3"
    if key in IRREGULAR_ADJECTIVE_LEXICON:
        return IRREGULAR_ADJECTIVE_LEXICON[key]
    if key.endswith(("οσ", "ουσ")):
        return "ADJ1"  # ἀγαθός, δίκαιος, contract χρυσοῦς
    if key.endswith(("υσ", "εισ", "ασ")):
        return "ADJ2"  # three-ending 3rd decl: ταχύς, χαρίεις, πᾶς, μέλας
    return "ADJ3"  # two-ending 3rd decl (-ης, -ων), comparatives, irregulars


def add_declension_features(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Add declension_code / declension_label columns for noun and adjective rows."""
    out = combined_df.copy()
    out["declension_code"] = ""
    out["declension_label"] = ""

    if out.empty or "postag" not in out.columns or "lemma" not in out.columns:
        return out

    postag = out["postag"].astype(str)
    greek_lemma_mask = out["lemma"].apply(is_greek_lemma)
    noun_mask = postag.str.startswith("n") & greek_lemma_mask
    adjective_mask = postag.str.startswith("a") & greek_lemma_mask

    if noun_mask.any():
        noun_rows = out.loc[noun_mask, ["lemma", "form", "postag"]].copy()
        noun_rows["key"] = noun_rows["lemma"].apply(_classification_key)
        noun_rows["gender"] = noun_rows["postag"].astype(str).str.slice(
            POSTAG_GENDER_INDEX, POSTAG_GENDER_INDEX + 1
        )

        gendered = noun_rows[noun_rows["gender"].isin(["m", "f", "n"])]
        majority_gender = (
            gendered.groupby("key")["gender"].agg(lambda genders: genders.value_counts().idxmax()).to_dict()
            if not gendered.empty
            else {}
        )

        genitive_singular_mask = (
            noun_rows["postag"].astype(str).str.slice(POSTAG_CASE_INDEX, POSTAG_CASE_INDEX + 1).eq("g")
            & noun_rows["postag"].astype(str).str.slice(POSTAG_NUMBER_INDEX, POSTAG_NUMBER_INDEX + 1).eq("s")
        )
        genitive_signals = {
            key: _genitive_singular_signal(group["form"].astype(str).tolist())
            for key, group in noun_rows[genitive_singular_mask].groupby("key")
        }

        code_by_key = {
            row["key"]: classify_noun_declension(
                row["lemma"],
                majority_gender.get(row["key"], "-"),
                genitive_signals.get(row["key"]),
            )
            for _, row in noun_rows.drop_duplicates("key").iterrows()
        }
        out.loc[noun_mask, "declension_code"] = noun_rows["key"].map(code_by_key)

    if adjective_mask.any():
        adjective_code_cache: dict[str, str] = {}

        def adjective_code(lemma: str) -> str:
            key = _classification_key(lemma)
            if key not in adjective_code_cache:
                adjective_code_cache[key] = classify_adjective_declension(lemma)
            return adjective_code_cache[key]

        out.loc[adjective_mask, "declension_code"] = out.loc[adjective_mask, "lemma"].map(adjective_code)

    out["declension_label"] = out["declension_code"].map(DECLENSION_LABELS).fillna("")
    return out


def apply_declension_syllabus(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Replace the case-based syllabus of noun/adjective rows with declension labels."""
    out = combined_df if "declension_label" in combined_df.columns else add_declension_features(combined_df)
    out = out.copy()

    noun_adjective_mask = out["postag"].astype(str).str.startswith(("n", "a"))
    has_label = out["declension_label"].astype(str).ne("")

    out.loc[noun_adjective_mask, "syllabus"] = "NA"
    out.loc[noun_adjective_mask & has_label, "syllabus"] = out.loc[
        noun_adjective_mask & has_label, "declension_label"
    ]
    return out


def build_declension_summary(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Per-category token counts, lemma counts and example lemmas, sorted by frequency."""
    columns = ["declension_code", "declension_label", "tokens", "distinct_lemmas", "example_lemmas"]
    if combined_df is None or combined_df.empty:
        return pd.DataFrame(columns=columns)

    df = combined_df if "declension_code" in combined_df.columns else add_declension_features(combined_df)
    classified = df[df["declension_code"].astype(str).ne("")]
    if classified.empty:
        return pd.DataFrame(columns=columns)

    summary_rows = []
    for code, label in DECLENSION_LABELS.items():
        subset = classified[classified["declension_code"] == code]
        if subset.empty:
            continue
        lemma_counts = subset["lemma"].value_counts()
        summary_rows.append(
            {
                "declension_code": code,
                "declension_label": label,
                "tokens": int(len(subset)),
                "distinct_lemmas": int(lemma_counts.size),
                "example_lemmas": ", ".join(lemma_counts.head(5).index.astype(str).tolist()),
            }
        )

    return pd.DataFrame(summary_rows, columns=columns).sort_values("tokens", ascending=False, ignore_index=True)


def build_combined_df(
    folder: str | Path,
    selected_files: list[str],
    syllabus_mode: str = "case",
    formats: Mapping[str, str | None] | None = None,
) -> pd.DataFrame:
    # ``formats`` maps a selected filename to its declared corpus format (from the
    # registry). Missing/None entries let the dispatcher auto-detect by extension
    # or content, which keeps uploads and ad-hoc URLs working.
    formats = formats or {}
    all_dfs = []

    for filename in selected_files:
        file_path = Path(folder) / filename
        df = parse_treebank_file(file_path, formats.get(filename))
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
    combined_df["is_deponent"] = (combined_df["pos_category"] == "verb") & combined_df["lemma"].map(is_deponent_lemma)

    if syllabus_mode == "declension":
        combined_df = add_declension_features(combined_df)
        combined_df = apply_declension_syllabus(combined_df)

    # Object (Python-string) columns dominate memory: a 258k-token frame lands at
    # ~220 MB, and a full "select all treebanks" run is ~940k tokens. Generation
    # needs a working frame on top of that, so a large corpus breaches Cloud Run's
    # memory limit and the container is OOM-killed (the app "just stops" with no
    # error). These columns each hold a small fixed vocabulary repeated across
    # every token, so storing them as categoricals roughly halves the frame with
    # no behavioural change. Deliberately excluded: form/lemma/sentence_id (used
    # in text assembly, regex, and groupby keys); word_id/token_index (coerced
    # with pd.to_numeric downstream); and syllabus/pos_category, which are groupby
    # keys and reassignment targets in build_frequency_syllabus — as categoricals
    # they would reject new values in .where() and make groupby emit spurious
    # zero-count combinations.
    for column in ("document_id", "subdoc", "postag", "relation", "head",
                   "file", "verb_subcategory"):
        if column in combined_df.columns:
            combined_df[column] = combined_df[column].astype("category")

    return combined_df


IRREGULAR_LESSON_LABEL = "irregular verbs"
IRREGULAR_LESSON_FILENAME = "irregular_verbs.md"

TENSE_NAMES = frozenset(TENSE_MAP.values())
MOOD_NAMES = frozenset(MOOD_MAP.values())
VOICE_NAMES = frozenset(VOICE_MAP.values())
CASE_NAMES = frozenset(CASE_MAP.values())
DECLENSION_LABEL_NAMES = frozenset(DECLENSION_LABELS.values())


def is_decodable_verb_label(label: str) -> bool:
    """True when a verb row names a real tense, mood and voice.

    ``parse_postag`` passes an unmarked slot through as "unknown" and an
    undefined tagset code through unchanged, so rows like "present, unknown,
    active" or "aorist, indicative, d" reach the syllabus looking like paradigms.
    The concept lessons, which are not tense/mood/voice rows at all, are exempt.
    """
    text = str(label)
    if text in {IRREGULAR_LESSON_LABEL, DEPONENT_LESSON_LABEL}:
        return True
    base_label, _ = split_syllabus_label_and_bucket(text)
    parts = [part.strip() for part in base_label.split(",")]
    if len(parts) != 3:
        return False
    tense, mood, voice = parts
    return tense in TENSE_NAMES and mood in MOOD_NAMES and voice in VOICE_NAMES


def is_decodable_nominal_label(label: str) -> bool:
    """True when a noun/adjective row names a real case or declension class.

    The case slot of a postag can be unmarked ("unknown") or absent entirely
    ("_" in CoNLL-U), which produces rows no lesson can serve. Both syllabus
    modes are accepted here: case mode labels a nominal row with its case,
    declension mode with its declension class.
    """
    return str(label) in CASE_NAMES or str(label) in DECLENSION_LABEL_NAMES

# In these tenses the middle and the passive are the same form, so one
# middle/passive lesson teaches both. The aorist and future are excluded: they
# build the passive on a separate -θη-/-θησ- stem and need lessons of their own.
VOICE_SYNCRETIC_TENSES = ("present", "imperfect", "perfect", "pluperfect", "future perfect")

# ...except where a single-voice lesson was deliberately written. These two keep
# their own module, so their rows must not be folded into the middle/passive one.
UNMERGED_SINGLE_VOICE_LABELS = frozenset(
    {"present, indicative, middle (w)", "imperfect, indicative, middle (w)"}
)


def _syncretic_voice_merges() -> dict[str, str]:
    merges: dict[str, str] = {}
    for tense in VOICE_SYNCRETIC_TENSES:
        for mood in MOOD_MAP.values():
            for bucket in ("", " (w)", " (mi)"):
                target = f"{tense}, {mood}, middle/passive{bucket}"
                for voice in ("middle", "passive"):
                    source = f"{tense}, {mood}, {voice}{bucket}"
                    if source not in UNMERGED_SINGLE_VOICE_LABELS:
                        merges[source] = target
    return merges


# Syllabus rows that share another row's lesson. The merge is at the row level
# only — the token-level "syllabus" value is untouched, so exercise answer keys
# and the combined-rows export still name the real case and voice of a form.
# The vocative repeats the nominative everywhere except a few singular endings,
# so nominative.md covers both and there is no vocative.md.
MERGED_SYLLABUS_LABELS = {"vocative": "nominative", **_syncretic_voice_merges()}


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

    # The irregular verbs are not a conjugation class: they are ~150 suppletive
    # aorists (εἶπον, εἶδον), perfects with present meaning (οἶδα, ἔοικα) and
    # impersonals (δεῖ, χρή), each defective in its own way. Split by
    # tense/mood/voice they produced 57 thin lessons, so they collapse into one
    # concept lesson instead — the same treatment deponency gets.
    irregular_mask = verb_mask & combined_df["verb_subcategory"].astype(str).eq(IRREGULAR_VERB_BUCKET)
    syllabus_with_verb_bucket = syllabus_with_verb_bucket.where(~irregular_mask, IRREGULAR_LESSON_LABEL)

    frequency_syllabus = (
        pd.DataFrame(
            {
                "syllabus": syllabus_with_verb_bucket.replace(MERGED_SYLLABUS_LABELS),
                "pos_category": combined_df["pos_category"],
            }
        )
        .groupby(["syllabus", "pos_category"], dropna=False)
        .size()
        .reset_index(name="frequency")
        .sort_values("frequency", ascending=False, ignore_index=True)
    )
    frequency_syllabus["syllabus_normalized"] = frequency_syllabus["syllabus"].apply(normalize_frequency_row_name)

    # Always skip placeholder rows like NA/unknown in the "other" POS bucket.
    skip_labels = {"na", "unknown", ""}
    skip_mask = (
        frequency_syllabus["pos_category"].astype(str).eq("other")
        & frequency_syllabus["syllabus_normalized"].astype(str).isin(skip_labels)
    )

    # Same treatment for verb rows the source treebank left unmarked or mis-tagged
    # ("present, unknown, active", "aorist, indicative, d"): a slot that decodes to
    # nothing is not a paradigm, so no lesson can be written for it and it should
    # not occupy a place in the syllabus.
    pos_series = frequency_syllabus["pos_category"].astype(str)
    undecodable_mask = (
        pos_series.eq("verb") & ~frequency_syllabus["syllabus"].apply(is_decodable_verb_label)
    ) | (
        pos_series.eq("noun/adjective")
        & ~frequency_syllabus["syllabus"].apply(is_decodable_nominal_label)
    )

    frequency_syllabus = frequency_syllabus.loc[~(skip_mask | undecodable_mask)].reset_index(drop=True)

    return frequency_syllabus


def syllabus_to_filename(syllabus_label: str) -> str | None:
    if pd.isna(syllabus_label):
        return None

    normalized = normalize_frequency_row_name(str(syllabus_label))
    if normalized in {"na", "unknown", ""}:
        return None
    return normalized + ".md"


SIMPLE_POS_LESSONS = {
    "adverb": "d",
    "preposition": "r",
    "particle": "g",
    "conjunction": "c",
    "interjection": "i",
}


POS_LABEL_FOR_PROMPT = {
    "verb": "verb",
    "noun/adjective": "noun or adjective",
    "article": "article",
    "pronoun": "pronoun",
    "adverb": "adverb",
    "preposition": "preposition",
    "particle": "particle",
    "conjunction": "conjunction",
    "interjection": "interjection",
}


PERSON_MAP = {"1": "1st person", "2": "2nd person", "3": "3rd person", "-": "not marked"}
NUMBER_MAP = {"s": "singular", "p": "plural", "d": "dual", "-": "not marked"}


# Bidi handling for RTL output languages (e.g. Persian). Greek is strongly
# left-to-right; inside an RTL paragraph the Unicode bidi algorithm misplaces
# neutral characters (hyphens, punctuation) that sit at the edges of a Greek
# run, so Greek runs get isolated explicitly.
_GREEK_LETTER = "Ͱ-Ͽἀ-῿"
_GREEK_MARKS = "̀-ͯ᾽᾿’'"
_GREEK_TOKEN = f"[{_GREEK_LETTER}][{_GREEK_LETTER}{_GREEK_MARKS}]*"
# The Arabic comma and semicolon are separators too: a Persian lesson writes its
# Greek lists with «،», and if that ended the run every form became its own isolate
# and the paradigm displayed right-to-left — first person last.
_GREEK_SEP = "[  ,،;؛.··‐‑-]+"
# A parenthesised tail belongs to the word in front of it: the movable ν of
# ἐστί(ν) has to stay inside the run, or it is isolated on its own and jumps past
# the form it belongs to.
_GREEK_WORD = rf"{_GREEK_TOKEN}(?:\([{_GREEK_LETTER}{_GREEK_MARKS}]+\))?"
# A tag-free phrase: Greek words joined by spaces/neutral punctuation.
_GREEK_PHRASE = f"{_GREEK_WORD}(?:{_GREEK_SEP}{_GREEK_WORD})*"
# A phrase wrapped in one balanced inline element (e.g. <strong>δέ</strong>,
# rendered from **δέ**), so emphasis inside a Greek sentence does not split
# the run into separate isolates and reverse the word order.
_GREEK_ELEM = "(?:" + "|".join(
    f"<{tag}>{_GREEK_PHRASE}</{tag}>" for tag in ("u", "em", "strong", "b", "i")
) + ")"
_GREEK_ATOM = f"(?:{_GREEK_PHRASE}|{_GREEK_ELEM})"
# A run: atoms joined by separators, with optional attached hyphens
# (endings like "-η", stems like "λυ-").
_GREEK_RUN_RE = re.compile(f"-?{_GREEK_ATOM}(?:{_GREEK_SEP}{_GREEK_ATOM})*-?")


def wrap_greek_runs_in_html(html: str) -> str:
    """Wrap runs of Greek text in ``<bdi dir="ltr">`` isolates.

    Safe to apply to rendered HTML because Greek characters only ever occur in
    text content, never inside tag markup.
    """
    return _GREEK_RUN_RE.sub(
        lambda match: f'<bdi lang="grc" dir="ltr">{match.group(0)}</bdi>',
        html,
    )


def _ltr_isolate(text: str, rtl: bool) -> str:
    """Wrap a fully-Greek fragment (possibly containing inline tags such as
    ``<u>``) in an LTR span so word order survives an RTL paragraph."""
    if not rtl:
        return text
    return f'<span lang="grc" dir="ltr">{text}</span>'


def _citation_suffix(row: Mapping[str, Any], rtl: bool) -> str:
    """Inline source citation to append to an exercise sentence line, e.g.
    ``"  (*Hom. Il. 1.1-1.7*)"``. Empty when the sentence's provenance cannot be
    resolved. The citation is Latin-script, so it is LTR-isolated to survive the
    RTL (Persian) layout intact."""
    citation = format_citation(row.get("file"), row.get("document_id"), row.get("subdoc"))
    if not citation:
        return ""
    return f"  (*{_ltr_isolate(citation, rtl)}*)"


def _pos_label(lesson_pos_category: str, lang: str) -> str:
    key = "pos_label_" + re.sub(r"[^a-z0-9]+", "_", str(lesson_pos_category).lower()).strip("_")
    value = t(key, lang)
    if value == key:
        return POS_LABEL_FOR_PROMPT.get(lesson_pos_category, "target form")
    return value


def _feature_label(feature_value: str, lang: str) -> str:
    key = "feat_" + re.sub(r"[^a-z0-9]+", "_", str(feature_value).lower()).strip("_")
    value = t(key, lang)
    return feature_value if value == key else value


def split_syllabus_label_and_bucket(syllabus_label: str) -> tuple[str, str | None]:
    if not isinstance(syllabus_label, str):
        return syllabus_label, None
    match = re.match(r"^(.*)\s\(([^()]*)\)$", syllabus_label.strip())
    if not match:
        return syllabus_label, None
    return match.group(1), match.group(2)


def decode_marked_verb_features(postag: str) -> dict[str, str]:
    if not isinstance(postag, str) or len(postag) < 6:
        return {
            "person": "unknown",
            "number": "unknown",
            "tense": "unknown",
            "voice": "unknown",
            "mood": "unknown",
        }

    person_code = postag[1] if len(postag) > 1 else "-"
    number_code = postag[2] if len(postag) > 2 else "-"
    tense_code = postag[3] if len(postag) > 3 else "-"
    mood_code = postag[4] if len(postag) > 4 else "-"
    voice_code = postag[5] if len(postag) > 5 else "-"

    return {
        "person": PERSON_MAP.get(person_code, "unknown"),
        "number": NUMBER_MAP.get(number_code, "unknown"),
        "tense": TENSE_MAP.get(tense_code, "unknown") if tense_code != "-" else "not marked",
        "voice": VOICE_MAP.get(voice_code, "unknown") if voice_code != "-" else "not marked",
        "mood": MOOD_MAP.get(mood_code, "unknown") if mood_code != "-" else "not marked",
    }


DEPONENT_LESSON_LABEL = "deponent verbs"
DEPONENT_LESSON_FILENAME = "deponent_verbs.md"


def get_topic_rows_for_label(syllabus_label: str, combined_df: pd.DataFrame) -> pd.DataFrame:
    # The synthetic "deponent verbs" concept lesson draws on every deponent
    # (middle-only lemma) verb token regardless of tense/mood/voice.
    if normalize_frequency_row_name(str(syllabus_label)) == normalize_frequency_row_name(DEPONENT_LESSON_LABEL):
        if "is_deponent" in combined_df.columns:
            return combined_df[(combined_df["pos_category"] == "verb") & combined_df["is_deponent"]].copy()
        return combined_df.iloc[0:0].copy()

    # Likewise the "irregular verbs" lesson: every token of an irregular verb,
    # whatever tense, mood or voice it is in.
    if normalize_frequency_row_name(str(syllabus_label)) == normalize_frequency_row_name(IRREGULAR_LESSON_LABEL):
        return combined_df[
            (combined_df["pos_category"] == "verb")
            & combined_df["verb_subcategory"].astype(str).eq(IRREGULAR_VERB_BUCKET)
        ].copy()

    # A lesson that absorbed other rows (nominative + vocative, or a middle and a
    # passive row folded into one middle/passive lesson) draws on the tokens of
    # all of them, so its exercises can show any of those forms.
    candidate_labels = [str(syllabus_label)] + [
        source for source, target in MERGED_SYLLABUS_LABELS.items() if target == syllabus_label
    ]

    matched = []
    for candidate in candidate_labels:
        base_label, verb_bucket = split_syllabus_label_and_bucket(candidate)
        rows = combined_df[combined_df["syllabus"] == base_label]
        if verb_bucket is not None:
            rows = rows[
                (rows["pos_category"] == "verb") & (rows["verb_subcategory"] == verb_bucket)
            ]
        if not rows.empty:
            matched.append(rows)

    if matched:
        return pd.concat(matched).copy() if len(matched) > 1 else matched[0].copy()

    normalized_target = normalize_frequency_row_name(syllabus_label)
    normalized_series = combined_df["syllabus"].apply(normalize_frequency_row_name)

    verb_suffix_map = {
        "_w": "w",
        "_mi": "mi",
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
        marked_text = re.sub(rf"(?<!\w)({re.escape(form)})(?!\w)", r"<u>\1</u>", marked_text)
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
    topic_words = topic_rows.drop_duplicates(subset=["lemma"], keep="first").head(num_words)
    return topic_words[["form", "lemma", "postag", "token_index", "sentence_index"]]


def assemble_sentences(df: pd.DataFrame) -> pd.DataFrame:
    attach_to_prev = {",", ".", ";", ":", "!", "?", ")", "']"}

    def join_forms(forms: list[str]) -> str:
        words = []
        # A trailing-hyphen token (crasis first half, e.g. "τ-") waits here for
        # the following word so the two can be glued: "τ-" + "ἀναντία" -> "τἀναντία".
        pending_prefix = ""
        for form in forms:
            token = str(form).strip()
            if not token:
                continue

            # A bare hyphen is a stray join marker with nothing to attach; drop it.
            if set(token) == {"-"}:
                continue

            # Enclitic marked with a leading hyphen (e.g. "-δὲ", "-τε"): glue it to
            # the preceding word, dropping the seam marker -> "οὐ" + "-δὲ" = "οὐδὲ".
            if token.startswith("-"):
                glued = token.lstrip("-")
                if words:
                    words[-1] += glued
                else:
                    words.append(glued)
                continue

            # Crasis first half marked with a trailing hyphen (e.g. "τ-", "κ-"):
            # hold it and prepend it to the next word.
            if token.endswith("-"):
                pending_prefix += token.rstrip("-")
                continue

            if pending_prefix:
                token = pending_prefix + token
                pending_prefix = ""

            if token in attach_to_prev and words:
                words[-1] += token
            else:
                words.append(token)

        # A dangling crasis prefix with no following word: keep it rather than lose text.
        if pending_prefix:
            words.append(pending_prefix)

        text = " ".join(words)
        text = re.sub(r"\s+([,.:;!?\)])", r"\1", text)
        text = re.sub(r"([\(\[])\s+", r"\1", text)

        # Drop bracketed index markers from source data like [0], [12].
        text = re.sub(r"\[\s*\d+\s*\]", "", text)

        # Remove hidden Unicode formatting chars that can appear as odd symbols.
        text = re.sub(r"[\u200b-\u200f\u2060\ufeff]", "", text)

        text = re.sub(r"\s+", " ", text).strip()
        return text

    # Pull the needed columns as plain lists once and address them by positional
    # group indices. The previous loop did a per-group .sort_values() and
    # per-group .iloc[0] for all ~19k sentences, which dominated build time. A
    # sentence's tokens are appended contiguously and in token order by every
    # parser, so the groups are already ordered and no per-group sort is needed.
    # groupby(...).indices preserves first-appearance order, matching the
    # ngroup(sort=False) sentence_index assigned to the token frame elsewhere.
    forms = df["form"].tolist()
    doc_ids = df["document_id"].tolist() if "document_id" in df.columns else None
    subdocs = df["subdoc"].tolist() if "subdoc" in df.columns else None
    file_ids = df["file"].tolist() if "file" in df.columns else None

    rows = []
    for sent_id, positions in df.groupby("sentence_id", sort=False).indices.items():
        first = positions[0]
        rows.append(
            {
                "sentence_id": sent_id,
                "document_id": doc_ids[first] if doc_ids is not None else None,
                "subdoc": subdocs[first] if subdocs is not None else None,
                "file": file_ids[first] if file_ids is not None else None,
                "sentence_text": join_forms([forms[position] for position in positions]),
                "word_count": len(positions),
            }
        )

    sentences = pd.DataFrame(rows)
    return _blank_whole_work_subdocs(sentences)


def _blank_whole_work_subdocs(sentences: pd.DataFrame) -> pd.DataFrame:
    """Drop non-informative ``subdoc`` values used as a citation reference.

    Some Perseus files tag *every* sentence with one coarse whole-work range
    (e.g. Lysias 1 uses ``subdoc="1-50"`` throughout), which is noise rather than
    a per-sentence citation. Where a file's ``subdoc`` is constant across all its
    sentences *and* looks like a range, blank it so the citation degrades to the
    work label. Genuinely varying refs (Homer's per-sentence line ranges) are
    untouched."""
    if sentences.empty or "subdoc" not in sentences.columns or "file" not in sentences.columns:
        return sentences

    for _, idx in sentences.groupby("file", sort=False).groups.items():
        values = sentences.loc[idx, "subdoc"].dropna().unique()
        if len(values) == 1 and "-" in str(values[0]):
            sentences.loc[idx, "subdoc"] = None

    return sentences


def add_sentence_scores(sentences_df: pd.DataFrame, combined_df: pd.DataFrame) -> pd.DataFrame:
    out = sentences_df.copy()

    greek = combined_df[combined_df["lemma"].apply(is_greek_lemma)].copy()
    if "lemma_frequency" in greek.columns:
        greek["lemma_frequency"] = pd.to_numeric(greek["lemma_frequency"], errors="coerce").fillna(0.0)
    else:
        counts = greek["lemma"].value_counts()
        greek["lemma_frequency"] = greek["lemma"].map(counts).astype(float)

    # Function words are frequent enough to drown out the words that actually
    # gate comprehension, so lexical difficulty looks at content words only.
    # Log frequencies tame the Zipf skew: a couple of articles no longer make a
    # sentence full of rare words look easy.
    content = greek[greek["postag"].astype(str).str.startswith(CONTENT_POS_PREFIXES)].copy()

    stat_columns = ["avg_log_lemma_freq", "min_log_lemma_freq"]
    out = out.drop(columns=stat_columns, errors="ignore")
    if content.empty:
        out["avg_log_lemma_freq"] = 0.0
        out["min_log_lemma_freq"] = 0.0
    else:
        content["log_lemma_freq"] = np.log1p(content["lemma_frequency"])
        sent_stats = content.groupby("sentence_id", as_index=False).agg(
            avg_log_lemma_freq=("log_lemma_freq", "mean"),
            min_log_lemma_freq=("log_lemma_freq", "min"),
        )
        out = out.merge(sent_stats, on="sentence_id", how="left")
        # Sentences with no content words carry no lexical load; score them easy.
        for column in stat_columns:
            max_value = out[column].max()
            out[column] = out[column].fillna(max_value if pd.notna(max_value) else 0.0)

    def to_0_100(series: pd.Series) -> pd.Series:
        max_value = series.max()
        if pd.notna(max_value) and max_value > 0:
            return series / max_value * 100
        return pd.Series(0.0, index=series.index)

    out["mean_rarity_score"] = 100 - to_0_100(pd.to_numeric(out["avg_log_lemma_freq"], errors="coerce").fillna(0.0))
    out["rarest_word_score"] = 100 - to_0_100(pd.to_numeric(out["min_log_lemma_freq"], errors="coerce").fillna(0.0))
    out["sentence_length_score"] = pd.to_numeric(to_0_100(out["word_count"]), errors="coerce").fillna(0.0)
    out["difficulty_score"] = (
        DIFFICULTY_WEIGHT_MEAN_RARITY * out["mean_rarity_score"]
        + DIFFICULTY_WEIGHT_RAREST_WORD * out["rarest_word_score"]
        + DIFFICULTY_WEIGHT_LENGTH * out["sentence_length_score"]
    )
    return out


def build_known_lemma_seed(
    combined_df: pd.DataFrame,
    top_n: int = KNOWN_FUNCTION_LEMMA_SEED_COUNT,
) -> set[str]:
    """Top function-word lemmas (articles, particles, conjunctions, ...) that
    every reader meets from the first page; they seed the known-vocabulary set
    used for stage-aware sentence selection."""
    if combined_df is None or combined_df.empty:
        return set()
    greek = combined_df[combined_df["lemma"].apply(is_greek_lemma)]
    function_rows = greek[~greek["postag"].astype(str).str.startswith(CONTENT_POS_PREFIXES)]
    top_lemmas = function_rows["lemma"].value_counts().head(top_n).index
    return {normalize_greek_lemma(str(lemma)) for lemma in top_lemmas}


def _known_lemma_coverage_by_sentence(combined_df: pd.DataFrame, known_lemmas: set[str]) -> pd.Series:
    """Fraction of content-word lemmas per sentence_index that are known."""
    greek = combined_df[combined_df["lemma"].apply(is_greek_lemma)]
    content = greek[greek["postag"].astype(str).str.startswith(CONTENT_POS_PREFIXES)]
    if content.empty:
        return pd.Series(dtype=float)
    known = content["lemma"].astype(str).map(normalize_greek_lemma).isin(known_lemmas)
    return known.groupby(content["sentence_index"]).mean()


def get_topic_sentences(
    syllabus_label: str,
    combined_df: pd.DataFrame,
    sentences_df: pd.DataFrame,
    num_sentences: int = 20,
    known_lemmas: set[str] | None = None,
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

    if not known_lemmas:
        return topic_sentences.sort_values("difficulty_score").head(num_sentences)

    # The lesson's own target lemmas are being taught right now, so they count
    # as known when judging whether a sentence fits the learner's stage.
    effective_known = known_lemmas | {
        normalize_greek_lemma(str(lemma)) for lemma in matching_rows["lemma"].dropna()
    }
    candidate_rows = combined_df[combined_df["sentence_index"].isin(matching_sentence_indices)]
    coverage = _known_lemma_coverage_by_sentence(candidate_rows, effective_known)
    topic_sentences["known_lemma_coverage"] = topic_sentences["sentence_index"].map(coverage).fillna(1.0)

    # Stage-appropriate sentences first; if the corpus cannot fill the quota,
    # fall back to the remaining sentences ranked by difficulty alone.
    qualified_mask = topic_sentences["known_lemma_coverage"] >= KNOWN_LEMMA_COVERAGE_THRESHOLD
    qualified = topic_sentences[qualified_mask].sort_values("difficulty_score")
    remainder = topic_sentences[~qualified_mask].sort_values("difficulty_score")
    return pd.concat([qualified, remainder]).head(num_sentences)


def format_exercise_set1(topic_words: pd.DataFrame, lesson_pos_category: str, lang: str = DEFAULT_LANG) -> str:
    if topic_words is None or topic_words.empty:
        return ""

    rtl = is_rtl(lang)
    pos_label = _pos_label(lesson_pos_category, lang)
    lines = [
        f"### {t('tb_ex1_header', lang)}",
        "",
        t("tb_ex1_prompt", lang, pos_label=pos_label),
        "",
    ]
    for idx, (_, row) in enumerate(topic_words.iterrows(), 1):
        item = t(
            "tb_ex1_item",
            lang,
            form=_ltr_isolate(str(row["form"]), rtl),
            lemma=_ltr_isolate(str(row["lemma"]), rtl),
        )
        lines.append(f"{idx}. {item}")
    lines.append("")
    return "\n".join(lines)


def _build_sentence_target_rows(
    syllabus_label: str,
    lesson_pos_category: str,
    combined_df: pd.DataFrame,
) -> pd.DataFrame:
    topic_rows = get_topic_rows_for_label(syllabus_label, combined_df)
    if topic_rows.empty:
        return pd.DataFrame()

    topic_rows = topic_rows.dropna(subset=["form", "postag", "sentence_index"]).copy()
    topic_rows["form"] = topic_rows["form"].astype(str).str.strip()
    topic_rows["lemma"] = topic_rows["lemma"].astype(str).str.strip()
    topic_rows["postag"] = topic_rows["postag"].astype(str).str.strip()
    topic_rows = topic_rows[(topic_rows["form"] != "") & (topic_rows["postag"] != "")]
    topic_rows = filter_topic_rows_by_lesson_rules(syllabus_label, lesson_pos_category, topic_rows)

    if "token_index" not in topic_rows.columns:
        if "word_id" in topic_rows.columns:
            topic_rows["token_index"] = pd.to_numeric(topic_rows["word_id"], errors="coerce")
        else:
            topic_rows["token_index"] = pd.Series(range(len(topic_rows)), index=topic_rows.index, dtype="int64")

    return topic_rows


# Minimum number of real words a sentence must have to be eligible as an
# exercise. Filters out single-word or fragment "sentences" that shouldn't
# appear as full-sentence exercises.
MIN_EXERCISE_SENTENCE_WORDS = 3

# A token counts as a "word" only if it contains at least one letter, so that
# standalone punctuation (Greek ano teleia "·", dashes, brackets) is not
# counted toward the minimum-length check.
_WORD_TOKEN_RE = re.compile(r"\w", re.UNICODE)


def _count_words(text: str) -> int:
    return sum(1 for token in text.split() if _WORD_TOKEN_RE.search(token))


def _normalize_answer_word(word: str) -> str:
    return str(word).strip().lower()


def _pick_unique_exercise_sentences(
    topic_sentences: pd.DataFrame,
    topic_rows: pd.DataFrame,
    max_sentences: int = 20,
) -> tuple[pd.DataFrame, dict[object, pd.DataFrame]]:
    if topic_sentences is None or topic_sentences.empty or topic_rows is None or topic_rows.empty:
        return pd.DataFrame(), {}

    selected_sentence_ids = []
    selected_targets_by_sentence: dict[object, pd.DataFrame] = {}
    used_sentence_texts: set[str] = set()
    used_answer_words: set[str] = set()

    # Sort the target rows once and index them by positional location per
    # sentence. The previous code sorted+copied every sentence group into its own
    # DataFrame up front (~135k tiny sort_values per build) even though only a
    # handful of sentences are ever selected; iterating numpy arrays and slicing
    # with .iloc once per chosen sentence is dramatically cheaper.
    topic_rows = topic_rows.sort_values("token_index")
    group_positions = topic_rows.groupby("sentence_index", sort=False).indices
    target_forms = topic_rows["form"].to_numpy(dtype=object)

    sentence_index_values = topic_sentences["sentence_index"].to_numpy()
    sentence_text_values = topic_sentences.get("sentence_text")
    sentence_text_values = (
        sentence_text_values.astype(str).to_numpy()
        if sentence_text_values is not None
        else np.array([""] * len(topic_sentences), dtype=object)
    )

    for sentence_index, sentence_text in zip(sentence_index_values, sentence_text_values):
        if len(selected_sentence_ids) >= max_sentences:
            break

        sentence_text_key = re.sub(r"\s+", " ", str(sentence_text).strip())

        if not sentence_text_key or sentence_text_key in used_sentence_texts:
            continue

        if _count_words(sentence_text_key) < MIN_EXERCISE_SENTENCE_WORDS:
            continue

        positions = group_positions.get(sentence_index)
        if positions is None or len(positions) == 0:
            continue

        candidate_positions = []
        for position in positions:
            answer_form = _normalize_answer_word(target_forms[position])
            if not answer_form or answer_form in used_answer_words:
                continue
            candidate_positions.append(position)

        if not candidate_positions:
            continue

        chosen_targets = topic_rows.iloc[candidate_positions]
        selected_sentence_ids.append(sentence_index)
        selected_targets_by_sentence[sentence_index] = chosen_targets
        used_sentence_texts.add(sentence_text_key)
        used_answer_words.update(_normalize_answer_word(form) for form in chosen_targets["form"].tolist())

    if not selected_sentence_ids:
        return pd.DataFrame(), {}

    selected_sentences = topic_sentences[topic_sentences["sentence_index"].isin(selected_sentence_ids)].copy()
    selected_sentences = selected_sentences.drop_duplicates(subset=["sentence_text"], keep="first")
    return selected_sentences, selected_targets_by_sentence


def _format_exercise_nonverb(
    lesson_pos_category: str,
    exercise_sentences: pd.DataFrame,
    sentence_form_lookup: dict[object, list[str]],
    lang: str = DEFAULT_LANG,
) -> str:
    if exercise_sentences is None or exercise_sentences.empty:
        return ""

    rtl = is_rtl(lang)
    pos_label = _pos_label(lesson_pos_category, lang)
    lines = [
        f"### {t('tb_ex2_header', lang)}",
        "",
        t("tb_ex2_prompt", lang, pos_label=pos_label),
        "",
    ]
    for idx, (_, row) in enumerate(exercise_sentences.iterrows(), 1):
        lines.append(f"{idx}. {_ltr_isolate(str(row['sentence_text']), rtl)}{_citation_suffix(row, rtl)}")
    lines.append("")
    lines.append(f"#### {t('tb_answer_key_header', lang)}")
    lines.append("")

    for idx, (_, row) in enumerate(exercise_sentences.iterrows(), 1):
        targets = sentence_form_lookup.get(row["sentence_index"], [])
        if targets:
            answer = _ltr_isolate(", ".join(targets), rtl)
        else:
            answer = t("tb_no_target_form", lang)
        lines.append(f"{idx}. {answer}")

    lines.append("")
    return "\n".join(lines)


def _format_exercise_verb(
    exercise_sentences: pd.DataFrame,
    sentence_verb_rows: Mapping[Any, pd.DataFrame],
    lang: str = DEFAULT_LANG,
) -> str:
    if exercise_sentences is None or exercise_sentences.empty:
        return ""

    rtl = is_rtl(lang)
    lines = [
        f"### {t('tb_ex2_verb_header', lang)}",
        "",
        t("tb_ex2_verb_prompt", lang),
        "",
    ]

    for idx, (_, row) in enumerate(exercise_sentences.iterrows(), 1):
        sentence_rows = sentence_verb_rows.get(row["sentence_index"])
        forms = set()
        if sentence_rows is not None and not sentence_rows.empty:
            forms = set(sentence_rows["form"].tolist())
        marked = mark_topic_words_in_sentence(row["sentence_text"], forms)
        lines.append(f"{idx}. {_ltr_isolate(marked, rtl)}{_citation_suffix(row, rtl)}")

    lines.append("")
    lines.append(f"#### {t('tb_answer_key_header', lang)}")
    lines.append("")

    for idx, (_, row) in enumerate(exercise_sentences.iterrows(), 1):
        sentence_rows = sentence_verb_rows.get(row["sentence_index"])
        if sentence_rows is None or sentence_rows.empty:
            lines.append(f"{idx}. " + t("tb_no_marked_verbs", lang))
            continue

        sentence_answers = []
        for _, verb_row in sentence_rows.iterrows():
            features = decode_marked_verb_features(verb_row.get("postag", ""))
            answer = t(
                "tb_verb_answer",
                lang,
                form=_ltr_isolate(str(verb_row.get("form", "")), rtl),
                lemma=_ltr_isolate(str(verb_row.get("lemma", "")), rtl),
                person=_feature_label(features["person"], lang),
                number=_feature_label(features["number"], lang),
                tense=_feature_label(features["tense"], lang),
                voice=_feature_label(features["voice"], lang),
                mood=_feature_label(features["mood"], lang),
            )
            if is_deponent_lemma(str(verb_row.get("lemma", ""))):
                answer += f" ({t('feat_deponent_note', lang)})"
            sentence_answers.append(answer)

        lines.append(f"{idx}. " + " | ".join(sentence_answers))

    lines.append("")
    return "\n".join(lines)


def generate_exercises_for_topic(
    syllabus_label: str,
    lesson_pos_category: str,
    combined_df: pd.DataFrame,
    sentences_df: pd.DataFrame,
    num_sentences: int = 20,
    lang: str = DEFAULT_LANG,
    topic_words: pd.DataFrame | None = None,
    known_lemmas: set[str] | None = None,
) -> str:
    exercise_blocks = []

    if topic_words is None:
        topic_words = get_topic_words(syllabus_label, lesson_pos_category, combined_df, num_words=15)
    words_exercise = format_exercise_set1(topic_words, lesson_pos_category, lang=lang)
    if words_exercise:
        exercise_blocks.append(words_exercise)

    topic_sentences = get_topic_sentences(
        syllabus_label=syllabus_label,
        combined_df=combined_df,
        sentences_df=sentences_df,
        num_sentences=num_sentences,
        known_lemmas=known_lemmas,
    )

    if not topic_sentences.empty:
        topic_rows = _build_sentence_target_rows(syllabus_label, lesson_pos_category, combined_df)

        if not topic_rows.empty:
            selected_sentences, selected_targets_by_sentence = _pick_unique_exercise_sentences(
                topic_sentences,
                topic_rows,
                max_sentences=num_sentences,
            )

            if selected_sentences.empty:
                return "\n".join(exercise_blocks)

            if lesson_pos_category == "verb":
                exercise_blocks.append(_format_exercise_verb(selected_sentences, selected_targets_by_sentence, lang=lang))
            else:
                sentence_form_lookup: dict[object, list[str]] = {}
                for sent_idx, grp in selected_targets_by_sentence.items():
                    ordered_forms = list(dict.fromkeys(grp["form"].tolist()))
                    sentence_form_lookup[sent_idx] = ordered_forms
                exercise_blocks.append(_format_exercise_nonverb(lesson_pos_category, selected_sentences, sentence_form_lookup, lang=lang))

    return "\n".join(exercise_blocks)


# Lesson files carry their own display title (the leading heading), so the table
# of contents is only as uniform as those files are. Titles are normalized on the
# way in: the textbook already numbers every entry and labels it a module, so a
# "Lesson:" / "درس:" prefix is redundant, and emphasis markup would make one
# entry shout in an otherwise plain contents list.
LESSON_TITLE_PREFIX_RE = re.compile(
    r"^\s*(?:lesson|module|unit|chapter|درس|بخش|مبحث|فصل)\s*[:：]\s*",
    re.IGNORECASE,
)


def normalize_lesson_title(title: str) -> str:
    cleaned = re.sub(r"^\s*#{1,6}\s+", "", str(title))
    cleaned = LESSON_TITLE_PREFIX_RE.sub("", cleaned)
    cleaned = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" #*_")
    return cleaned


# Lessons that read as a contrast with a class the learner is assumed to know
# already — the residual declension buckets and the irregular-verb paradigms.
# Frequency order alone can put them before the class they contrast with, so each
# declares the lessons it should follow. A lesson whose prerequisites are all
# absent from this syllabus keeps its frequency position; its title stands on its
# own, so nothing dangles.
LESSON_PREREQUISITES: dict[str, tuple[str, ...]] = {
    "two_ending_and_irregular_adjectives": (
        "first_second_declension_adjectives",
        "third_declension_adjectives",
    ),
    "sigma_stem_and_irregular_nouns": (
        "third_declension_consonant_stem_nouns",
        "third_declension_iota_upsilon_stem_nouns",
        "third_declension_nasal_liquid_stem_nouns",
    ),
    # An irregular verb is only recognizable as irregular against a paradigm the
    # learner already knows.
    "irregular_verbs": (
        "present_indicative_active_w",
        "aorist_indicative_active_w",
    ),
}


def lesson_prerequisites(normalized_label: str) -> tuple[str, ...]:
    """Normalized labels that should precede ``normalized_label``, if present."""
    return LESSON_PREREQUISITES.get(normalized_label, ())


def apply_lesson_prerequisite_order(lesson_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Move each dependent lesson just past the last prerequisite that outranks it.

    Everything else keeps its frequency position, so the syllabus stays
    frequency-ordered apart from these local swaps.
    """
    normalized = [normalize_frequency_row_name(str(lesson["label"])) for lesson in lesson_data]

    for _ in range(len(lesson_data)):
        for position, label in enumerate(normalized):
            prerequisites = lesson_prerequisites(label)
            if not prerequisites:
                continue
            last_prerequisite = max(
                (index for index, other in enumerate(normalized) if other in prerequisites),
                default=-1,
            )
            if last_prerequisite <= position:
                continue
            lesson_data.insert(last_prerequisite, lesson_data.pop(position))
            normalized.insert(last_prerequisite, normalized.pop(position))
            break
        else:
            break

    for index, lesson in enumerate(lesson_data, 1):
        lesson["rank"] = index
    return lesson_data


def _split_lesson_title(lesson_text: str) -> tuple[str | None, str]:
    """Return ``(title, body)`` where ``title`` is the lesson file's leading
    heading (if any) and ``body`` is the remaining markdown.

    Leading YAML frontmatter is dropped so its metadata does not leak into the
    rendered textbook.
    """
    lines = lesson_text.splitlines()
    start = 0
    if lines and lines[0].strip() == "---":
        for index in range(1, len(lines)):
            if lines[index].strip() in ("---", "..."):
                start = index + 1
                break
    for index in range(start, len(lines)):
        stripped = lines[index].strip()
        if not stripped:
            continue
        match = re.match(r"#{1,6}\s+(.+)", stripped)
        if match:
            title = match.group(1).strip()
            body = "\n".join(lines[index + 1:]).lstrip("\n")
            return (title or None), body
        break
    return None, "\n".join(lines[start:]).lstrip("\n")


def _render_source_summary(
    source_summary: Mapping[str, Any],
    lesson_count: int,
    syllabus_mode: str,
    lang: str = DEFAULT_LANG,
) -> list[str]:
    """Markdown lines for the "About This Textbook" front-matter block.

    Describes the build setting (lesson count / corpus size / how the syllabus is
    organized), the works the frequency analysis was built on (grouped by
    author), and the corpora used (name, description, license, source link), and
    closes with the copyright and license statement for the textbook itself.
    Returns an empty list when there is nothing to show so the caller can skip the
    section cleanly.
    """
    works = list(source_summary.get("works") or [])
    corpora = list(source_summary.get("corpora") or [])
    has_custom = bool(source_summary.get("has_custom_sources"))
    if not works and not corpora and not has_custom:
        return []

    mode_key = (
        "tb_syllabus_mode_declension" if syllabus_mode == "declension" else "tb_syllabus_mode_case"
    )

    lines: list[str] = [f"## {t('tb_about_header', lang)}", ""]

    # --- Setting: one sentence describing the build parameters. ---
    setting_body = t(
        "tb_setting_body",
        lang,
        lesson_count=int(lesson_count),
        work_count=int(source_summary.get("work_count", len(works))),
        token_count=int(source_summary.get("token_count", 0)),
        mode=t(mode_key, lang),
    )
    lines.append(f"**{t('tb_setting_label', lang)}** {setting_body}")
    lines.append("")

    # --- Texts included, grouped by author (author-less works bucketed last). ---
    if works:
        lines.append(f"**{t('tb_texts_label', lang)}**")
        lines.append("")
        grouped: dict[str, list[str]] = {}
        for author, work in works:
            key = author or t("tb_unknown_author", lang)
            grouped.setdefault(key, []).append(work)
        for author in sorted(grouped):
            lines.append(f"- **{author}** — {', '.join(grouped[author])}")
        lines.append("")

    # --- Corpora used: name — license (Source link). Description. ---
    if corpora or has_custom:
        lines.append(f"**{t('tb_corpora_label', lang)}**")
        lines.append("")
        for corpus in corpora:
            name = (corpus.get("name") or "").strip()
            if not name:
                continue
            license_name = (corpus.get("license") or "").strip()
            head = [
                t("tb_corpus_line", lang, name=name, license=license_name)
                if license_name
                else f"**{name}**"
            ]
            source_url = (corpus.get("source_url") or "").strip()
            if source_url:
                head.append(f"([{t('tb_corpus_source_label', lang)}]({source_url}))")
            line = f"- {' '.join(head)}"
            desc_key = f"tb_corpus_desc_{corpus.get('id')}"
            desc = t(desc_key, lang)
            if desc and desc != desc_key:
                line += f". {desc}"
            lines.append(line)
        if has_custom:
            lines.append(f"- {t('tb_corpus_custom', lang)}")
        lines.append("")

    # --- Copyright and license for the generated textbook itself. Exported files
    # leave the app entirely, so this is the only place a downstream reader can
    # learn the terms. The ShareAlike corpora above oblige us to state them.
    lines.append(f"**{t('tb_license_label', lang)}** {t('tb_copyright', lang, year=date.today().year)}")
    lines.append("")
    lines.append(t("tb_license_body", lang))
    lines.append("")

    return lines


def generate_textbook_markdown(
    frequency_syllabus: pd.DataFrame,
    grammar_folder: str | Path,
    lesson_count: int = 40,
    combined_df: pd.DataFrame | None = None,
    syllabus_mode: str = "case",
    lang: str = DEFAULT_LANG,
    source_summary: Mapping[str, Any] | None = None,
) -> str:
    starter_modules = ["about", "alphabet", "introduction_nouns", "introduction_adjectives", "introduction_verbs"]
    lesson_separator = "═════ ⟡ ═════"
    lesson_separator_markup = f"<div align=\"center\" style=\"font-size: 200%; line-height: 1.2;\">{lesson_separator}</div>"

    rtl = is_rtl(lang)
    if syllabus_mode == "declension":
        intro_text = t("tb_intro_declension", lang)
    else:
        intro_text = t("tb_intro_case", lang)

    lesson_rows = frequency_syllabus[
        frequency_syllabus["syllabus"].notna() & (frequency_syllabus["syllabus"] != "NA")
    ].head(int(lesson_count))

    lesson_data = []
    rank = 0

    # Always prepend core starter modules in this fixed order.
    for module_name in starter_modules:
        rank += 1
        lesson_data.append(
            {
                "rank": rank,
                "label": module_name,
                "pos_category": "module",
                "frequency": "core",
                "filename": f"{module_name}.md",
                "is_starter": True,
            }
        )

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
                "is_starter": False,
            }
        )

    apply_lesson_prerequisite_order(lesson_data)

    # Deponency is a lexical property, not a paradigm of its own: deponent
    # tokens are counted inside the regular voice lessons, and one concept
    # lesson introduces them right after the learner first meets the middle
    # voice. If no middle-voice lesson made the cut, the concept lesson is not
    # needed either.
    for position, lesson in enumerate(lesson_data):
        if lesson["is_starter"] or "middle" not in normalize_frequency_row_name(str(lesson["label"])):
            continue
        deponent_frequency: int | str = "—"
        if combined_df is not None and "is_deponent" in combined_df.columns:
            deponent_frequency = int(combined_df["is_deponent"].sum())
        lesson_data.insert(
            position + 1,
            {
                "rank": 0,
                "label": DEPONENT_LESSON_LABEL,
                "pos_category": "verb",
                "frequency": deponent_frequency,
                "filename": DEPONENT_LESSON_FILENAME,
                "is_starter": False,
            },
        )
        for index, entry in enumerate(lesson_data, 1):
            entry["rank"] = index
        break

    grammar_folder = Path(grammar_folder)

    # Load lesson bodies up front so the table of contents can use each file's
    # own H1 title (localized in translated lesson folders) instead of the raw
    # syllabus label.
    for lesson in lesson_data:
        lesson["display_label"] = lesson["label"]
        lesson_path = grammar_folder / lesson["filename"]
        if not lesson_path.exists():
            lesson["body"] = f"*{t('tb_module_not_found', lang, filename=lesson['filename'])}*"
            continue
        try:
            title, body = _split_lesson_title(lesson_path.read_text(encoding="utf-8"))
        except Exception as exc:
            lesson["body"] = f"*{t('tb_error_reading', lang, error=exc)}*"
            continue
        if title:
            lesson["display_label"] = normalize_lesson_title(title) or lesson["label"]
        lesson["body"] = body

    markdown_content = []
    markdown_content.append(f"# {t('tb_doc_title', lang)}")
    markdown_content.append("")
    markdown_content.append(intro_text)
    markdown_content.append("")

    if source_summary:
        markdown_content.extend(
            _render_source_summary(source_summary, lesson_count, syllabus_mode, lang)
        )

    markdown_content.append(f"## {t('tb_toc_header', lang)}")
    markdown_content.append("")

    for lesson in lesson_data:
        markdown_content.append(f"{lesson['rank']}. {lesson['display_label']}")

    markdown_content.append("")
    markdown_content.append(lesson_separator_markup)
    markdown_content.append("")

    working_combined_df = None
    working_sentences_df = None
    known_lemmas: set[str] = set()

    if combined_df is not None and not combined_df.empty:
        # Work on the passed frame directly. A full .copy() here duplicated the
        # entire token table (hundreds of MB for a large corpus) and was the main
        # driver of the OOM container kills; adding the two derived columns below
        # in place costs only a few MB. The two extra columns (lemma_frequency,
        # sentence_index) then also appear in the app's combined-rows CSV export.
        working_combined_df = combined_df

        if "lemma_frequency" not in working_combined_df.columns:
            greek_rows = working_combined_df[working_combined_df["lemma"].apply(is_greek_lemma)]
            lemma_counts = greek_rows["lemma"].value_counts()
            working_combined_df["lemma_frequency"] = working_combined_df["lemma"].map(lemma_counts).fillna(0)

        working_sentences_df = assemble_sentences(working_combined_df)
        if not working_sentences_df.empty:
            working_sentences_df["sentence_index"] = range(len(working_sentences_df))
            working_combined_df["sentence_index"] = working_combined_df.groupby("sentence_id", sort=False).ngroup()
            working_sentences_df = add_sentence_scores(working_sentences_df, working_combined_df)

        known_lemmas = build_known_lemma_seed(working_combined_df)

    for lesson in lesson_data:
        markdown_content.append(f"## {lesson['rank']}. {lesson['display_label']}")
        if lesson.get("is_starter"):
            markdown_content.append(t("tb_module_type_core", lang))
        else:
            markdown_content.append(t("tb_pos_family", lang, pos=_pos_label(lesson["pos_category"], lang)))
            markdown_content.append(t("tb_frequency", lang, frequency=lesson["frequency"]))
        markdown_content.append("")
        markdown_content.append(lesson["body"])

        if not lesson.get("is_starter"):
            markdown_content.append("")
            markdown_content.append(f"### {t('tb_exercises_header', lang)}")
            markdown_content.append("")

            if working_combined_df is not None and working_sentences_df is not None and not working_sentences_df.empty:
                topic_words = get_topic_words(
                    lesson["label"], lesson["pos_category"], working_combined_df, num_words=15
                )
                exercises = generate_exercises_for_topic(
                    lesson["label"],
                    lesson["pos_category"],
                    working_combined_df,
                    working_sentences_df,
                    lang=lang,
                    topic_words=topic_words,
                    known_lemmas=known_lemmas,
                )
                # Vocabulary introduced here counts as known for later lessons.
                if not topic_words.empty:
                    known_lemmas.update(
                        normalize_greek_lemma(str(lemma)) for lemma in topic_words["lemma"]
                    )
                if exercises:
                    markdown_content.append(exercises)
                else:
                    markdown_content.append(f"*{t('tb_no_exercises', lang, label=lesson['display_label'])}*")
            else:
                markdown_content.append(f"*{t('tb_exercises_unavailable', lang)}*")

        markdown_content.append("")
        markdown_content.append(lesson_separator_markup)
        markdown_content.append("")

    document = "\n".join(markdown_content)

    if rtl:
        # Set the base paragraph direction for the whole document. The blank
        # lines make renderers such as GitHub keep parsing the inner markdown;
        # markdown="1" does the same for python-markdown's md_in_html.
        document = f'<div dir="rtl" markdown="1">\n\n{document}\n\n</div>\n'

    return document


def generate_textbook_html(
    frequency_syllabus: pd.DataFrame,
    grammar_folder: str | Path,
    lesson_count: int = 40,
    doc_title: str | None = None,
    combined_df: pd.DataFrame | None = None,
    syllabus_mode: str = "case",
    lang: str = DEFAULT_LANG,
    markdown_content: str | None = None,
    source_summary: Mapping[str, Any] | None = None,
) -> str:
    rtl = is_rtl(lang)
    if doc_title is None:
        doc_title = t("tb_doc_title", lang)

    if markdown_content is None:
        markdown_content = generate_textbook_markdown(
            frequency_syllabus=frequency_syllabus,
            grammar_folder=grammar_folder,
            lesson_count=lesson_count,
            combined_df=combined_df,
            syllabus_mode=syllabus_mode,
            lang=lang,
            source_summary=source_summary,
        )
    # "extra" bundles md_in_html, which keeps parsing the markdown inside the
    # <div dir="rtl" markdown="1"> document wrapper.
    body_html = markdown_to_html(markdown_content, extensions=["extra", "toc", "tables"])

    if rtl:
        body_html = wrap_greek_runs_in_html(body_html)

    dir_attr = "rtl" if rtl else "ltr"
    rtl_font_link = ""
    rtl_style = ""
    if rtl:
        rtl_font_link = (
            '\n    <link rel="stylesheet" '
            'href="https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic:wght@400;500;600;700&display=swap">'
        )
        rtl_style = """
        body {
            font-family: 'Noto Naskh Arabic', 'B Lotus', 'Segoe UI', Tahoma, sans-serif;
        }
        /* Code blocks stay left-to-right even in an RTL document. */
        pre, code {
            direction: ltr;
            text-align: left;
        }
"""

    return f"""<!doctype html>
<html lang=\"{lang}\" dir=\"{dir_attr}\">
<head>
    <meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
    <title>{doc_title}</title>{rtl_font_link}
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
            text-align: start;
        }}
        th {{
            background: #f0f0f0;
        }}
{rtl_style}    </style>
</head>
<body>
{body_html}
</body>
</html>"""
