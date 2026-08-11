from __future__ import annotations

import base64
import os
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


def _force_utf8_stdio() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_force_utf8_stdio()

from didaskalos_pipeline import (
    build_combined_df,
    build_frequency_syllabus,
    generate_textbook_html,
    generate_textbook_markdown,
)
from i18n import AVAILABLE_LANGS, DEFAULT_LANG, LANG_NAMES, is_rtl, rtl_css, t
from idle_timeout import render_idle_watcher
from theme import (
    LOGO_CONTAINER_KEYS,
    render_theme_sync,
    render_theme_toggle,
    resolve_theme,
)
from work_catalog import resolve_author_work, tlg_work_key


APP_DIR = Path(__file__).resolve().parent
FAVICON_PATH = APP_DIR / "assets" / "logo.png"

st.set_page_config(
    page_title="Didaskalos",
    page_icon=str(FAVICON_PATH) if FAVICON_PATH.exists() else "DB",
    layout="wide",
)

# Active language. Session state is wiped by a websocket reconnect, so the URL
# query param is the durable store. Order: URL -> session_state -> default.
qp_lang = st.query_params.get("lang")
if qp_lang in AVAILABLE_LANGS and qp_lang != st.session_state.get("lang"):
    # Seeding a widget key is only allowed before its widget is instantiated.
    st.session_state["lang"] = qp_lang
lang = st.session_state.get("lang", DEFAULT_LANG)
if st.query_params.get("lang") != lang:
    st.query_params["lang"] = lang
if is_rtl(lang):
    st.markdown(rtl_css(), unsafe_allow_html=True)

# Light or dark, resolved from the URL and applied in the browser; see theme.py.
theme = resolve_theme()
render_theme_sync(theme)
render_theme_toggle(lang, theme)

# Installed before any st.stop() below so the early-abort paths are timed out too.
render_idle_watcher(lang, theme)


def _sync_lang_query_param() -> None:
    # Selectbox on_change hook: mirror the new choice into the URL.
    st.query_params["lang"] = st.session_state["lang"]

HEADER_IMAGE_PATH = APP_DIR / "assets" / "electroplato.png"
# Greek wordmark, in two inks: the "-ink" file is the dark red-brown recolour for
# the light theme, the plain one the logo's own gold for the dark sidebar. Both
# are rendered and theme.py's stylesheet hides the one that would be invisible.
LOGO_IMAGE_STEM = "greek"


def _logo_image_path(theme: str) -> Path:
    return APP_DIR / "assets" / f"{LOGO_IMAGE_STEM}{'-ink' if theme == 'light' else ''}.png"


LOGO_IMAGE_PATHS = {
    variant: _logo_image_path(variant) for variant in LOGO_CONTAINER_KEYS
}
# Cover logo for the exported textbook, inlined so a downloaded HTML file still
# shows it offline. The markdown export keeps the plain URL instead.
TEXTBOOK_LOGO_PATH = APP_DIR / "assets" / "textbook-logo.svg"
textbook_logo_data_uri = ""
if TEXTBOOK_LOGO_PATH.exists():
    encoded_logo = base64.b64encode(TEXTBOOK_LOGO_PATH.read_bytes()).decode("ascii")
    textbook_logo_data_uri = f"data:image/svg+xml;base64,{encoded_logo}"
header_image_html = ""
if HEADER_IMAGE_PATH.exists():
    encoded_image = base64.b64encode(HEADER_IMAGE_PATH.read_bytes()).decode("ascii")
    header_image_html = (
        f'<img src="data:image/png;base64,{encoded_image}" '
        'style="float: right; width: 34%; max-width: 360px; min-width: 200px; margin: 0 0 0.9rem 1.1rem; border-radius: 8px;" '
        'alt="Didaskalos header image" />'
    )

st.title(t("app_title", lang))
st.markdown(
    f"""
    <p style="font-size: 1.2rem; font-weight: 600; margin-top: -0.3rem; margin-bottom: 1rem;">
        {t("subtitle", lang)}
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    t("intro_html", lang).replace("{header_image}", header_image_html),
    unsafe_allow_html=True,
)

GITHUB_OWNER = "farnoosh-shamsian"
GITHUB_REPO = "didaskalos"
GITHUB_BRANCH = "main"
GITHUB_TREE_API = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/git/trees/{GITHUB_BRANCH}?recursive=1"
GITHUB_RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}"
TREEBANK_PREFIX = "treebanks/perseus/"
# Manifest of treebank collections (folder + format + provenance). Drives
# discovery when present; falls back to TREEBANK_PREFIX when missing.
TREEBANK_REGISTRY_PATH = "treebanks/registry.json"
FETCH_TIMEOUT_SECONDS = 20
FETCH_MAX_WORKERS = 8
# Title/author live in the XML header, so a bounded range read fills the selector
# table without downloading whole files (the Iliad is ~20 MB).
METADATA_HEADER_BYTES = 65536
# One lesson folder per language, holding case and declension modules alike.
# Filenames are the same across languages, so a translated file shadows its
# English counterpart and missing ones fall back.
LESSON_PREFIX = "lessons/en/"
LOCALIZED_LESSON_PREFIXES = {"fa": "lessons/fa/"}
LESSON_PREFIXES = (LESSON_PREFIX,) + tuple(LOCALIZED_LESSON_PREFIXES.values())
REPO_ROOT = Path(__file__).resolve().parent.parent
STARTER_LESSON_FILES = [
    "about.md",
    "alphabet.md",
    "using_a_dictionary.md",
    "introduction_nouns.md",
    "introduction_adjectives.md",
    "introduction_verbs.md",
    "greek_dialects.md",
]


def _read_from_local_repo_if_available(source_url: str) -> bytes | None:
    # Fallback for Streamlit Cloud: read file from local checkout when HTTP fetch fails.
    try:
        parsed = urlparse(source_url)
        raw_prefix = f"/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}/"
        if not parsed.path.startswith(raw_prefix):
            return None

        repo_relative_path = parsed.path[len(raw_prefix):]
        local_path = (REPO_ROOT / repo_relative_path).resolve()

        if local_path.exists() and local_path.is_file():
            return local_path.read_bytes()
    except Exception:
        return None

    return None


def _unique_name(name: str, used_names: set[str]) -> str:
    base = Path(name).stem
    suffix = Path(name).suffix or ".xml"
    candidate = f"{base}{suffix}"
    counter = 2
    while candidate in used_names:
        candidate = f"{base}_{counter}{suffix}"
        counter += 1
    used_names.add(candidate)
    return candidate


def _extract_xml_metadata(xml_bytes: bytes) -> tuple[str | None, str | None, str | None]:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return None, None, None

    def _text_for(xpath: str) -> str | None:
        element = root.find(xpath)
        if element is None:
            return None
        value = " ".join(element.itertext()).strip()
        return value or None

    title = _text_for(".//title")
    author = _text_for(".//author")
    sentence = root.find(".//sentence")
    document_id = sentence.get("document_id") if sentence is not None else None
    return title, author, document_id


def _extract_xml_metadata_from_header(header_bytes: bytes) -> tuple[str | None, str | None, str | None]:
    # A range read gives a truncated document, so use a pull parser rather than
    # ET.fromstring. <title>/<author> close inside the header; the first
    # <sentence> start tag carries the document_id and ends the header, so
    # reaching it means everything available has been read.
    parser = ET.XMLPullParser(events=("start", "end"))
    title: str | None = None
    author: str | None = None
    document_id: str | None = None
    try:
        parser.feed(header_bytes)
        for event, element in parser.read_events():
            tag = element.tag
            local_tag = tag.rsplit("}", 1)[-1] if isinstance(tag, str) else tag
            if event == "start":
                if local_tag == "sentence":
                    document_id = element.get("document_id") or None
                    break
                continue
            if local_tag == "title" and title is None:
                title = (" ".join(element.itertext()).strip()) or None
            elif local_tag == "author" and author is None:
                author = (" ".join(element.itertext()).strip()) or None
    except ET.ParseError:
        pass
    return title, author, document_id


def _parse_list_input(text: str) -> list[str]:
    parts: list[str] = []
    for line in (text or "").splitlines():
        parts.extend(item.strip() for item in line.split(","))
    urls = [item for item in parts if item]

    seen = set()
    deduped = []
    for item in urls:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _normalize_url(url: str) -> str:
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        return url

    path = quote(parts.path, safe="/%")
    query = quote(parts.query, safe="=&?/%")
    fragment = quote(parts.fragment, safe="%")
    return urlunsplit((parts.scheme, parts.netloc, path, query, fragment))


# Cached for the life of the process so a rerun never re-downloads a file.
# Failures raise rather than return None, so transient errors are not memoized.
@st.cache_data(show_spinner=False, max_entries=256)
def _fetch_url_bytes(url: str) -> bytes:
    source_url = _normalize_url(url)
    try:
        request = Request(source_url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
            return response.read()
    except (HTTPError, URLError, TimeoutError, ValueError):
        local_payload = _read_from_local_repo_if_available(source_url)
        if local_payload is not None:
            return local_payload
        raise


# Range request for the header bytes only; read() still caps at max_bytes if a
# server ignores Range. Cached apart from _fetch_url_bytes so header slices and
# full payloads never evict or shadow each other.
@st.cache_data(show_spinner=False, max_entries=256)
def _fetch_url_header_bytes(url: str, max_bytes: int = METADATA_HEADER_BYTES) -> bytes:
    source_url = _normalize_url(url)
    try:
        request = Request(
            source_url,
            headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes=0-{max_bytes - 1}"},
        )
        with urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
            return response.read(max_bytes)
    except (HTTPError, URLError, TimeoutError, ValueError):
        local_payload = _read_from_local_repo_if_available(source_url)
        if local_payload is not None:
            return local_payload[:max_bytes]
        raise


@st.cache_data(show_spinner=False)
def _fetch_xml_metadata(url: str) -> tuple[str | None, str | None, str | None]:
    return _extract_xml_metadata_from_header(_fetch_url_header_bytes(url))


def _prefetch_xml_metadata(urls: list[str]) -> dict[str, tuple[str | None, str | None, str | None]]:
    def fetch(url: str) -> tuple[str | None, str | None, str | None]:
        try:
            return _fetch_xml_metadata(url)
        except Exception:
            return None, None, None

    if not urls:
        return {}
    with ThreadPoolExecutor(max_workers=FETCH_MAX_WORKERS) as executor:
        return dict(zip(urls, executor.map(fetch, urls)))


def _warm_url_cache(urls: list[str]) -> None:
    def fetch(url: str) -> None:
        try:
            _fetch_url_bytes(url)
        except Exception:
            pass

    if not urls:
        return
    with ThreadPoolExecutor(max_workers=FETCH_MAX_WORKERS) as executor:
        list(executor.map(fetch, urls))


def _download_url_records_to_dir(records: list[dict], suffix_dir_name: str) -> tuple[Path | None, list[dict]]:
    if not records:
        return None, []

    target_dir = Path(tempfile.mkdtemp(prefix=f"didaskalos_{suffix_dir_name}_"))
    enriched_records: list[dict] = []
    failed_records: list[dict] = []

    _warm_url_cache([item["source_url"] for item in records])
    for item in records:
        try:
            payload = _fetch_url_bytes(item["source_url"])
        except Exception:
            failed_records.append(item)
            continue
        (target_dir / item["file"]).write_bytes(payload)
        title, author, document_id = _extract_xml_metadata(payload)
        enriched_records.append({**item, "title": title, "author": author, "document_id": document_id})

    if not enriched_records:
        return None, []

    return target_dir, enriched_records


@st.cache_data(show_spinner=False)
def _github_tree_paths() -> list[str]:
    # One recursive tree call shared by every discovery helper.
    request = Request(GITHUB_TREE_API, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return []

    tree_nodes = payload.get("tree") if isinstance(payload, dict) else None
    if not isinstance(tree_nodes, list):
        return []

    return [str(node.get("path", "")) for node in tree_nodes if node.get("type") == "blob"]


@st.cache_data(show_spinner=False)
def load_github_tree_urls(prefix: str) -> list[str]:
    urls = []
    for path in _github_tree_paths():
        if not path.startswith(prefix):
            continue
        if prefix == TREEBANK_PREFIX and not path.lower().endswith(".xml"):
            continue
        if prefix in LESSON_PREFIXES and not path.lower().endswith(".md"):
            continue
        urls.append(f"{GITHUB_RAW_BASE}/{path}")

    return sorted(urls)


@st.cache_data(show_spinner=False)
def load_treebank_registry() -> list[dict]:
    # [] when the manifest is absent, which falls back to the single-prefix scan.
    try:
        raw = _fetch_url_bytes(f"{GITHUB_RAW_BASE}/{TREEBANK_REGISTRY_PATH}")
    except Exception:
        return []
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
        return []
    corpora = payload.get("corpora") if isinstance(payload, dict) else None
    return [c for c in corpora if isinstance(c, dict)] if isinstance(corpora, list) else []


def _glob_suffix(file_glob: str) -> str:
    # Discovery only supports simple "*.ext" globs; return the ".ext" to match on.
    if file_glob and file_glob.startswith("*."):
        return file_glob[1:].lower()
    return ""


@st.cache_data(show_spinner=False)
def load_registered_treebank_urls() -> list[dict]:
    # One entry per treebank file, tagged with its corpus provenance and format.
    registry = load_treebank_registry()
    if not registry:
        return [
            {"url": url, "corpus_id": "perseus", "corpus_name": None,
             "format": "agdt-xml", "license": None, "author": None}
            for url in load_github_tree_urls(TREEBANK_PREFIX)
        ]

    paths = _github_tree_paths()
    entries: list[dict] = []
    for corpus in registry:
        prefix = corpus.get("path", "")
        if not prefix:
            continue
        suffix = _glob_suffix(corpus.get("file_glob", "*.xml"))
        for path in paths:
            if not path.startswith(prefix):
                continue
            if suffix and not path.lower().endswith(suffix):
                continue
            entries.append(
                {
                    "url": f"{GITHUB_RAW_BASE}/{path}",
                    "corpus_id": corpus.get("id"),
                    "corpus_name": corpus.get("name"),
                    "format": corpus.get("format", "agdt-xml"),
                    "license": corpus.get("license"),
                    "author": corpus.get("author"),
                }
            )
    return sorted(entries, key=lambda entry: entry["url"])


# Memoized on the URL set so a rerun reuses the records instead of re-running
# the parallel metadata prefetch.
@st.cache_data(show_spinner=False)
def _build_records_from_urls(urls: list[str], extract_xml_metadata: bool = False) -> list[dict]:
    used_names = set()
    records = []
    metadata_by_url = _prefetch_xml_metadata(urls) if extract_xml_metadata else {}
    for i, url in enumerate(urls, start=1):
        parsed = urlparse(url)
        file_name = Path(parsed.path).name or f"file_{i}"
        title, author, document_id = metadata_by_url.get(url, (None, None, None))
        records.append(
            {
                "file": _unique_name(file_name, used_names),
                "source_url": url,
                "title": title,
                "author": author,
                "document_id": document_id,
            }
        )
    return records


def _ensure_starter_lesson_urls(urls: list[str]) -> list[str]:
    required_urls = [f"{GITHUB_RAW_BASE}/{LESSON_PREFIX}{filename}" for filename in STARTER_LESSON_FILES]
    combined = list(urls or []) + required_urls

    seen = set()
    unique_urls = []
    for url in combined:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    return unique_urls


def _list_local_lesson_urls(prefix: str = LESSON_PREFIX) -> list[str]:
    lesson_dir = REPO_ROOT / prefix
    if not lesson_dir.exists() or not lesson_dir.is_dir():
        return []

    urls: list[str] = []
    for path in sorted(lesson_dir.glob("*.md")):
        # Keep URL format consistent with GitHub raw sources for downstream handling.
        urls.append(f"{GITHUB_RAW_BASE}/{prefix}{path.name}")
    return urls


def _dedupe_lesson_urls_by_filename(urls: list[str]) -> list[str]:
    # First URL wins per filename, so a localized lesson shadows the English one.
    seen_names: set[str] = set()
    deduped: list[str] = []
    for url in urls:
        name = Path(urlparse(url).path).name
        if name not in seen_names:
            seen_names.add(name)
            deduped.append(url)
    return deduped


def _resolve_default_lesson_urls(lang: str = DEFAULT_LANG) -> list[str]:
    # Localized folder first, so untranslated modules fall back to English.
    prefixes = [prefix for prefix in (LOCALIZED_LESSON_PREFIXES.get(lang), LESSON_PREFIX) if prefix]

    merged: list[str] = []
    for prefix in prefixes:
        # Remote and local merged so a GitHub API gap does not hide lessons.
        merged.extend(load_github_tree_urls(prefix))
        merged.extend(_list_local_lesson_urls(prefix))
    return _dedupe_lesson_urls_by_filename(_ensure_starter_lesson_urls(merged))


def _merge_treebank_entries(default_entries: list[dict], custom_urls: list[str]) -> list[dict]:
    # Registry defaults first, then user-pasted URLs. The latter have no
    # manifest, so their format is left None for the parser to auto-detect.
    seen = {entry["url"] for entry in default_entries}
    entries = list(default_entries)
    for url in custom_urls:
        if url in seen:
            continue
        seen.add(url)
        entries.append(
            {"url": url, "corpus_id": None, "corpus_name": None,
             "format": None, "license": None, "author": None}
        )
    return entries


def _build_treebank_records(entries: list[dict]) -> list[dict]:
    used_names = set()
    # Only XML files with no manifest author need a header read; CoNLL-U has no
    # header and relies on the manifest.
    urls_needing_meta = [
        entry["url"]
        for entry in entries
        if not entry.get("author")
        and entry.get("format") in (None, "agdt-xml")
        and entry["url"].lower().endswith(".xml")
    ]
    metadata_by_url = _prefetch_xml_metadata(urls_needing_meta) if urls_needing_meta else {}

    records = []
    for i, entry in enumerate(entries, start=1):
        url = entry["url"]
        parsed = urlparse(url)
        file_name = Path(parsed.path).name or f"file_{i}"
        meta_title, meta_author, meta_document_id = metadata_by_url.get(url, (None, None, None))
        records.append(
            {
                "file": _unique_name(file_name, used_names),
                "source_url": url,
                "title": meta_title or entry.get("corpus_name"),
                "author": meta_author or entry.get("author"),
                "document_id": meta_document_id,
                "corpus": entry.get("corpus_id"),
                "corpus_name": entry.get("corpus_name"),
                "format": entry.get("format"),
                "license": entry.get("license"),
            }
        )
    return records


def _build_records_from_uploads(uploaded_files) -> list[dict]:
    used_names = set()
    records = []
    for i, uploaded_file in enumerate(uploaded_files or []):
        file_bytes = uploaded_file.getvalue()
        title, author, document_id = (
            _extract_xml_metadata(file_bytes) if uploaded_file.name.lower().endswith(".xml") else (None, None, None)
        )
        records.append(
            {
                "file": _unique_name(uploaded_file.name, used_names),
                "upload_index": i,
                "source_url": "uploaded",
                "title": title,
                "author": author,
                "document_id": document_id,
                # Format left None so the dispatcher auto-detects.
                "corpus": None,
                "corpus_name": None,
                "format": None,
                "license": None,
            }
        )
    return records


def _materialize_uploaded_records(uploaded_files, selected_records: list[dict], suffix_dir_name: str) -> Path | None:
    if not uploaded_files or not selected_records:
        return None

    target_dir = Path(tempfile.mkdtemp(prefix=f"didaskalos_{suffix_dir_name}_"))
    for item in selected_records:
        uploaded_file = uploaded_files[item["upload_index"]]
        (target_dir / item["file"]).write_bytes(uploaded_file.getbuffer())
    return target_dir


def _build_treebank_display_table(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df
    # Clean author/work labels from the curated catalog, so the picker never
    # shows a raw TLG filename or source URL.
    resolved = [
        resolve_author_work(rec["file"], rec.get("author"), rec.get("title"), rec.get("document_id"))
        for rec in records
    ]
    df["display_author"] = [author for author, _ in resolved]
    df["display_work"] = [work for _, work in resolved]
    # The work key collapses a work split across many passage files into one
    # picker entry; texts with no TLG id fall back to their file name.
    df["work_key"] = [
        tlg_work_key(rec["file"], rec.get("document_id")) or rec["file"]
        for rec in records
    ]
    df["corpus_id"] = [rec.get("corpus") for rec in records]
    df["corpus_name"] = [rec.get("corpus_name") for rec in records]
    return df[["file", "display_author", "display_work", "work_key", "corpus_id", "corpus_name"]]


# st.fragment (stable in 1.37, experimental in 1.33) lets the treebank grid rerun
# on its own; an older Streamlit degrades to running the selector inline.
st_fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)
if st_fragment is None:
    def st_fragment(func):
        return func


def _tb_checkbox_key(item_id: str) -> str:
    return f"tb_cb_{item_id}"


def _set_treebank_selection(item_ids: list[str], value: bool) -> None:
    # Runs as a button on_click, before the checkboxes are re-instantiated, so
    # seeding their session values here is allowed.
    for item_id in item_ids:
        st.session_state[_tb_checkbox_key(item_id)] = value


def _aggregate_works(available_treebanks: pd.DataFrame, lang: str) -> dict[str, dict]:
    # Collapse the per-file table into one entry per whole work, keyed on
    # (corpus, work_key): item_id -> {author, work, corpus_id, corpus_name, files}.
    items: dict[str, dict] = {}
    for _, row in available_treebanks.iterrows():
        corpus_id = row.get("corpus_id")
        item_id = f"{corpus_id}|{row.get('work_key')}"
        item = items.get(item_id)
        if item is None:
            raw_author = row.get("display_author")
            author = str(raw_author).strip() if pd.notna(raw_author) else ""
            corpus_name = row.get("corpus_name")
            item = items[item_id] = {
                "author": author or t("unknown_author", lang),
                "work": str(row.get("display_work")),
                "corpus_id": corpus_id,
                "corpus_name": corpus_name if pd.notna(corpus_name) else None,
                "files": [],
            }
        item["files"].append(row["file"])
    return items


@st_fragment
def render_treebank_selector(available_treebanks: pd.DataFrame, lang: str) -> None:
    # A fragment, so ticking a checkbox reruns only this block and not the
    # sidebar's metadata prefetch. One flat checkbox per whole work; ticking it
    # selects all of that work's files, and the union goes to session state.
    st.subheader(t("available_treebanks_header", lang))
    st.caption(t("picker_hint", lang))

    items = _aggregate_works(available_treebanks, lang)
    # Author first, then work, so an author's works sit together in the list.
    ordered = sorted(
        items.items(),
        key=lambda kv: (kv[1]["author"].casefold(), kv[1]["work"].casefold()),
    )
    all_item_ids = [item_id for item_id, _ in ordered]

    btn_all, btn_clear = st.columns(2)
    btn_all.button(
        t("select_all_button", lang),
        key="tb_select_all_btn",
        use_container_width=True,
        on_click=_set_treebank_selection,
        args=(all_item_ids, True),
    )
    btn_clear.button(
        t("clear_selection_button", lang),
        key="tb_clear_btn",
        use_container_width=True,
        on_click=_set_treebank_selection,
        args=(all_item_ids, False),
    )

    for item_id, it in ordered:
        st.checkbox(f"{it['author']} — {it['work']}", key=_tb_checkbox_key(item_id))

    selected_files: list[str] = []
    for item_id, it in items.items():
        if st.session_state.get(_tb_checkbox_key(item_id), False):
            selected_files.extend(it["files"])
    st.session_state["selected_treebank_files"] = selected_files


def _render_sources_note(records: list[dict], lang: str) -> None:
    # Corpus attribution as a caption beneath the picker; CC BY-SA requires the
    # credit be shown somewhere.
    seen: set[tuple[str, str]] = set()
    parts: list[str] = []
    for rec in records:
        name = (rec.get("corpus_name") or "").strip()
        license_name = (rec.get("license") or "").strip()
        if not name and not license_name:
            continue
        key = (name, license_name)
        if key in seen:
            continue
        seen.add(key)
        if name and license_name:
            parts.append(f"{name} ({license_name})")
        else:
            parts.append(name or license_name)
    if parts:
        st.caption(t("sources_licenses_note", lang, sources="; ".join(parts)))


with st.sidebar:
    for variant, logo_path in LOGO_IMAGE_PATHS.items():
        if logo_path.exists():
            with st.container(key=LOGO_CONTAINER_KEYS[variant]):
                st.image(str(logo_path), use_container_width=True)

    st.selectbox(
        t("language_label", lang),
        options=AVAILABLE_LANGS,
        format_func=lambda code: LANG_NAMES[code],
        key="lang",
        on_change=_sync_lang_query_param,
    )

    st.header(t("sidebar_inputs", lang))

    # Stable option keys; only the displayed labels are translated.
    input_mode = st.radio(
        t("input_source_label", lang),
        options=["github", "upload"],
        index=0,
        format_func=lambda code: t(f"input_source_opt_{code}", lang),
        help=t("input_source_help", lang),
    )

    lesson_count = int(
        st.number_input(
            t("lesson_count_label", lang),
            min_value=1,
            max_value=200,
            value=40,
            step=1,
        )
    )

    syllabus_mode = st.radio(
        t("textbook_type_label", lang),
        options=["declension","case"],
        index=0,
        format_func=lambda code: t(f"textbook_type_opt_{code}", lang),
        help=t("textbook_type_help", lang),
    )

    if input_mode == "github":
        default_treebank_entries = load_registered_treebank_urls()
        default_lesson_urls = _resolve_default_lesson_urls(lang)

        with st.expander(t("custom_treebank_urls_expander", lang), expanded=False):
            custom_treebank_url_input = st.text_area(
                t("custom_treebank_urls_label", lang),
                value="",
                height=120,
                help=t("custom_treebank_urls_help", lang),
                key="custom_treebank_urls",
            )

        custom_treebank_urls = _parse_list_input(custom_treebank_url_input)
        treebank_entries = _merge_treebank_entries(default_treebank_entries, custom_treebank_urls)
        treebank_records = _build_treebank_records(treebank_entries)
        lesson_records = _build_records_from_urls(default_lesson_urls)
        uploaded_treebanks = []
    else:
        uploaded_treebanks = st.file_uploader(
            t("upload_label", lang),
            type=["xml", "conllu", "conll"],
            accept_multiple_files=True,
            help=t("upload_help", lang),
        )
        default_lesson_urls = _resolve_default_lesson_urls(lang)
        treebank_records = _build_records_from_uploads(uploaded_treebanks)
        lesson_records = _build_records_from_urls(default_lesson_urls)


available_treebanks = _build_treebank_display_table(treebank_records)
available_lessons = pd.DataFrame(lesson_records)

if available_treebanks.empty:
    st.warning(t("no_treebanks_warning", lang))
    st.stop()

render_treebank_selector(available_treebanks, lang)
_render_sources_note(treebank_records, lang)
selected_treebank_files = st.session_state.get("selected_treebank_files", [])

if available_lessons.empty:
    st.warning(t("no_lessons_warning", lang))
    st.stop()

selected_lesson_files = available_lessons["file"].tolist()

build_clicked = st.button(t("build_button", lang), type="primary", use_container_width=True)

if build_clicked:
    if not selected_treebank_files:
        st.warning(t("select_at_least_one_warning", lang))
        st.stop()

    selected_treebank_records = [row for row in treebank_records if row["file"] in selected_treebank_files]
    selected_lesson_records = [row for row in lesson_records if row["file"] in selected_lesson_files]

    with st.spinner(t("spinner_preparing", lang)):
        if input_mode == "github":
            treebank_dir, selected_treebank_records = _download_url_records_to_dir(selected_treebank_records, "treebanks")
        else:
            treebank_dir = _materialize_uploaded_records(uploaded_treebanks, selected_treebank_records, "treebanks")

        lesson_dir, selected_lesson_records = _download_url_records_to_dir(selected_lesson_records, "lessons")

        if treebank_dir is None:
            st.error(t("error_prepare_treebanks", lang))
            st.stop()
        if lesson_dir is None:
            st.error(t("error_prepare_lessons", lang))
            st.stop()

    # Declared format per file; None (custom URLs, uploads) auto-detects.
    treebank_formats = {
        row["file"]: row.get("format")
        for row in selected_treebank_records
        if row["file"] in selected_treebank_files
    }

    with st.spinner(t("spinner_parsing", lang)):
        combined_df = build_combined_df(
            treebank_dir,
            selected_treebank_files,
            syllabus_mode=syllabus_mode,
            formats=treebank_formats,
        )
        frequency_syllabus = build_frequency_syllabus(combined_df)

        # Front-matter "About This Textbook" data. Works are deduped by TLG work
        # key; corpora are enriched from the registry for their source links.
        corpus_meta = {c["id"]: c for c in load_treebank_registry() if c.get("id")}
        summary_works: list[tuple[str | None, str]] = []
        seen_work_keys: set[str] = set()
        summary_corpora: list[dict] = []
        seen_corpus_ids: set[str] = set()
        has_custom_sources = False
        for rec in selected_treebank_records:
            work_key = tlg_work_key(rec["file"], rec.get("document_id")) or rec["file"]
            if work_key not in seen_work_keys:
                seen_work_keys.add(work_key)
                summary_works.append(
                    resolve_author_work(
                        rec["file"], rec.get("author"), rec.get("title"), rec.get("document_id")
                    )
                )
            corpus_id = rec.get("corpus")
            if not corpus_id:
                has_custom_sources = True
                continue
            if corpus_id in seen_corpus_ids:
                continue
            seen_corpus_ids.add(corpus_id)
            meta = corpus_meta.get(corpus_id, {})
            summary_corpora.append(
                {
                    "id": corpus_id,
                    "name": meta.get("name") or rec.get("corpus_name"),
                    "license": meta.get("license") or rec.get("license"),
                    "source_url": meta.get("source_url"),
                }
            )
        summary_works.sort(key=lambda aw: ((aw[0] or "").lower(), (aw[1] or "").lower()))
        source_summary = {
            "works": summary_works,
            "corpora": summary_corpora,
            "token_count": int(len(combined_df)),
            "work_count": len(summary_works),
            "has_custom_sources": has_custom_sources,
        }

        textbook_markdown = generate_textbook_markdown(
            frequency_syllabus=frequency_syllabus,
            grammar_folder=lesson_dir,
            lesson_count=lesson_count,
            combined_df=combined_df,
            syllabus_mode=syllabus_mode,
            lang=lang,
            source_summary=source_summary,
        )
        textbook_html = generate_textbook_html(
            frequency_syllabus=frequency_syllabus,
            grammar_folder=lesson_dir,
            lesson_count=lesson_count,
            combined_df=combined_df,
            syllabus_mode=syllabus_mode,
            lang=lang,
            markdown_content=textbook_markdown,
            source_summary=source_summary,
            logo_data_uri=textbook_logo_data_uri or None,
        )

    # is_deponent is an internal flag driving the deponent lesson, not treebank
    # data, so it stays out of the export. Selecting columns in to_csv avoids
    # copying the frame just to drop one column.
    combined_csv_columns = [column for column in combined_df.columns if column != "is_deponent"]

    # Kept in session state so later reruns do not rebuild. CSV bytes are made
    # once here rather than holding the full token frame per session.
    st.session_state["build_result"] = {
        "treebank_count": len(selected_treebank_files),
        "token_rows": int(len(combined_df)),
        "frequency_rows": int(len(frequency_syllabus)),
        "frequency_syllabus": frequency_syllabus,
        "frequency_csv": frequency_syllabus.to_csv(index=False).encode("utf-8"),
        "combined_csv": combined_df.to_csv(index=False, columns=combined_csv_columns).encode("utf-8"),
        "textbook_markdown": textbook_markdown,
        "textbook_html": textbook_html,
    }

build_result = st.session_state.get("build_result")
if build_result:
    c1, c2, c3 = st.columns(3)
    c1.metric(t("metric_selected_treebanks", lang), build_result["treebank_count"])
    c2.metric(t("metric_token_rows", lang), build_result["token_rows"])
    c3.metric(t("metric_frequency_rows", lang), build_result["frequency_rows"])

    st.subheader(t("frequency_syllabus_header", lang))
    st.dataframe(build_result["frequency_syllabus"], use_container_width=True, height=420)

    st.download_button(
        label=t("download_frequency_syllabus", lang),
        data=build_result["frequency_csv"],
        file_name="frequency_syllabus.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.download_button(
        label=t("download_combined_rows", lang),
        data=build_result["combined_csv"],
        file_name="combined_treebank_rows.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.download_button(
        label=t("download_textbook_md", lang),
        data=build_result["textbook_markdown"].encode("utf-8"),
        file_name="textbook.md",
        mime="text/markdown",
        use_container_width=True,
    )

    st.download_button(
        label=t("download_textbook_html", lang),
        data=build_result["textbook_html"].encode("utf-8"),
        file_name="textbook.html",
        mime="text/html",
        use_container_width=True,
    )

    st.subheader(t("textbook_md_preview_header", lang))
    st.code(build_result["textbook_markdown"][:6000], language="markdown")

    st.subheader(t("textbook_html_preview_header", lang))
    # The HTML can be several MB, so it is only sent when asked for.
    if st.toggle(t("show_html_preview_label", lang), value=False, key="show_html_preview"):
        components.html(build_result["textbook_html"], height=800, scrolling=True)

st.markdown("---")
st.caption(t("footer_caption", lang))
st.caption(t("image_credit", lang))
st.caption(t("license_caption", lang, year=date.today().year))
