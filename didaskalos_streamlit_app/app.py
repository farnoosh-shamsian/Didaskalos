from pathlib import Path
import json
import tempfile
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from didaskalos_pipeline import (
    build_combined_df,
    build_frequency_syllabus,
    generate_textbook_html,
    generate_textbook_markdown,
)


st.set_page_config(
    page_title="Didaskalos",
    page_icon="DB",
    layout="wide",
)

st.title("Didaskalos: Frequency-Based Greek Grammar Builder")
st.caption("Treebank input via URL(s) or upload")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GRAMMAR_DIR = PROJECT_ROOT / "lessons-no-decl"
GITHUB_TREE_API = "https://github.com/farnoosh-shamsian/Didaskalos/tree/main/treebanks/perseus?raw=true"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/farnoosh-shamsian/Didaskalos/main"
GITHUB_TREEBANK_PREFIX = "treebanks/perseus/"


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


def _parse_url_input(url_text: str) -> list[str]:
    raw_parts = []
    for line in (url_text or "").splitlines():
        raw_parts.extend([part.strip() for part in line.split(",")])

    urls = [part for part in raw_parts if part]

    seen = set()
    deduped = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped


def _build_url_records(urls: list[str]) -> list[dict]:
    used_names = set()
    records = []

    for i, url in enumerate(urls, start=1):
        parsed = urlparse(url)
        file_name = Path(parsed.path).name or f"treebank_{i}.xml"
        records.append(
            {
                "file": _unique_name(file_name, used_names),
                "source_url": url,
            }
        )

    return records


def _build_upload_records(uploaded_files) -> list[dict]:
    used_names = set()
    records = []

    for i, uploaded_file in enumerate(uploaded_files or []):
        records.append(
            {
                "file": _unique_name(uploaded_file.name, used_names),
                "upload_index": i,
                "source_url": "uploaded",
            }
        )

    return records


def _download_url_records_to_dir(records: list[dict], suffix_dir_name: str) -> Path | None:
    if not records:
        return None

    target_dir = Path(tempfile.mkdtemp(prefix=f"didaskalos_{suffix_dir_name}_"))
    for item in records:
        request = Request(item["source_url"], headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request) as response:
            (target_dir / item["file"]).write_bytes(response.read())
    return target_dir


def _materialize_uploaded_records(uploaded_files, records: list[dict], suffix_dir_name: str) -> Path | None:
    if not uploaded_files or not records:
        return None

    target_dir = Path(tempfile.mkdtemp(prefix=f"didaskalos_{suffix_dir_name}_"))
    for item in records:
        uploaded_file = uploaded_files[item["upload_index"]]
        (target_dir / item["file"]).write_bytes(uploaded_file.getbuffer())
    return target_dir


@st.cache_data(show_spinner=False)
def load_default_github_treebank_urls(api_url: str = GITHUB_TREE_API):
    request = Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return []

    tree_nodes = payload.get("tree") if isinstance(payload, dict) else None
    if not isinstance(tree_nodes, list):
        return []

    urls = []
    for node in tree_nodes:
        path = str(node.get("path", ""))
        if node.get("type") != "blob":
            continue
        if not path.startswith(GITHUB_TREEBANK_PREFIX) or not path.lower().endswith(".xml"):
            continue
        urls.append(f"{GITHUB_RAW_BASE}/{path}")

    return sorted(urls)


with st.sidebar:
    st.header("Inputs")

    input_mode = st.radio(
        "Treebank source",
        options=["Provide treebank URL(s)", "Upload treebank file(s)"],
        index=0,
        help="Choose either URL input or file upload.",
    )

    uploaded_treebanks = []
    url_records = []

    if input_mode == "Provide treebank URL(s)":
        default_urls = load_default_github_treebank_urls()
        url_input = st.text_area(
            "Treebank XML URL(s)",
            value="\n".join(default_urls),
            height=220,
            help="One URL per line (or comma-separated). Defaults come from your GitHub repo treebanks/perseus.",
        )
        url_records = _build_url_records(_parse_url_input(url_input))
    else:
        uploaded_treebanks = st.file_uploader(
            "Upload treebank XML files",
            type=["xml"],
            accept_multiple_files=True,
            help="Upload one or more XML files.",
        )

    uploaded_lessons = st.file_uploader(
        "Upload lesson markdown files (optional)",
        type=["md"],
        accept_multiple_files=True,
        help="Optional lesson modules used in textbook generation.",
    )

    repo_grammar_dir = st.text_input(
        "Lesson module folder in app repo",
        value=str(DEFAULT_GRAMMAR_DIR),
        help="Used only when lesson markdown files are not uploaded.",
    )

    lesson_count = int(
        st.number_input(
            "Lesson count",
            min_value=1,
            max_value=200,
            value=20,
            step=1,
        )
    )


if input_mode == "Provide treebank URL(s)":
    available_records = url_records
else:
    available_records = _build_upload_records(uploaded_treebanks)

available_treebanks = pd.DataFrame(available_records)
if available_treebanks.empty:
    st.warning("No treebanks available. Provide URL(s) or upload XML files.")
    st.stop()

st.subheader("Available Treebanks")
st.dataframe(available_treebanks[["file", "source_url"]], use_container_width=True, height=260)

selected_files = st.multiselect(
    "Select treebank files",
    options=available_treebanks["file"].tolist(),
    default=available_treebanks["file"].tolist(),
)

build_clicked = st.button("Build Syllabus", type="primary", use_container_width=True)

if build_clicked:
    if not selected_files:
        st.warning("Select at least one treebank file before building.")
        st.stop()

    selected_records = [row for row in available_records if row["file"] in selected_files]

    with st.spinner("Preparing selected treebanks..."):
        if input_mode == "Provide treebank URL(s)":
            treebank_dir = _download_url_records_to_dir(selected_records, "url_treebanks")
        else:
            treebank_dir = _materialize_uploaded_records(uploaded_treebanks, selected_records, "uploaded_treebanks")

        if treebank_dir is None:
            st.error("Could not prepare selected treebank files.")
            st.stop()

        lesson_upload_dir = None
        if uploaded_lessons:
            lesson_upload_dir = Path(tempfile.mkdtemp(prefix="didaskalos_lessons_"))
            for lesson_file in uploaded_lessons:
                (lesson_upload_dir / lesson_file.name).write_bytes(lesson_file.getbuffer())

        grammar_dir = lesson_upload_dir or Path(repo_grammar_dir)

    with st.spinner("Parsing selected treebanks and building frequency table..."):
        combined_df = build_combined_df(treebank_dir, selected_files)
        frequency_syllabus = build_frequency_syllabus(combined_df)
        textbook_markdown = generate_textbook_markdown(
            frequency_syllabus=frequency_syllabus,
            grammar_folder=grammar_dir,
            lesson_count=lesson_count,
        )
        textbook_html = generate_textbook_html(
            frequency_syllabus=frequency_syllabus,
            grammar_folder=grammar_dir,
            lesson_count=lesson_count,
        )

    c1, c2, c3 = st.columns(3)
    c1.metric("Selected files", len(selected_files))
    c2.metric("Token rows", int(len(combined_df)))
    c3.metric("Frequency rows", int(len(frequency_syllabus)))

    st.subheader("Frequency Syllabus")
    st.dataframe(frequency_syllabus, use_container_width=True, height=420)

    st.subheader("Combined Token Data")
    st.dataframe(combined_df.head(250), use_container_width=True, height=320)

    st.download_button(
        label="Download frequency_syllabus.csv",
        data=frequency_syllabus.to_csv(index=False).encode("utf-8"),
        file_name="frequency_syllabus.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.download_button(
        label="Download combined_treebank_rows.csv",
        data=combined_df.to_csv(index=False).encode("utf-8"),
        file_name="combined_treebank_rows.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.download_button(
        label="Download textbook.md",
        data=textbook_markdown.encode("utf-8"),
        file_name="textbook.md",
        mime="text/markdown",
        use_container_width=True,
    )

    st.download_button(
        label="Download textbook.html",
        data=textbook_html.encode("utf-8"),
        file_name="textbook.html",
        mime="text/html",
        use_container_width=True,
    )

    st.subheader("Textbook Markdown Preview")
    st.code(textbook_markdown[:6000], language="markdown")

    st.subheader("Textbook HTML Preview")
    components.html(textbook_html, height=800, scrolling=True)

st.markdown("---")
st.caption(
    "Didaskalos is an open-source project. Contributions and feedback are welcome! "
    "Visit the GitHub repository for more information."
)
