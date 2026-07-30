# Archive

Retired work kept for provenance. **Nothing here is imported, read, or fetched by
the Didaskalos app.** Paths and dependencies inside these files are stale — treat
them as historical records, not as runnable code.

Archived 2026-07-30.

## `ragbot/` — the RAG pipeline that drafted the first grammar modules

`smyth-rag.ipynb` is a LlamaIndex + Groq pipeline that read Smyth's *Greek Grammar*
and Crosby–Schaeffer, then generated the first drafts of the lesson modules. Those
drafts have since been rewritten by hand, so the pipeline is superseded.

| File | What it is |
|---|---|
| `smyth-rag.ipynb` | The pipeline itself: chunking, embedding, retrieval, lesson generation. |
| `lessons/` | The 48 generated first drafts (`1st_decl_αη-stem.md` and friends). Superseded by `lessons/en/`. |
| `frequency_syllabus.csv` | `syllabus,pos_category,frequency` — the topic order fed to the generator. |
| `Didaskalos-frequency-syllabus.py` | Standalone script that produced that CSV. |
| `.progress.json` | The generator's bookkeeping, moved here from `lessons-no-decl/`. Its keys use Greek letters (`aorist_indicative_active_ω`) and never matched the `_w` lesson filenames. |
| `.env` | Holds a `GROQ_API_KEY`. Gitignored. **Rotate or delete it** — it is not needed by anything. |

### Stale paths in `smyth-rag.ipynb`

The notebook writes to `OUTPUT_DIR = os.path.join("..", "lessons-no-decl")` with a
`PROGRESS_FILE` beside it. That folder no longer exists; the lessons now live in
`lessons/en/` and `lessons/fa/`. Both paths would need updating before the notebook
could run again.

### What was deleted rather than archived

Two directories, ~112 MB, both regenerable:

- `embedding_model/` — a HuggingFace cache of `sentence-transformers/all-MiniLM-L6-v2`
  (snapshot `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`). Re-downloads automatically
  the first time the notebook instantiates the embedding model.
- `vector-database/` — the LlamaIndex persisted store (`docstore.json`,
  `default__vector_store.json`, `index_store.json`). Rebuilt by re-running the
  indexing cells of `smyth-rag.ipynb`.

Rebuilding also needs the Smyth and Crosby–Schaeffer source texts, which were never
in this repo — see `ragbot/README.md` for where they came from.

## `notebooks/`

| File | What it is |
|---|---|
| `Didaskalos-sandbox.ipynb` | The original monolithic prototype of the whole pipeline: frequency syllabus, lesson assembly, textbook generation. Refactored into `didaskalos_streamlit_app/didaskalos_pipeline.py` (`build_frequency_syllabus`, `generate_textbook_markdown`). Its `GRAMMAR_FOLDER = Path("lessons-no-decl")` is stale. |
| `deponent-quary.ipynb` | A one-off query notebook used while working out how to handle deponent verbs. Reads `../treebanks/perseus`, which still exists. |
