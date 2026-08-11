# Didaskalos Streamlit App

This folder contains the Streamlit web app for Didaskalos. I'm just testing out things at this point and a lot more is coming.

## What the app does

- Loads treebanks from the GitHub repository by default
- Loads lesson modules from the same GitHub repository by default
- Lets users paste one or more treebank URLs if they want to override the defaults
- Lets users upload XML treebanks and markdown lesson modules instead of using GitHub
- Builds a combined token dataframe
- Computes a frequency-based syllabus table
- Exports CSV, markdown, and HTML downloads

## GitHub sources used by default

- Treebank collections: listed in [treebanks/registry.json](https://github.com/farnoosh-shamsian/didaskalos/blob/main/treebanks/registry.json) (Perseus by default)
- Lesson modules: [lessons/en](https://github.com/farnoosh-shamsian/didaskalos/tree/main/lessons/en) (Persian: [lessons/fa](https://github.com/farnoosh-shamsian/didaskalos/tree/main/lessons/fa))
- App folder: [didaskalos_streamlit_app](https://github.com/farnoosh-shamsian/didaskalos/tree/main/didaskalos_streamlit_app)

## Adding a treebank collection

Treebanks are discovered from a manifest, `treebanks/registry.json`, so adding a new
collection needs no pipeline changes. Two supported formats:

- `agdt-xml` — Perseus / Ancient Greek Dependency Treebank XML (`<sentence>`/`<word>` with
  9-character postags). This is the format under `treebanks/perseus/`.
- `conllu` — CoNLL-U (Universal Dependencies / PROIEL). The parser normalizes UD morphology
  into the same 9-character postag the rest of the pipeline reads, preferring the original
  AGDT tag when the source carries it in the XPOS column.

To add a collection:

1. Put the files in a new folder, e.g. `treebanks/gorman/` (AGDT XML) or `treebanks/proiel/`
   (`*.conllu`).
2. Add an entry to `treebanks/registry.json`:

   ```json
   {
     "id": "gorman",
     "name": "Vanessa Gorman Treebanks",
     "path": "treebanks/gorman/",
     "format": "agdt-xml",
     "file_glob": "*.xml",
     "language": "grc",
     "license": "CC BY 4.0",
     "source_url": "https://github.com/vgorman1/Greek-Dependency-Trees"
   }
   ```

   Use `"format": "conllu"` and `"file_glob": "*.conllu"` for a CoNLL-U corpus. `name`,
   `author`, and `license` are shown in the treebank selector (and are the metadata source for
   CoNLL-U, which has no XML header).

   Check the `license` before adding a corpus. Didaskalos is distributed under CC BY-NC-SA 4.0,
   and generated textbooks quote corpus sentences directly, so a corpus whose terms forbid
   redistribution, or whose ShareAlike terms cannot coexist with NonCommercial (a bare
   `CC BY-SA` corpus, for instance), creates an obligation that the project cannot satisfy for
   the exported textbook. Public-domain, CC BY, CC BY-NC-SA, and permissive software licenses
   are safe. Record the license verbatim — it is reproduced in the exported textbook's
   "About This Textbook" section.

3. Check that the files parse before committing:

   ```
   py -3 validate_treebank.py path/to/file.xml
   py -3 validate_treebank.py path/to/file.conllu --format conllu
   ```

   A healthy result shows non-zero sentences/tokens and a low "Undecodable postag" count.

4. Commit and push to `main`. Note: `.gcloudignore` excludes `treebanks/` from the Cloud Build
   upload, so the deployed app reads corpora from GitHub raw — a new collection goes live only
   once it is pushed to `main`. The local folder is used only when running the app locally.

A new *format* (beyond `agdt-xml`/`conllu`) is added once by writing an adapter in
`treebank_parsers.py` and registering it in `PARSERS`; every adapter must emit the same token
schema and normalize morphology into the 9-character postag.

## Writing a lesson module

Lesson files live in one folder per language: `lessons/en/` and `lessons/fa/`. Each folder holds
both the case modules and the declension-class modules — the two sets share no filenames, so the
syllabus mode selects which of them the textbook uses, not which folder is read.
The pipeline takes each file's leading heading as the lesson's display title, so that heading is
what a reader sees in the table of contents next to every other lesson:

- One `#` heading at the top (after any YAML frontmatter), and nothing above it.
- Sections within the lesson start at `##`. The textbook drops the file's own top heading and
  re-emits the title as the lesson's `#`, so the body's `##` sections nest under it and the
  lesson title stays the largest heading on the page.
- A plain noun phrase: `# The Aorist Indicative Active (ω-Verbs)`, `# Adverbs`. No `Lesson:`
  prefix and no bold or italic markup — the textbook already numbers each entry and labels it a
  module. (`normalize_lesson_title` strips these anyway, but the file should read correctly on
  its own.)
- Paradigm lessons follow their filename: tense, then mood, then voice, with the verb bucket in
  parentheses — `(ω-Verbs)`, `(μι-Verbs)`, `(Irregular Verbs)`; the bucket-less fallback file
  drops the parenthesis. Infinitives and participles read tense-voice-mood
  (`# The Aorist Middle Infinitive (ω-Verbs)`), which is how Greek grammars name them.
- A title must stand on its own. Lessons are ordered by corpus frequency, so any lesson can turn
  up first: a title like "Other Adjectives" would arrive before the learner has met any
  adjectives. Name what the lesson contains instead. Where the *order* also matters
  pedagogically, give the lesson a kind in `lesson_kinds` and a rule in
  `LESSON_PREREQUISITE_KINDS` in `didaskalos_pipeline.py`, and it will be placed after one lesson
  of the kind it contrasts with whenever such a lesson is in the same syllabus.
- No level, audience, or prerequisite line under the heading (`*Target level: …*`,
  `*Prerequisites: …*`, `**Target:** …`). The textbook prints the part of speech and the corpus
  frequency there, and where a lesson sits in the sequence is decided by
  `LESSON_PREREQUISITE_KINDS`, not by prose in the file. A subtitle naming the lesson's example
  verbs — `*(e.g., δίδωμι, τίθημι, ἵημι)*` — or a `**Focus:**` line describing what the lesson
  covers is fine.

`lessons/fa/` uses the same filenames as `lessons/en/` and the same title shape in the target
language, so the two tables of contents stay line-for-line parallel. Keeping the languages in
separate folders is what lets a translated file shadow its English counterpart by name.

Some lessons teach a concept rather than a paradigm slot. `deponent_verbs.md` and
`irregular_verbs.md` each collect their tokens from across the whole corpus — every middle-only
lemma, every irregular verb — instead of from one tense/mood/voice row, and appear once in the
syllabus. Add one by giving it a label and filename constant, a branch in
`get_topic_rows_for_label` that selects its tokens, and (if it needs to follow something) a kind in
`lesson_kinds` and a rule in `LESSON_PREREQUISITE_KINDS`.

Two syllabus rows can also share a single lesson: `MERGED_SYLLABUS_LABELS` folds one label into
another, adding its token count to the host lesson and its tokens to that lesson's exercise pool,
while leaving the token-level `syllabus` value untouched for answer keys and the CSV export. The
nominative and vocative are merged this way — the vocative repeats the nominative except in a few
singular endings, so `nominative.md` teaches both and there is no `vocative.md`.

## Project layout

- `app.py`: Streamlit UI and source discovery
- `didaskalos_pipeline.py`: reusable data and export functions
- `treebank_parsers.py`: pluggable per-format parser adapters + dispatcher
- `validate_treebank.py`: CLI to check a treebank file parses correctly
- `requirements.txt`: Python dependencies
- `.streamlit/config.toml`: Streamlit runtime and theme config
