# Didaskalos

Didaskalos (διδάσκαλος, *teacher*) is a corpus-driven pipeline that reads an annotated Ancient
Greek corpus and writes a textbook for it. Grammar topics are ordered by what the selected texts
actually use rather than by a fixed curriculum, and the exercises are authentic sentences from that
corpus, each carrying its citation.

- **Live app:** <https://didaskalos-app-11551311398.europe-west1.run.app/>
- **Project site:** <https://farnoosh-shamsian.github.io/didaskalos/> (English and Persian)

## Why

Ancient Greek pedagogy is mostly limited to a few languages. For most of the world's speakers,
learning Greek therefore means going through a second foreign language first, or translating a
textbook that was written for a different audience — which can turn the translation into a rewrite,
or complicate the learning experience. Didaskalos is built the other way round: the pedagogy lives
in code, the teaching language is data, and the Greek is never simplified to make either easier. A
textbook in a new language is a translation job rather than a rewrite — one locale file and a
translated set of lesson modules, with no change to the pipeline.

English and Persian are available today. Persian comes first because it is the language I work in
and care about most, and it is where the project started (see
[Where this started](#where-this-started)), but nothing in the design is specific to it. Didaskalos
is worth most exactly where pedagogical resources for learning Greek are lacking or limited, and
that is a shared problem involving many other low-resource languages.

Reaching more languages is not all it does. Even in English this is not the conventional method:
there is no fixed curriculum and no simplified sentences, the syllabus is ordered by what the chosen
texts actually use, and every exercise is a real sentence carrying its citation.

## What is in it

| | |
| --- | --- |
| Works built in | 85 |
| Authors and collections | 25 |
| Treebank corpora | 4 (Perseus, Gorman, Harrington, PROIEL New Testament) |
| Lesson modules | 157 per language |
| Interface and textbook languages | English, Persian |

Users can also paste treebank URLs or upload their own XML.

## How the pipeline works

1. **Parse.** Annotated treebanks are read by pluggable per-format adapters — Perseus/AGDT XML and
   CoNLL-U (Universal Dependencies, PROIEL) — which normalize every token's morphology into one
   shared 9-character postag.
2. **Rank.** Every token is counted by the grammatical feature it exhibits, turning the corpus into
   a ranked list of topics: which case, which declension class, which tense–mood–voice combination
   this author actually leans on.
3. **Order.** The ranking is then constrained so that it teaches (see below).
4. **Assemble.** Each ranked topic pulls in its lesson module, its paradigm, and its exercises, and
   the book is exported as Markdown, HTML, or CSV — with a colophon naming every source corpus, its
   license, and its URL.

Thucydides, *The Peloponnesian War*, for example, yields 31,924 analyzed tokens and 119 ranked
grammar topics.

Two textbook types are available. The case-based textbook explains nouns and adjectives case by
case; the declension-based one first classifies every noun and adjective into declension classes
and orders those lessons by how frequent each class is in the selected corpus.

## Ordering the syllabus

A purely statistical syllabus might be a flawed teacher, so the frequency count is overruled in
three places:

- **A fixed opening.** The alphabet and an orientation to nouns, adjectives, and verbs come first
  and map the system before the counts take over — otherwise the most frequent form in a corpus,
  which can be an irregular or an advanced one, would open the course.
- **Declared prerequisites.** `LESSON_PREREQUISITE_KINDS` in `didaskalos_pipeline.py` makes a
  lesson wait for one of the kind it contrasts with, whenever such a lesson is in the same
  syllabus: an irregular verb, noun, or adjective class follows a regular one of the same part of
  speech, and every adjective class follows a noun class. The middle voice is always followed by
  deponent verbs.
- **Merged labels.** `MERGED_SYLLABUS_LABELS` folds a syllabus row into another lesson where a
  separate module would only repeat it — the vocative is taught inside the nominative.

After that, corpus frequency arranges everything else.

## Exercises and difficulty scoring

Exercises are generated exclusively from authentic sentences in the corpus; nothing is rewritten or
simplified. Each sentence is scored on three factors — the mean rarity of its words, the rarity of
its single rarest word, and its length — and selection also prefers sentences whose vocabulary
earlier lessons have already introduced. Difficulty therefore increases slowly across the book,
and every sentence stays exactly as its author wrote it.

## Lesson modules and localization

Lesson modules are hand-written Markdown, one folder per language (`lessons/en`, `lessons/fa`),
with parallel filenames so a translated file shadows its English counterpart. Initial drafts came
from a Retrieval-Augmented Generation pipeline over standard reference grammars (Smyth,
Crosby & Schaeffer); that pipeline has been retired and is kept for reference in
[`archive/ragbot/`](archive/ragbot/), and the modules are now hand-corrected and still under review.

Being translatable is a design goal, not an afterthought — Didaskalos was bilingual from the start.
Translating the interface alone would not solve the accessibility problem, so localization is
end-to-end rather than interface-deep: the UI, the lesson modules, the grammatical terminology
(kept consistent by a curated English–Persian table), and the exported textbook are all translated.
One rule makes the modules translatable at all: no lesson explains Greek by comparison with the
learner's first language.

### Adding a language

Everything language-specific lives in two places:

1. `didaskalos_streamlit_app/locales/<lang>.json` — one locale file, registered in `AVAILABLE_LANGS`
   in `i18n.py` (add the code to `RTL_LANGS` if the script runs right to left).
2. `lessons/<lang>/` — the lesson modules, using the same filenames as `lessons/en/` so the two
   tables of contents stay line-for-line parallel.

Translate those and Didaskalos teaches Greek in that language; the pipeline itself does not change.
If you would like to do this for your language, I would be glad to help you through it — get in
touch via <https://farnoosh-shamsian.github.io/pages/contact.html>.

## Implementation

The project is implemented in Python (Streamlit app, deployed on Cloud Run) and integrates treebank
processing, frequency-based linguistic analysis, and modular content generation and assembly.

Treebank collections are declared in `treebanks/registry.json` and parsed by pluggable, per-format
adapters (currently Perseus/AGDT XML and CoNLL-U / Universal Dependencies). Adding a new corpus is a
matter of dropping files into a folder and adding one manifest entry — see
[Adding a treebank collection](didaskalos_streamlit_app/README.md#adding-a-treebank-collection).

## Where this started

Didaskalos began as an answer to one language's version of that problem. For readers in Iran and
across West and Central Asia, and in parts of Africa, Ancient Greek texts are primary sources for
their own past: much of the surviving narrative of ancient Persia reaches us only in Greek. Yet
Ancient Greek is not taught regularly at any institution in Iran, and there is no functional
grammar, reader, or textbook written for Persian speakers.

So those texts are read at second hand, and every mediating edition adds its own interpretive
layer: Xenophon's *Cyropaedia* exists in at least eight Persian translations, not one of them made
from the Greek. That is the problem a tool like Didaskalos is trying to solve, by helping a larger
audience read the sources directly.

## Known limitations

Other than reviewing the lesson modules, the core infrastructure is also being modified and
improved. These are some of the issues I am working through:

- The syllabus is driven by morphology almost alone; syntax is barely used, though the treebanks
  carry it. I am looking for a way to use it without overcomplicating the system.
- The pedagogical approach needs further refinement, particularly in how it integrates with the
  syntax. I am working on integrating Didakta into the system.
- Rare paradigms in the long tail have no lesson file yet, so a very large build can still report a
  missing module.
- Source treebanks sometimes disagree with each other, or carry undecodable and misaligned tags.
- Whether the automatically selected exercises are genuinely useful still needs systematic
  evaluation.

## License

Copyright © 2026 Farnoosh Shamsian.

Didaskalos is released under the
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/)
(CC BY-NC-SA 4.0). You may share and adapt the material for non-commercial purposes,
provided you give attribution and license your adaptations under the same terms.
See [LICENSE](LICENSE) for the full text.

**Scope.** This license covers the original work of this project: the pipeline and
application code, the grammar and lesson modules, the syllabus-generation method, and
the Persian localization. It does *not* relicense the source texts.

**Source texts.** Each treebank under `treebanks/` retains the license granted by its
own authors, as declared in [`treebanks/registry.json`](treebanks/registry.json) and
shown in the app:

| Corpus | Upstream license |
| --- | --- |
| Perseus Ancient Greek Dependency Treebank | CC BY-SA 3.0 |
| Gorman Ancient Greek Dependency Trees | CC BY-NC-SA 4.0 |
| Harrington Treebanked Commentaries | MIT |
| PROIEL New Testament | CC BY-NC-SA 4.0 |

Note that Perseus is CC BY-SA 3.0, which does **not** permit a NonCommercial
restriction to be added to that material; its ShareAlike terms continue to govern the
Perseus data itself. Because Gorman and PROIEL are ShareAlike, a generated textbook
containing their sentences must itself be distributed under CC BY-NC-SA 4.0 — the
exported textbook states this in its "About This Textbook" section.
