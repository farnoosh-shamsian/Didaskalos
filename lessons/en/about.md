# About Didaskalos

Didaskalos is a corpus-driven system designed to generate fully personalized textbooks for learning Ancient Greek. Instead of relying on fixed curricula or generalized pedagogical sequences, Didaskalos builds each textbook directly from the texts a learner or instructor actually wants to read. This ensures that every lesson, explanation, and exercise is immediately relevant to the learner’s goals and grounded in authentic material.

At its core, Didaskalos operates by integrating three key resources: a corpus of selected Ancient Greek texts, linguistic treebanks, and a modular grammar framework. The process begins when a user chooses a target corpus. The system then analyzes this corpus using treebank data, identifying vocabulary frequency, morphological patterns, and syntactic structures. These linguistic features are ranked according to their frequency and importance within the selected texts.

Based on this analysis, Didaskalos constructs a frequency-driven learning path. Rather than following a predetermined order of topics, the system introduces vocabulary and grammar in the sequence that maximizes immediate reading comprehension of the chosen corpus. Each unit in the resulting textbook contains self-contained grammatical explanations and exercises that are automatically generated from the source material. **Crucially, there is not a single word or sentence in a Didaskalos textbook that does not exist in the corpus provided by the user.** This guarantees complete alignment between instruction and authentic language use.

Because the corpus is the user's own, the Greek in a given textbook may span nine centuries and several dialects. The grammar modules therefore take **Attic** — the Greek of Classical Athens — as their reference point: it is the dialect around which grammars and dictionaries are organized, and the one out of which Koine grew. Every paradigm is the Attic one unless the lesson says otherwise, and each lesson closes with a section on how its forms behave in other dialects and periods. How much weight those sections carry depends entirely on the corpus chosen — background reading for an Attic prose selection, essential grammar for one built on Homer or Herodotus. The dialects module, the last of the fixed opening lessons, sets this out in full.

**A note on transliteration.** In this opening group of modules — the alphabet, the dictionary module, and the introductions to nouns, adjectives and verbs — every Greek word and example is followed by its transliteration in Latin letters. This is a temporary scaffold, there to help you internalise the alphabet and check your reading while the letters are still new. It stops after these lessons: from the first frequency-driven lesson onward the Greek stands on its own, because by then you should be reading the Greek script directly rather than through a Latin echo of it. If a form ever defeats you later on, the alphabet module is always there to go back to.

The textbook itself is assembled modularly. Grammar explanations are organized into reusable units that correspond to specific linguistic phenomena, while exercises are dynamically created by extracting and adapting examples from the corpus. Earlier versions of these explanation modules were generated using retrieval-augmented methods; current development focuses on refining their quality and enabling localization into other languages.

Because the underlying pipeline is language-independent, Didaskalos can be extended beyond its original context through translation and cultural adaptation of the grammar modules. This makes it possible to generate fully functional textbooks in multiple languages, including low-resource ones such as Persian, without altering the core system.

Ultimately, Didaskalos represents a shift toward a data-driven, learner-centered model of language education—one in which pedagogy emerges directly from texts, and learning is shaped by the needs and interests of each individual user.

---

## About the Developer

Didaskalos was developed by Farnoosh Shamsian. For more information, please refer to the relevant project pages and publications, or check out https://farnoosh-shamsian.github.io/
