## About Didaskalos

Didaskalos is a corpus-driven system designed to generate fully personalized textbooks for learning Ancient Greek. Instead of relying on fixed curricula or generalized pedagogical sequences, Didaskalos builds each textbook directly from the texts a learner or instructor actually wants to read. This ensures that every lesson, explanation, and exercise is immediately relevant to the learner’s goals and grounded in authentic material.

At its core, Didaskalos operates by integrating three key resources: a corpus of selected Ancient Greek texts, linguistic treebanks, and a modular grammar framework. The process begins when a user chooses a target corpus. The system then analyzes this corpus using treebank data, identifying vocabulary frequency, morphological patterns, and syntactic structures. These linguistic features are ranked according to their frequency and importance within the selected texts.

Based on this analysis, Didaskalos constructs a frequency-driven learning path. Rather than following a predetermined order of topics, the system introduces vocabulary and grammar in the sequence that maximizes immediate reading comprehension of the chosen corpus. Each unit in the resulting textbook contains self-contained grammatical explanations and exercises that are automatically generated from the source material. **Crucially, there is not a single word or sentence in a Didaskalos textbook that does not exist in the corpus provided by the user.** This guarantees complete alignment between instruction and authentic language use.

The textbook itself is assembled modularly. Grammar explanations are organized into reusable units that correspond to specific linguistic phenomena, while exercises are dynamically created by extracting and adapting examples from the corpus. Earlier versions of these explanation modules were generated using retrieval-augmented methods; current development focuses on refining their quality and enabling localization into other languages.

Because the underlying pipeline is language-independent, Didaskalos can be extended beyond its original context through translation and cultural adaptation of the grammar modules. This makes it possible to generate fully functional textbooks in multiple languages, including low-resource ones such as Persian, without altering the core system.

Ultimately, Didaskalos represents a shift toward a data-driven, learner-centered model of language education—one in which pedagogy emerges directly from texts, and learning is shaped by the needs and interests of each individual user.

---

## About the Developer

Didaskalos was developed by Farnoosh Shamsian. For more information, please refer to the relevant project pages and publications, or check out https://farnoosh-shamsian.github.io/
