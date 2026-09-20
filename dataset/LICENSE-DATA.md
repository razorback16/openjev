# Dataset licensing

Compiler code and methodology contributed by OpenJev are available under the
repository's Apache-2.0 license. Source-derived data retains its original license.
There is no blanket Apache-2.0 relicensing of the records.

| Source | Record license | Attribution |
|---|---|---|
| BANKING77 | CC-BY-4.0 | PolyAI; Casanueva et al., Efficient Intent Detection with Dual Sentence Encoders (2020) |
| WANLI | CC-BY-4.0 | Liu et al., WANLI: Worker and AI Collaboration for Natural Language Inference Dataset Creation (2022) |
| bAbI / bAbI-NLI | BSD-3-Clause | Facebook; Weston et al.; TaskSource NLI conversion by Damien Sileo |
| RuleTaker | Apache-2.0 | Allen Institute for AI; Clark, Tafjord, Richardson (2020) |
| Logical entailment | Apache-2.0 | DeepMind; Evans, Saxton, Amos, Kohli, Grefenstette (2018) |
| Cosmos QA | CC-BY-4.0 | Huang, Le Bras, Bhagavatula, Choi (2019) |
| GoEmotions | Apache-2.0 | Google Research; Demszky et al. (2020) |
| HelpSteer | CC-BY-4.0 | NVIDIA; Wang et al. (2023/2024) |

TaskSource's instruction conversion is credited to Damien Sileo. FLAN templates
and methodology are credited to Google Research and Longpre et al.; the
download mirror is credited to TaskSource. See `training_sources.yaml` for
upstream links and license evidence, `sources.lock.json` for exact source bytes,
and `notices/` for source cards and license texts.

OpenJev's changes are deterministic filtering, schema conversion, option
selection and reordering, additional binary questions, vote aggregation,
ordinal rubric abbreviation, grouping, splitting, and deduplication. Attribution
and source licenses must accompany redistributed records and their derivatives
as required by those licenses. CC BY attribution requirements are not waived.
Original licensors do not endorse OpenJev. Third-party rights in underlying
content are not expanded by this compilation.

No noncommercial, research-only, share-alike, or unknown-license records are
intentionally included. License declarations are supported by the linked
upstream evidence; this manifest records that evidence rather than asserting
ownership of third-party content.
