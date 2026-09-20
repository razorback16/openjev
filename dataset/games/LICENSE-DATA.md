# Game data licensing

* **Chess source rows and their compiled decisions: CC0-1.0.** Lichess explicitly
  releases its database exports for commercial use, modification, and
  redistribution: https://database.lichess.org/. The exact downloaded declaration
  is preserved in `source/lichess-license-source.html`; source URLs, extraction
  counts, response metadata, and checksums are in `sources.lock.json`. Attribution
  to Lichess and its contributors is retained as provenance even under CC0.
* **OpenJev-authored procedural Gomoku/Connect Four data, poker solver outputs,
  compiler, and methodology: Apache-2.0**, under the repository's LICENSE.
* **OpenSpiel: Apache-2.0**, Copyright DeepMind Technologies Limited. Its license
  is included in `dataset/games/notices/OpenSpiel-LICENSE` in the release. It is an external dependency; this
  archive does not redistribute its executable or source tree.

The release has mixed permissive licenses; do not relabel the entire source
collection as exclusively Apache-2.0. Per-record provenance identifies the data
license. No Rapfi binary, Stockfish binary, GPL chess library, or CC-BY-SA artwork
is bundled. The already published CC0 Lichess evaluation data was produced with
Stockfish; the original engine license is not being described as permissive.
No source here requires a Hugging Face token or paid API.

License scope is this dataset and its build tooling. Future model release terms
must also respect the selected base model's actual license and dependencies.
