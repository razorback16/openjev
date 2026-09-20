# Hold'em data and tooling licenses

OpenJev-authored procedural records, compiler, independent audits, tests, and
methodology are released under the repository's **Apache-2.0** license, included
as `LICENSE` in the archive. Every record declares Apache-2.0 provenance.
No third-party player hand histories, proprietary charts, solver databases,
or pretrained poker policies are redistributed.

The hand evaluator is [Treys](https://github.com/ihendley/treys), **MIT**, version
0.1.8. Its original copyright/license notice is retained in
`dataset/holdem/notices/Treys-LICENSE`. Treys is an external dependency, not
vendored code. The independent audit does not use its evaluation tables.

The LP implementation uses SciPy/HiGHS as an external dependency. NumPy,
SciPy, and PyArrow are also externally installed dependencies with their own
permissive notices. The archive does not include their binaries or source trees.
No RLCard, Stockfish, Rapfi, paid API, or GPL poker solver is required.

The license applies to this dataset and its authored tooling; it does not grant
rights to future model weights beyond the selected base model's own terms.
