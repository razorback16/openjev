# Vision pool licensing and attribution

This is a **mixed permissive-license** dataset. Do not label the complete pool
as exclusively Apache-2.0.

* OpenJev-authored game renderings, structured images, compiler, tests, and
  methodology: Apache-2.0. Original chess source facts/labels come from CC0
  Lichess exports; the other authored game data is Apache-2.0. Parent source
  licenses remain in provenance.
* CLEVR images, questions, and scene annotations: CC BY 4.0. Attribute Justin
  Johnson, Bharath Hariharan, Laurens van der Maaten, Li Fei-Fei, C. Lawrence
  Zitnick, and Ross Girshick; *CLEVR: A Diagnostic Dataset for Compositional
  Language and Elementary Visual Reasoning*, CVPR 2017. Source and declaration:
  https://cs.stanford.edu/people/jcjohns/clevr/ . Original images are unchanged;
  questions are compiled into OpenJev slots and answer choices.
* Open Images annotations: CC BY 4.0, Google LLC. Source:
  https://storage.googleapis.com/openimages/web/factsfigures.html . Each selected
  photograph has its own verified CC BY or CC0 declaration, author, source URL,
  check timestamp, and modification note in record provenance and the photo
  source snapshot. Preserve these image-level attributions when redistributing.
  JPEGs may be resized, reoriented, and recompressed as documented per image.
* Bundled unmodified DejaVu fonts: Bitstream Vera license with DejaVu changes
  in the public domain. See `dataset/vision/assets/LICENSE-DejaVu.txt`. The font
  license is not being relabeled Apache-2.0.

No share-alike artwork, noncommercial-only dataset, proprietary stock-photo
collection, or paid image-generation output is included. Dependencies remain
under their own licenses; their executable packages are not bundled.

The original source license declarations are preserved under `sources/` in the
release. Shard JSON records carry image-level provenance and notices; distributing
bare photos while dropping that metadata would discard required attribution.
