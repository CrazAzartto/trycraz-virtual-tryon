# Demo catalogue: image sources and licences

This is a **small demonstration catalogue** assembled for the project. It is not a research dataset and no statistics about it should be read as general findings.

## Images `cd_*.jpg`

Source: **Clothing Dataset** by Alexey Grigorev and contributors (2020), https://github.com/alexeygrigorev/clothing-dataset (also on Kaggle as `agrigorev/clothing-dataset-full`).
Licence: the dataset README states it "can be freely used for any purpose, including commercial"; Kaggle lists it as CC0 (public domain).

- 52 images were chosen by hand for: plain-ish background, whole garment visible, no visible brand logos or cartoon characters.
- The GitHub copy is low resolution (400 px on the short side). The Kaggle copy has higher-resolution versions of the same images.
- The original file name (`<dataset label>__<image id>.jpg`) is recorded in each catalogue row's `notes`.
- Processing: EXIF orientation fixed, cropped to the garment where the background allowed, saved as JPEG.

## Annotations

- `manual_annotations.csv`: garment type, colours, pattern, styles, formality/warmth overrides, and wear line, written by hand from visual inspection during development (by the AI assistant used on the project). Rows carry `annotation=manual` until the project owner confirms them (`verified`).
- Occasion tags are derived from those attributes by the rule in `scripts/build_catalogue.py::derive_occasions`.
- Automatic guesses (CLIP type, k-means colours) are kept in each row's notes; their agreement with the manual labels is reported in `evaluation/results/auto_annotation_agreement.md`.

## Known gaps

No traditional Indian wear (kurta, sherwani, saree, lehenga): the source dataset has none, and the project owner has no photos of them yet. Wedding recommendations therefore draw on blazers, dresses and festive tops only. This is stated as a limitation.
