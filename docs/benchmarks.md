# Benchmarks

RETICLE is evaluated on five benchmarks, three of them new. All are built from public data; regions and items are sampled with fixed
seeds before any model output, and query ids are neutral (they carry no slide, case or label information). Balanced accuracy is the
mean per-class recall; confidence intervals resample the grouping unit given below.

| Benchmark | Question | Source (licence) | Test n | Metric (bootstrap group) |
|---|---|---|---:|---|
| CAMELYON16-MS (new) | Largest metastatic deposit in a 4.6–5.0 mm lymph-node region: none / ITC (≤ 0.2 mm) / micro (0.2–2 mm) / macro | CAMELYON16 test slides, official masks (CC0) | 188 | balanced accuracy (129 slides) |
| MIDOG-MS (new) | Number of mitotic figures in a 0.8 × 0.6 mm region: 0 / 1 / 2–3 / ≥ 4 | MIDOG 2021, 60 held-out cases (CC BY 4.0) | 240 | balanced accuracy (60 cases) |
| UniToPatho-MS 800 µm | Polyp type and dysplasia grade, six classes | UniToPatho, official split | 2,399 | balanced accuracy (88 slides) |
| UniToPatho-MS 7 mm | Same question on 7 mm regions | UniToPatho, official split | 248 | balanced accuracy (88 slides) |
| PathMMU | Mixed pathology multiple choice | PathMMU test | 4,538 | accuracy (images) |
| MultiCaRe-Dx (new) | Three turns: diagnosis and differential; IHC / special-stain / molecular work-up; treatment and outcome | MultiCaRe case reports (CC BY) | 1,000 | top-1 diagnosis (items) |

## CAMELYON16-MS — `reticle/benchmarks/camelyon16_ms.py`

- **Regions.** 10,240 px at level 1 (4.6–5.0 mm, 0.45–0.49 µm/px); the agent sees a downsampled overview and full-resolution views at the regions located by the metastasis expert.
- **Label.** The bounding-box extent of the largest connected deposit of the official exhaustive tumour mask (level 3), mapped to the
  pN size categories: none, isolated tumour cells (≤ 0.2 mm), micrometastasis (0.2–2 mm), macrometastasis (> 2 mm).
- **Sampling.** Around the largest deposit and up to two small deposits (≤ 2 mm) of each tumour slide; one random tissue region per
  negative slide.
- **Splits.** Test: 188 regions of the 129 official test slides (80 without tumour, 13 isolated tumour cells, 55 micrometastases,
  40 macrometastases). The training
  slides are split by slide hash into expert-training slides (tiles for the metastasis expert, `trreg` regions for calibrating its
  reader) and development slides (`dev`, `dev2`).

## MIDOG-MS — `reticle/benchmarks/midog_ms.py`

- **Regions.** The ~2 mm² high-power region of each of the 150 MIDOG 2021 training cases (three scanners, 0.23–0.25 µm/px) is split
  into 2 × 2 quadrants of about 0.8 × 0.6 mm.
- **Label.** The number of consensus mitotic figures in the quadrant, binned 0 / 1 / 2–3 / ≥ 4 (bins fixed on the training cases).
  Consensus look-alikes are kept as hard negatives for the mitosis expert.
- **Splits.** Per scanner, cases are split by hash into 20 test, 5 development and 25 expert-training cases: 240 test regions
  (70 / 56 / 64 / 50 per category).

## UniToPatho-MS — `reticle/benchmarks/unitopatho_ms.py`

- **Question.** Six classes: normal mucosa, hyperplastic polyp, tubular or tubulovillous adenoma with low- or high-grade dysplasia.
- **Scales.** 800 µm regions (1,812 px at 0.44 µm/px; 2,399 test regions) and 7 mm regions (15,855 px; 248 test regions), on the
  official split by whole-slide image (204 training / 88 test slides). On 7 mm regions the overview loses the nuclei; grading needs
  full-resolution views.
- This set was also used during development, so its results are less clean than those of the new sets.

## PathMMU — `reticle/benchmarks/pathmmu.py`

- The val and test questions whose images are available in the release are frozen (4,538 of the 8,521 test questions); answers are
  stored apart from the model inputs.
- The teaching-caption library is decontaminated against PathMMU: captions sharing a 10-gram with any PathMMU question, option or
  explanation are removed (`pathmmu.py`), and so are images with MUSK cosine ≥ 0.90 to a PathMMU image and whole sources with any
  image at ≥ 0.93 (`reticle/library/retrieve_captions.py`).

## MultiCaRe-Dx — `reticle/benchmarks/multicare_dx.py`

- **Items.** 1,000 published cases, one per article, sampled from the MultiCaRe articles with an H&E figure and at least 800
  characters of case text (articles of the demo examples excluded).
- **Construction.** For each case the VLM (Qwen3.8-27B, no reasoning, JSON schema) writes from the case text an 80–200-word clinical
  presentation without the pathology result, and extracts the final diagnosis with aliases, the reported work-up, the treatment and
  the follow-up. Items describing several patients, lacking a histopathological diagnosis or leaking the diagnosis (flagged by the
  model or found by a lexical check of diagnosis-specific words) are excluded.
- **Turns.** (1) diagnosis with a differential from the figure and the presentation, (2) confirmatory immunohistochemistry / special
  stains / molecular work-up with expected results, (3) treatment and reported outcomes. The questions of the reported runs are in
  `configs/evaluation/consultation.json`.
- **Leakage control.** At answer time the case's own article and every figure with MUSK cosine ≥ 0.90 to the case figure are excluded
  from the evidence.
- **Splits.** A development split of 299 cases from disjoint articles (`--split dev`) was used for all design choices.
- **Metric.** The same VLM, blind to the arm, judges whether the first-turn diagnosis is the same disease entity as the gold
  diagnosis (top-1); `reticle/evaluation/consultation.py` also reports top-3 (gold among the differential).

## Evidence library — `reticle/library/`

The agent's evidence and the PathMMU caption evidence are built from public sources below the data root (`RETICLE_DATA`); none of them is
redistributed here.

| Library | Source (terms) | Raw files under the data root | Built by | Output |
|---|---|---|---|---|
| Published case reports | MultiCaRe 3.0.1, Zenodo [10.5281/zenodo.20416562](https://doi.org/10.5281/zenodo.20416562): case narratives, figures and captions of open-access PMC case reports (each article's own licence, CC BY or CC0) | `datasets/multicare/` (fetched by `cases download`) | `cases.py` | `library/cases/` |
| Teaching image–caption pairs | [Quilt-1M](https://github.com/wisdomikezogwo/quilt1m) (release CSV and image folders, authors' terms) and the 2,141-pair PathGen sample shared with the hackathon (PathGen-1.6M) | `datasets/quilt_1m/`, `datasets/pathgen/` | `captions.py` | `library/captions/corpus.jsonl` |
| PathMMU images (for exclusion) | [PathMMU](https://github.com/PathMMU-Benchmark/PathMMU) `data.json`, `socialpath_mapping.json`, `images.zip` | `datasets/pathmmu/` | `captions.py` | `datasets/pathmmu/qa_image_cache/` |
| NCI PDQ treatment summaries | cancer.gov, Health Professional versions ([reuse policy](https://www.cancer.gov/policies/copyright-reuse)) | — | `pdq.py` | `library/pdq/` |

- **Case reports.** Only pathology figures whose case link is consistent between `case_images.parquet` and `captions_and_labels.csv`
  enter the figure index (27,547 in the reported runs); the full-text index covers every case narrative (98,641 cases). Eight articles
  with wrong upstream case records are left out of the figure index (`configs/library_cases.json`).
- **Caption pairs.** 591,524 images after removing non-histology and short captions, PathMMU tweets and file names, exact duplicates and
  every image whose SHA-256 or perceptual hash matches a PathMMU question image; the PathMMU steps run in the order `captions.py` →
  `reticle.benchmarks.pathmmu` → `encode.py` → `retrieve_captions.py`; the last step also removes captions sharing a 10-gram with
  PathMMU text and images or sources visually near a PathMMU image (363,905 pairs remain for the PathMMU evidence).
