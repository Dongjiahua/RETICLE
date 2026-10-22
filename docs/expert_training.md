# Training the RETICLE experts

RETICLE has three experts, one per measuring task. Each is built on MUSK and trained only on the training slides of its benchmark:

| Expert | Benchmark | What is trained | Record and evidence the agent reads |
|---|---|---|---|
| Metastasis | CAMELYON16-MS | MUSK-V fine-tuned as a tumour segmenter, 5 slide folds, 2 rounds | `c16seg3`, `evidence_c16_test_seg3.jsonl` |
| Mitosis | MIDOG-MS | MUSK-V fine-tuned as a mitosis detector, 5 case folds, 3 rounds | `mdg4`, `evidence_mdg_test_cv3.jsonl` |
| Polyp | UniToPatho-MS | a mixture of seven experts on frozen MUSK-V features | `u8`, `u7`, `evidence_u{8,7}_test_moe.jsonl` |

Records go to `$RETICLE_DATA/runs/contract_v1/` (format in [expert_interface.md](expert_interface.md)); the evidence files go next to
each benchmark in `$RETICLE_DATA/runs/<benchmark>/` and are what `reticle.agent.region_qa` gives the agent
(`configs/evaluation/region_arms.json`). Training recipes are in `configs/experts/`.

## Before you start

- Accept MUSK's terms on Hugging Face (`xiangjx/musk`) and log in, so the image can download the weights.
- Build the benchmarks (see [benchmarks.md](benchmarks.md)): `python -m reticle.benchmarks.camelyon16_ms`, `midog_ms` and
  `unitopatho_ms`. They also write the expert-training material: `tiles_train.jsonl` (CAMELYON16) and `cells_train.jsonl` (MIDOG).
- For round 3 of the mitosis expert, put MIDOG++ (CC0) under `$RETICLE_DATA/datasets/midogpp/` (`meta/MIDOG++.json`, `images/`).

Every command below runs in the runtime image, as in the README. GPU steps need `--gpus`; one GPU per command is enough, and the
fold commands (`--fold k`) and sharded commands (`--shard i --nshards n`) are independent, so they can run in parallel, one GPU each:

```bash
run() { docker run --rm --gpus "device=${GPU:-0}" --ipc host -e RETICLE_DATA=/data -v /path/to/data:/data -v $PWD:/workspace \
          -w /workspace reticle python -m "$@"; }
```

## Metastasis expert (CAMELYON16-MS)

```bash
run reticle.benchmarks.camelyon16_ms trreg                                       # regions of the training slides (CPU), for the reader
# frozen tile mixture: its maps are reader candidates of the segmenter
run reticle.experts.metastasis.features train
run reticle.experts.metastasis.features regions --split trreg
run reticle.experts.metastasis.features regions --split test
run reticle.experts.metastasis.tile_experts
# segmenter, round 1 (k = 0..4)
run reticle.experts.metastasis.segmenter tiles
run reticle.experts.metastasis.segmenter train --round 1 --fold $k
run reticle.experts.metastasis.segmenter apply --round 1 --fold $k --split trreg
# hard negatives from the round-1 maps, round 2 (k = 0..4)
run reticle.experts.metastasis.segmenter mine
run reticle.experts.metastasis.segmenter train --round 2 --fold $k
run reticle.experts.metastasis.segmenter apply --round 2 --fold $k --split trreg
run reticle.experts.metastasis.segmenter apply --round 2 --fold $k --split test
# reader, calibration, records and evidence
run reticle.experts.metastasis.segmenter read --round 2 --tag 3
```

Model k never sees the slides of fold k. The training regions are read with their out-of-fold model, the test regions with the
mean of the five models. `read` picks the reader (rule: largest cluster ≥ 0.95, then its extent) by cross-validation on the 319
training regions and calibrates it.

## Mitosis expert (MIDOG-MS)

```bash
# frozen mixture (averaged with the detectors in the final reading) and the point grid
run reticle.experts.mitosis.features train
run reticle.experts.mitosis.features regions --split train    # likewise dev, test
run reticle.experts.mitosis.counter
# round 1: fold 0 by finetune.py, folds 1..4 by crossfit.py; then maps of every region (k = 0..4)
run reticle.experts.mitosis.finetune train
run reticle.experts.mitosis.crossfit train --round 1 --fold $k                   # k = 1..4
run reticle.experts.mitosis.crossfit apply --round 1 --fold $k --split train     # likewise dev, test
# round 2: hard negatives and shifted views (k = 0..4)
run reticle.experts.mitosis.crossfit mine
run reticle.experts.mitosis.crossfit train --round 2 --fold $k
run reticle.experts.mitosis.crossfit apply --round 2 --fold $k --split train     # likewise dev, test
# round 3: adds the non-breast MIDOG++ cases (k = 0..4)
run reticle.experts.mitosis.crossfit extra
run reticle.experts.mitosis.crossfit train --round 3 --fold $k
run reticle.experts.mitosis.crossfit apply --round 3 --fold $k --split train     # likewise dev, test
# reader, calibration, records and evidence
run reticle.experts.mitosis.crossfit read --max-round 3      # the benchmark expert (records mdg4, evidence cv3)
run reticle.experts.mitosis.crossfit read --max-round 2      # the demo's live tool (records mdg3, cv_read2.json)
```

Model k of each round is trained on the cases of the other four folds. A fold whose training diverged was retrained once with
another seed; `seed_offset` in `configs/experts/mitosis_detectors.json` records which. `read` compares readers, maps and rounds by case-grouped
cross-validation on the 360-region pool (development and training regions) and calibrates the chosen one.

## Polyp expert (UniToPatho-MS)

```bash
# frozen MUSK-V features: whole regions, 2 x 2 and 4 x 4 sub-tiles, and the 800 µm tiles of the 7 mm regions
run reticle.experts.polyp.probes features
run reticle.experts.polyp.subfields features
run reticle.experts.polyp.subfields features_u7 --shard $i --nshards $n
run reticle.experts.polyp.subfields features_u7train --shard $i --nshards $n
# the mixture: choose the gate by cross-validation, apply it, and learn the 7 mm aggregation
run reticle.experts.polyp.mixture select
run reticle.experts.polyp.mixture apply
run reticle.experts.polyp.mixture agg7
# evidence for the agent and the development pool, then calibrated records
run reticle.experts.polyp.evidence test
run reticle.experts.polyp.evidence dev
run reticle.experts.build_contracts u8 u7
```

The seven experts read the same region at different magnifications (whole region, glands, nuclei, single tiles, attention over
tiles, nearest training regions, and type x grade heads). `select` chooses how to combine them by WSI-grouped cross-validation on
the official training slides (the geometric mean was chosen).

## Checks

Each step writes its cross-validation scores next to its outputs (`expert_selection.json`, `moe/selection.json`, the readers'
`read3.json` for metastasis and `cv_read3.json` for mitosis), and `build_contracts` writes `results/contract_v1/calibration.json`. The live demo tools reuse these files: the
mitosis tool loads the five round-2 detectors, and the polyp tool refits its classifiers on the cached features.
