# The RETICLE expert interface

Every RETICLE expert answers a question about an image through the same record, whatever it does inside (a segmenter and a size
rule, a detector and a counting rule, or a mixture of classifiers). The agent never sees model internals; it sees this record,
rendered as a few lines of text, next to the image and the question.

## Record

For a question with answer options `1..K`, an expert returns

| field | meaning |
|---|---|
| `card` | a short description of the expert: what it measures, its input domain, the data it was trained on and its known failure modes |
| `probs` | calibrated probabilities over the question's options (they sum to 1) |
| `confidence` | the probability of the top option (`max(probs)`) |
| `rois` | up to ten located regions `{box, score}`, strongest first; boxes are in the 0-1000 coordinates of the image overview |
| measurements | the raw readings behind the probabilities, e.g. the extent and area of the largest tumour cluster, or the number of candidate mitoses |

`reticle/experts/contract.py` defines the record (`record`), checks it (`validate`) and writes one JSONL file per task and split to
`$RETICLE_DATA/runs/contract_v1/<name>_<split>.jsonl`. The card and the measurements are part of the evidence text the agent
receives (`evidence_<task>_<split>_<reader>.jsonl` next to each benchmark), for example:

```
Largest tumour cluster (segmentation >= 0.95): extent 1318 um, area 0.870 mm2; 50 separate clusters.
Strongest clusters (0-1000 coordinates, probability): [420, 394, 705, 664] 1.00, ...
Expert's calibrated reading: option C (77%), then D (18%).
```

The agent is told that the reading is a measurement that can be wrong and that it may not be cited as literature. The regions
decide where the look-closer tool cuts full-resolution views.

## Calibration

Probabilities are calibrated on a held-out pool of the expert's own training data, never on test regions:

| expert | reading | calibration pool |
|---|---|---|
| metastasis (`metastasis/segmenter.py`) | largest 8-connected cluster of the token map at threshold 0.95 -> extent -> none / ITC / micro / macro | 319 regions of the expert-train slides, out-of-fold maps |
| mitosis (`mitosis/crossfit.py`) | peaks of the detector map above the chosen threshold -> count -> 0 / 1 / 2-3 / >= 4 | 360 regions (dev + expert-train), out-of-fold maps |
| polyp, 800 um (`polyp/mixture.py`) | gated mixture of seven experts | 600 dev regions |
| polyp, 7 mm (`polyp/mixture.py`) | learned aggregation of the 800 um tile readings | 619 training regions, out-of-fold |

For a reading that is a discrete measurement (a size class or a count class), the out-of-fold readings `r_i` and labels `y_i` of
the pool give a smoothed confusion matrix `R[k, j] = (#{y_i = k, r_i = j} + 1) / (n_k + K)`. The posterior under a uniform class
prior, `p(y = k | r = j) = R[k, j] / sum_k' R[k', j]`, matches the balanced-accuracy objective of the benchmarks. Probabilistic
readers are calibrated in the same way after removing the pool's class prevalence from their log-probabilities. Finally a
temperature `T` is fitted by negative log-likelihood on the pool: `q = softmax(log p / T)` (`fit_temperature`, `apply_temperature`).
Because the prior is uniform, `q` is calibrated for a balanced class mix, not for the prevalence of a particular test set.

## Adding an expert

A new task needs only a new expert that writes this record and its evidence text; the agent, its prompts and its decoding are
unchanged.
