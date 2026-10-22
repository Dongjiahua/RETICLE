# Evaluation

All commands run from the repository root with `RETICLE_DATA` pointing to the data root (see README). Answers are written next to
each benchmark under `$RETICLE_DATA/runs/<benchmark>/`; scores go to `$RETICLE_DATA/results/`. GPU steps need a vLLM server of
Qwen3.8-27B (or of LLaVA-Med / Quilt-LLaVA for those baselines) on the given ports; scoring runs on the CPU from the stored answers.
`reticle.agent.vlm_qa`, and `reticle.agent.region_qa` without `--api_ports`, load the VLM in-process with vLLM, so they run inside the
vLLM image (`vllm/vllm-openai:v0.31.0`, see README) rather than the runtime image; with `--api_ports` region_qa only needs the runtime
image and running servers.

Every system has one arm per benchmark:

| Arm | System |
|---|---|
| `reticle` | RETICLE: the expert's reading and the full-resolution views at the expert's regions (region benchmarks), retrieved captions (PathMMU), retrieved evidence with cited answers (MultiCaRe-Dx) |
| `qwen` | the same VLM answering directly |
| `musk` | MUSK linear probe (region benchmarks), zero-shot (PathMMU), nearest published case (MultiCaRe-Dx) |
| `llavamed`, `quilt` | LLaVA-Med v1.5, Quilt-LLaVA v1.5-7B |

On the region benchmarks the arm names carry the set as a prefix (`c16_reticle`, `mdg_qwen`, ...). On UniToPatho-MS 800 µm the
reported RETICLE score is the polyp expert's reading, `u8_reticle_expert` (written by `reticle.experts.polyp.evidence`).

## Region benchmarks (CAMELYON16-MS, MIDOG-MS, UniToPatho-MS)

```bash
python -m reticle.agent.region_qa --arms configs/evaluation/region_arms.json --shard 0 --nshards 1 --api_ports 8931,8932   # reticle, qwen
python -m reticle.evaluation.baselines.musk probe u8 u7 c16 mdg                                                           # musk (10 seeds)
python -m reticle.evaluation.baselines.chat_vlms mc --model llavamed --port 8980 --judge_ports 8931 --sets c16 mdg u7 u8
python -m reticle.evaluation.baselines.chat_vlms mc --model quilt --port 8980 --judge_ports 8931 --sets c16 mdg u7 u8
python -m reticle.evaluation.score_regions --config configs/evaluation/region_comparisons.json --out $RETICLE_DATA/results/regions_test.json
```

The `reticle` arms set `"route": true`: the agent first chooses the expert for each region from its overview and question
(`reticle.agent.router`), then receives that expert's reading and views. `python -m reticle.agent.router eval` checks the router on a
sample of test items of all six benchmarks.

LLaVA-Med and Quilt-LLaVA are served by vLLM (Quilt-LLaVA after `python -m reticle.evaluation.baselines.convert_llava_to_hf`).
`score_regions` reports balanced accuracy with slide-grouped bootstrap intervals for RETICLE against each baseline. The MUSK score in
the results table is the mean over the 10 probe seeds (`$RETICLE_DATA/runs/musk_baselines/probe.json`); seed 0 is the scored arm.

## PathMMU

```bash
python -m reticle.library.captions                       # caption library without PathMMU images; extracts the question images
python -m reticle.benchmarks.pathmmu                     # frozen questions, answers kept apart, text decontamination
python -m reticle.library.encode --rank 0 --world 1      # MUSK vectors of the caption library and the question images
python -m reticle.library.retrieve_captions              # decontamination and caption retrieval
python -m reticle.agent.vlm_qa --arms configs/evaluation/pathmmu_arms.json --split test --shard 0 --nshards 1 --out $RETICLE_DATA/runs/pathmmu_v3/qa
python -m reticle.evaluation.baselines.musk pathmmu --split test
python -m reticle.evaluation.baselines.chat_vlms mc --model llavamed --port 8980 --judge_ports 8931 --sets pmmu   # likewise quilt
python -m reticle.evaluation.score_pathmmu --split test --reference qwen --out $RETICLE_DATA/results/pathmmu_test.json
```

Accuracy on 4,538 questions with an image-grouped bootstrap against `qwen`.

## MultiCaRe-Dx

```bash
python -m reticle.library.cases download && python -m reticle.library.cases build    # case library (CPU)
python -m reticle.library.cases encode                                              # MUSK vectors of its figures
python -m reticle.benchmarks.multicare_dx --ports 8931,8932                         # items and gold (extraction by the VLM)
python -m reticle.evaluation.consultation answer --arm reticle --ports 8931,8932
python -m reticle.evaluation.consultation answer --arm qwen --ports 8931,8932
python -m reticle.evaluation.baselines.chat_vlms mcdx --model llavamed --port 8980                 # likewise quilt
python -m reticle.evaluation.consultation judge --ports 8931,8932                                  # reticle, qwen, llavamed, quilt
python -m reticle.evaluation.baselines.musk mcdx --ports 8931                                      # answers and judges musk
python -m reticle.evaluation.consultation summary
```

Top-1 (and top-3) diagnosis on 1,000 cases with an item bootstrap for RETICLE against each baseline.

## Results

| Benchmark | `reticle` | `qwen` | `musk` | `llavamed` | `quilt` |
|---|---:|---:|---:|---:|---:|
| CAMELYON16-MS | 72.1 | 28.6 | 42.2 | 29.0 | 26.0 |
| MIDOG-MS | 66.5 | 31.4 | 36.7 | 24.6 | 25.0 |
| UniToPatho-MS 800 µm | 53.4 (expert reading; agent 47.6) | 22.2 | 48.2 | 21.0 | 22.4 |
| UniToPatho-MS 7 mm | 49.2 | 21.1 | 42.4 | 11.6 | 13.2 |
| PathMMU | 75.3 | 72.4 | 47.7 | 15.7 | 42.8 |
| MultiCaRe-Dx (top-1) | 42.7 | 39.3 | 4.5 | 12.9 | 11.8 |

Answers of the reasoning model vary by about three points between runs because of batched inference.
