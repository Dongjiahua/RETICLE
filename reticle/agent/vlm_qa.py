"""Multiple-choice pathology QA with retrieved teaching captions as evidence (PathMMU): one offline vLLM engine per GPU, several arms per
model load, resumable per arm. The model reasons before answering. An arm (JSON) sets: name, max_pixels of the question image, effort
(reasoning), evidence (file of retrieved captions, from reticle.library.retrieve_captions) with k, seed, max_tokens.
  python3 -m reticle.agent.vlm_qa --arms arms.json --split test --shard i --nshards n --out DIR      (vLLM container, one GPU)"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import time

from PIL import Image, PngImagePlugin

from reticle.paths import DATA

PngImagePlugin.MAX_TEXT_CHUNK = 64*2**20  # some library PNGs carry large ICC/text chunks

MODEL = os.environ.get('RETICLE_VLM', 'Qwen/Qwen3.8-27B')   # answer model: Hugging Face id (pinned revision below) or a local snapshot
MODEL_REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
SYSTEM = 'You are an expert pathologist answering multiple-choice questions about pathology images.'
EVIDENCE_HEAD = ('Reference library: captions of the most visually similar images found by a pathology image search engine. '
                 'They describe other cases, can be irrelevant or misleading, and are not answer keys. '
                 'Use them only when they genuinely help interpret the query image.')


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().split('\n') if x]


def load_image(path, max_pixels):
    im = Image.open(path).convert('RGB')
    if im.width*im.height > max_pixels:
        s = math.sqrt(max_pixels/(im.width*im.height))
        im = im.resize((max(28, int(im.width*s)), max(28, int(im.height*s))), Image.BICUBIC)
    return im


def prompt(q, evidence):
    text = 'Question: '+q['question']+'\nOptions:\n'+'\n'.join(q['options'])
    if evidence:
        lines = [f"[{i}] (similarity {e['score']:.2f}) {e['caption'][:600]}" for i, e in enumerate(evidence, 1)]
        text += '\n\n'+EVIDENCE_HEAD+'\n'+'\n'.join(lines)
    return text+'\n\nThink it through, then end with a final line exactly of the form "Answer: <letter>".'


def parse(raw, options):
    letters = [o[0] for o in options]
    tail = raw.rsplit('</think>', 1)[-1]
    found = re.findall(r'Answer\s*[:：]\s*\**\(?([A-Z])\b', tail)
    if found and found[-1] in letters:
        return found[-1]
    bare = tail.strip().strip('*').strip()
    return bare[0] if len(bare) == 1 and bare in letters else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arms', required=True, help='JSON file with a list of arm specs')
    p.add_argument('--split', required=True)
    p.add_argument('--shard', type=int, required=True)
    p.add_argument('--nshards', type=int, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--model', default=MODEL, help='answer model (Hugging Face id or local snapshot)')
    p.add_argument('--base', default=str(DATA/'runs/pathmmu_v3'), help='run dir with inputs/ (reticle.benchmarks.pathmmu) and evidence files')
    a = p.parse_args()
    from vllm import LLM, SamplingParams
    base = Path(a.base)
    queries = read_jsonl(base/f'inputs/queries_{a.split}.jsonl')[a.shard::a.nshards]
    arms = json.loads(Path(a.arms).read_text())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    todo = [arm for arm in arms if not (out/f"{arm['name']}.{a.split}.{a.shard}.jsonl").exists()]
    if not todo:
        return
    llm = LLM(model=a.model, revision=MODEL_REVISION if a.model == 'Qwen/Qwen3.8-27B' else None, max_model_len=20480, gpu_memory_utilization=0.88,
              limit_mm_per_prompt={'image': 1}, seed=20261018, enable_prefix_caching=True)
    for arm in todo:
        began = time.monotonic()
        evidence = {}
        if arm.get('evidence'):
            evidence = {r['id']: r['evidence'][:arm.get('k')] for r in read_jsonl(base/arm['evidence'].format(split=a.split))}
        conversations = []
        for q in queries:
            content = [dict(type='image_pil', image_pil=load_image(q['image_path'], arm['max_pixels'])), dict(type='text', text=prompt(q, evidence.get(q['id'], [])))]
            conversations.append([dict(role='system', content=SYSTEM), dict(role='user', content=content)])
        params = SamplingParams(temperature=1.0, top_p=0.95, top_k=20, max_tokens=arm.get('max_tokens', 8192), seed=arm.get('seed', 20261018))
        results = llm.chat(conversations, params, chat_template_kwargs=dict(enable_thinking=True, reasoning_effort=arm.get('effort', 'low')), use_tqdm=False)
        rows = []
        for q, conv, r in zip(queries, conversations, results):
            raw = r.outputs[0].text
            rows.append(dict(id=q['id'], arm=arm['name'], choice=parse(raw, q['options']), raw=raw[-1500:],
                             finish=r.outputs[0].finish_reason, prompt_tokens=len(r.prompt_token_ids),
                             output_tokens=len(r.outputs[0].token_ids), n_evidence=conv[1]['content'][-1]['text'].count('(similarity ')))
        tmp = out/f"{arm['name']}.{a.split}.{a.shard}.tmp"
        tmp.write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in rows))
        tmp.rename(out/f"{arm['name']}.{a.split}.{a.shard}.jsonl")
        print(json.dumps(dict(arm=arm['name'], split=a.split, shard=a.shard, n=len(rows), seconds=round(time.monotonic()-began, 1),
                              unparsed=sum(x['choice'] is None for x in rows))), flush=True)


if __name__ == '__main__':
    main()
