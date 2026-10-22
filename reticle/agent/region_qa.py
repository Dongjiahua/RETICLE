"""The RETICLE agent on large-region multiple-choice questions (CAMELYON16-MS, MIDOG-MS, UniToPatho-MS), with the look-closer tool.

The VLM never receives one huge image. It sees the whole region as an overview of at most `overview_px` per side, then views of fixed
physical size at the regions returned by the RETICLE expert (evidence file fields 'coarse' / 'fine'), every image at most `crop_px` per side:
  level 2 (coarse): up to 3 windows of `coarse_um`, shown downsampled (e.g. 1.6 mm at ~2 um/px: architecture);
  level 1 (fine):   up to 3 fields of `fine_px` native pixels (~340 um at ~0.45 um/px: nuclei, single tumour cells).
'crops': 'expert' = views at the expert's regions (RETICLE), 'none' = overview only. 'route': true lets the agent
choose the expert first (reticle.agent.router, needs --api_ports); a region routed to another expert or to none gets no reading and no views. The expert's reading ('evidence':
key of runs/<set>/evidence_<set>_<split>_<key>.jsonl, written by the expert pipelines) is given as text. An arm (JSON) sets: name, set
(c16 | mdg | u8 | u7), evidence, crops, levels, coarse_um, fine_px, crop_px, n_coarse, n_fine, overview_px, seed, split.
  python3 -m reticle.agent.region_qa --arms arms.json --shard i --nshards n [--split test|dev] [--limit k] [--api_ports 8000,8001]"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from reticle.agent.vlm_qa import MODEL, MODEL_REVISION, SYSTEM, parse, read_jsonl
from reticle.paths import DATA

Image.MAX_IMAGE_PIXELS = None
BASE = (DATA/'runs')
RUNS = {'u8': BASE/'unitopatho_ms_v1', 'u7': BASE/'unitopatho_ms_v1', 'c16': BASE/'camelyon16_ms_v1', 'mdg': BASE/'midog_ms_v1'}
NATIVE_MPP = {'u8': 0.44, 'u7': 0.44}  # c16 queries carry their own 'mpp'


def fmt_um(um):
    return f'{um/1000:.1f} mm' if um >= 1000 else f'{um:.0f} um'


def window(cx, cy, wx, wy):  # fixed-size box (0-1000) centred at (cx, cy), shifted inside the region
    x1, y1 = min(max(cx-wx/2, 0), 1000-wx), min(max(cy-wy/2, 0), 1000-wy)
    return [round(x1, 1), round(y1, 1), round(x1+wx, 1), round(y1+wy, 1)]


def crop(native, box, max_px):
    H, W = native.shape[:2]
    x1, y1, x2, y2 = int(round(box[0]*W/1000)), int(round(box[1]*H/1000)), int(round(box[2]*W/1000)), int(round(box[3]*H/1000))
    im = Image.fromarray(np.asarray(native[y1:y2, x1:x2]))
    native_w = im.width
    if max(im.size) > max_px:
        s = max_px/max(im.size); im = im.resize((max(28, round(im.width*s)), max(28, round(im.height*s))), Image.LANCZOS)
    return im, native_w/im.width


class Item:
    def __init__(self, q, s, overview_px):
        self.q = q
        self.native = np.load(q['native'], mmap_mode='r') if q.get('native') else np.asarray(Image.open(q['image_path']).convert('RGB'))
        self.mpp = q.get('mpp', NATIVE_MPP.get(s))
        H, W = self.native.shape[:2]; self.w_um, self.h_um = W*self.mpp, H*self.mpp
        ov = Image.open(q['image_path']).convert('RGB')
        if max(ov.size) > overview_px:
            s_ = overview_px/max(ov.size); ov = ov.resize((round(ov.width*s_), round(ov.height*s_)), Image.LANCZOS)
        self.overview = ov; self.ov_mpp = self.w_um/ov.width

    def units(self, um):  # physical size -> (wx, wy) in 0-1000 units
        return min(1000.0, 1000*um/self.w_um), min(1000.0, 1000*um/self.h_um)


class APILLM:
    """llm.chat(convs, params, chat_template_kwargs=...) over loopback OpenAI-compatible servers, returning objects shaped like vLLM's
    RequestOutput (outputs[0].text / finish_reason / token_ids, prompt_token_ids); reasoning is re-joined before </think> as offline."""
    def __init__(self, ports):
        self.ports = ports

    def chat(self, convs, params, chat_template_kwargs=None, use_tqdm=False):
        import base64, io, itertools, urllib.request
        from concurrent.futures import ThreadPoolExecutor
        from types import SimpleNamespace
        cyc = itertools.cycle(self.ports)

        def part(c):
            if c.get('type') != 'image_pil':
                return c
            buf = io.BytesIO(); c['image_pil'].save(buf, 'PNG')
            return dict(type='image_url', image_url=dict(url='data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()))

        def one(args):
            conv, port = args
            msgs = [dict(m, content=[part(c) for c in m['content']]) if isinstance(m['content'], list) else m for m in conv]
            body = dict(model='qwen', messages=msgs, chat_template_kwargs=chat_template_kwargs or {}, **params)
            for attempt in range(3):
                try:
                    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
                    r = json.load(urllib.request.urlopen(req, timeout=3600)); break
                except Exception:
                    if attempt == 2:
                        raise
                    port = next(cyc)
            m = r['choices'][0]['message']; rs = m.get('reasoning_content') or m.get('reasoning') or ''
            text = (rs+'\n</think>\n\n' if rs else '')+(m.get('content') or '')
            return SimpleNamespace(outputs=[SimpleNamespace(text=text, finish_reason=r['choices'][0]['finish_reason'], token_ids=range(r['usage']['completion_tokens']))],
                                   prompt_token_ids=range(r['usage']['prompt_tokens']))
        import os
        with ThreadPoolExecutor(int(os.environ.get('RETICLE_API_WORKERS', min(64, 8*len(self.ports))))) as ex:  # concurrent requests across the servers
            return list(ex.map(one, [(c, next(cyc)) for c in convs]))


def prompt_text(q, ev_text):
    return q['question'], '\n'.join(q['options']), ('\n\n'+ev_text if ev_text else '')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arms', required=True); p.add_argument('--shard', type=int, required=True); p.add_argument('--nshards', type=int, required=True)
    p.add_argument('--split', default='test'); p.add_argument('--limit', type=int, default=None); p.add_argument('--qa_dir', default=None, help='write answers here instead of runs/<set>/qa')
    p.add_argument('--model', default=MODEL, help='answer model (Hugging Face id or local snapshot)')
    p.add_argument('--api_ports', default='', help='use loopback OpenAI-compatible vLLM servers of the same model instead of an in-process engine')
    a = p.parse_args()
    arms = json.loads(Path(a.arms).read_text())
    path = lambda arm: (Path(a.qa_dir) if a.qa_dir else RUNS[arm['set']]/'qa')/f"{arm['name']}.{arm.get('split', a.split)}.{a.shard}.jsonl"
    todo = [arm for arm in arms if not path(arm).exists()]
    if not todo:
        return
    if a.api_ports:  # the same VLM behind loopback OpenAI-compatible vLLM servers
        llm = APILLM([int(x) for x in a.api_ports.split(',')])
        sp = lambda arm, n: dict(temperature=1.0, top_p=0.95, top_k=20, max_tokens=n, seed=arm.get('seed', 20261018))
    else:
        from vllm import LLM, SamplingParams
        llm = LLM(model=a.model, revision=MODEL_REVISION if a.model == 'Qwen/Qwen3.8-27B' else None, max_model_len=24576, gpu_memory_utilization=0.88,
                  limit_mm_per_prompt={'image': 8}, seed=20261018, enable_prefix_caching=True)
        sp = lambda arm, n: SamplingParams(temperature=1.0, top_p=0.95, top_k=20, max_tokens=n, seed=arm.get('seed', 20261018))  # 'seed': replicate arms
    kw = dict(enable_thinking=True, reasoning_effort='low')
    for arm in todo:
        if path(arm).exists():
            continue
        began = time.monotonic(); s = arm['set']; run = RUNS[s]; path(arm).parent.mkdir(parents=True, exist_ok=True)
        split = arm.get('split', a.split)  # an arm may name its own split (dev sets)
        qs = read_jsonl(run/f'inputs/queries_{s}_{split}.jsonl')[a.shard::a.nshards][:a.limit]
        ev = {r['id']: r for r in read_jsonl(run/f"evidence_{s}_{split}_{arm['evidence']}.jsonl")} if arm.get('evidence') else {}
        items = [Item(q, s, arm.get('overview_px', 1024)) for q in qs]
        routed = [None]*len(items)
        if arm.get('route'):  # the agent chooses the expert from the overview and the question
            from reticle.agent import router
            assert a.api_ports, 'routing calls the VLM server (--api_ports)'
            res = llm.chat([router.conversation(it.overview, router.request_of(s, it.q)) for it in items], router.PARAMS, chat_template_kwargs=router.CHAT_KW)
            routed = [router.parse(r.outputs[0].text) for r in res]
        keep = [r is None or r == {'c16': 'metastasis', 'mdg': 'mitosis', 'u8': 'polyp', 'u7': 'polyp'}[s] for r in routed]
        crop_px, fine_px, nc, nf = arm.get('crop_px', 768), arm.get('fine_px', 768), arm.get('n_coarse', 3), arm.get('n_fine', 3)
        levels, mode = arm.get('levels', 0), arm.get('crops', 'none')
        coarse = [[] for _ in items]; fine = [[] for _ in items]
        if levels == 2 and mode == 'expert':  # coarse windows
            cw = [it.units(arm['coarse_um']) for it in items]
            for i, it in enumerate(items):
                coarse[i] = [] if not keep[i] else [window((b[0]+b[2])/2, (b[1]+b[3])/2, *cw[i]) for b in ev[it.q['id']]['coarse'][:nc]]
        coarse_ims = [[crop(it.native, b, crop_px) for b in coarse[i]] for i, it in enumerate(items)]

        def head(i, it):  # labelled images shown so far: overview + coarse windows
            content = [dict(type='text', text=f'Image 1: the whole region ({fmt_um(it.w_um)} wide) downsampled to {it.ov_mpp:.1f} um/px.'), dict(type='image_pil', image_pil=it.overview)]
            for k, (b, (im, sc)) in enumerate(zip(coarse[i], coarse_ims[i]), 2):
                content += [dict(type='text', text=f'Image {k}: window {[round(v) for v in b]} of image 1 ({fmt_um(arm["coarse_um"])} wide) at {it.mpp*sc:.1f} um/px.'),
                            dict(type='image_pil', image_pil=im)]
            return content
        if levels >= 1 and mode == 'expert':  # fine fields
            fu = [it.units(fine_px*it.mpp) for it in items]
            for i, it in enumerate(items):
                fine[i] = [] if not keep[i] else [window((b[0]+b[2])/2, (b[1]+b[3])/2, *fu[i]) for b in ev[it.q['id']]['fine'][:nf]]
        convs = []
        for i, it in enumerate(items):
            content = head(i, it); k0 = 2+len(coarse[i])
            for k, b in enumerate(fine[i], k0):
                im, sc = crop(it.native, b, crop_px)
                content += [dict(type='text', text=f'Image {k}: field {[round(v) for v in b]} of image 1 ({fmt_um(im.width*sc*it.mpp)} wide) at full resolution, {it.mpp*sc:.2f} um/px.'),
                            dict(type='image_pil', image_pil=im)]
            pt = prompt_text(it.q, ev[it.q['id']]['text'] if ev and keep[i] else '')
            content.append(dict(type='text', text=f'Question: {pt[0]}\nOptions:\n{pt[1]}{pt[2]}\n\nThink it through, then end with a final line exactly of the form "Answer: <letter>".'))
            convs.append([dict(role='system', content=SYSTEM), dict(role='user', content=content)])
        res = llm.chat(convs, sp(arm, 8192), chat_template_kwargs=kw, use_tqdm=False)
        rows = []
        for i, (it, r) in enumerate(zip(items, res)):
            o = r.outputs[0]; n_img = 1+len(coarse[i])+len(fine[i])
            rows.append(dict(id=it.q['id'], arm=arm['name'], choice=parse(o.text, it.q['options']), raw=o.text[-1500:], finish=o.finish_reason, prompt_tokens=len(r.prompt_token_ids),
                             output_tokens=len(o.token_ids), coarse=coarse[i], fine=fine[i], images=n_img, **({'routed': routed[i]} if routed[i] else {})))
        tmp = path(arm).with_suffix('.tmp'); tmp.write_text(''.join(json.dumps(x)+'\n' for x in rows)); tmp.rename(path(arm))
        print(json.dumps(dict(arm=arm['name'], shard=a.shard, n=len(rows), seconds=round(time.monotonic()-began), unparsed=sum(x['choice'] is None for x in rows),
                              truncated=sum(x['finish'] == 'length' for x in rows))), flush=True)


if __name__ == '__main__':
    main()
