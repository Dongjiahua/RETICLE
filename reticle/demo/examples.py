"""Large-image examples for the demo: benchmark test regions (shown as such; the reference answer is revealed after the agent answers) with
the RETICLE experts' precomputed readings. For each example: a <= 1,024 px overview, up to 3 full-resolution views at the expert's regions
(768 px; MIDOG-MS 384 px around candidate mitoses), the expert text the agent receives, and the reference label.
Experts: CAMELYON16-MS dense segmentation ('seg3'), MIDOG-MS cross-fitted detectors ('cv2'), UniToPatho-MS mixture ('moe').
  python3 -m reticle.demo.examples        (CPU) -> runs/demo_v4/examples/<id>/{overview.png, crop_k.png, meta.json}"""
import json

import numpy as np
from PIL import Image

from reticle.paths import DATA

Image.MAX_IMAGE_PIXELS = None
BASE = DATA
OUT = BASE/'runs/demo_v4/examples'
SETS = {
    'c16': dict(run='camelyon16_ms_v1', gold='camelyon16_ms_v1', split='c16_test', ev='evidence_c16_test_seg3.jsonl', contract='c16seg3_test.jsonl', crop=768,
                want=['C', 'B', 'D', 'A'], extra=['C16_test_110'],
                context='Sentinel lymph node from a patient with breast cancer, H&E, a 5 mm wide region.',
                question='Is there metastatic carcinoma in this lymph-node region? How large is the largest deposit, and which category is it?',
                labels={'A': 'no metastasis', 'B': 'isolated tumour cells (≤ 0.2 mm)', 'C': 'micrometastasis (0.2–2 mm)', 'D': 'macrometastasis (> 2 mm)'}),
    'mdg': dict(run='midog_ms_v1', gold='midog_ms_v1', split='mdg_test', ev='evidence_mdg_test_cv2.jsonl', contract='mdg3_test.jsonl', crop=384,
                want=['D', 'B'], context='Invasive breast carcinoma, H&E high-power region of about 0.8 × 0.6 mm.',
                question='How many mitotic figures are in this region, and where? What does this mean for the histological grade?',
                labels={'A': '0 mitotic figures', 'B': '1 mitotic figure', 'C': '2–3 mitotic figures', 'D': '≥ 4 mitotic figures'}),
    'u8': dict(run='unitopatho_ms_v1', gold='unitopatho_ms_v1', split='u8_test', ev='evidence_u8_test_moe.jsonl', contract='u8_test.jsonl', crop=768,
               want=['D', 'E'], context='Colorectal polyp biopsy, H&E, a region about 800 µm wide.', question='What type of polyp is this? Is the dysplasia low or high grade?',
               labels={'A': 'normal colonic mucosa', 'B': 'hyperplastic polyp', 'C': 'tubular adenoma, low grade', 'D': 'tubular adenoma, high grade',
                       'E': 'tubulovillous adenoma, low grade', 'F': 'tubulovillous adenoma, high grade'}),
}
SOURCE = {'c16': 'CAMELYON16-MS test (CC0)', 'mdg': 'MIDOG-MS test (MIDOG 2021, CC BY 4.0)', 'u8': 'UniToPatho-MS test'}


def crop(native, box, size):
    H, W = native.shape[:2]; cx, cy = (box[0]+box[2])/2000*W, (box[1]+box[3])/2000*H
    x0 = int(min(max(cx-size/2, 0), max(0, W-size))); y0 = int(min(max(cy-size/2, 0), max(0, H-size)))
    return Image.fromarray(np.ascontiguousarray(native[y0:y0+size, x0:x0+size]))


def main():
    OUT.mkdir(parents=True, exist_ok=True); index = []
    for s, c in SETS.items():
        run = BASE/'runs'/c['run']; gold = {g['id']: g for g in map(json.loads, open(BASE/f"scoring/{c['gold']}/gold.jsonl")) if g['split'] == c['split']}
        ev = {e['id']: e for e in map(json.loads, open(run/c['ev']))}; con = {r['id']: r for r in map(json.loads, open(BASE/'runs/contract_v1'/c['contract']))}
        q = {x['id']: x for x in map(json.loads, open(run/f"inputs/queries_{c['split']}.jsonl"))}
        picks = [next(i for i in sorted(ev) if gold[i]['answer'] == want and 'ABCDEF'[int(np.argmax(con[i]['probs']))] == want) for want in c['want']]  # the expert is right on these
        for pick in picks + c.get('extra', []):
            d = OUT/f'{s}_{pick.replace(":", "_")}'; d.mkdir(exist_ok=True)
            im = Image.open(q[pick]['image_path']).convert('RGB'); im.thumbnail((1024, 1024)); im.save(d/'overview.png')
            native = np.load(q[pick]['native'], mmap_mode='r') if q[pick].get('native') else np.asarray(Image.open(q[pick]['image_path']).convert('RGB'))
            boxes = (ev[pick].get('fine') or [r['box'] for r in con[pick]['rois']])[:3]
            for k, b in enumerate(boxes):
                cr = crop(native, b, c['crop']); cr = cr.resize((768, 768), Image.LANCZOS) if cr.width < 768 else cr; cr.save(d/f'crop_{k}.png')
            text = ev[pick]['text'].replace(' inside fields the tile classifiers call tumour with probability >= 0.00', '')  # the segmenter's reading has no tile gate
            meta = dict(id=d.name, set=s, item=pick, context=c['context'], question=c['question'], expert_text=text, boxes=boxes, options=q[pick]['options'],
                        reference=c['labels'][gold[pick]['answer']], reference_option=gold[pick]['answer'], expert_probs=con[pick]['probs'], source=SOURCE[s])
            (d/'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1)); index.append(dict(id=d.name, set=s, reference=meta['reference'], crops=len(boxes)))
            print(d.name, meta['reference'], len(boxes), flush=True)
    (OUT/'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()   # no options; --help shows the docstring
    main()
