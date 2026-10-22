"""Teaching image-caption library: Quilt-1M and PathGen pairs searched by the agent (reticle.agent.library) and retrieved as PathMMU evidence,
with PathMMU's own images and sources left out.
Kept: PathGen pairs, and Quilt-1M rows whose image exists once in the release, not marked not_histology, with a caption of at least 5 words and
30 characters. Dropped: OpenPath rows from tweets that PathMMU uses (socialpath_mapping.json) or with a PathMMU image name, and every image
whose SHA-256 or perceptual hash (Hamming distance <= 4) matches a PathMMU question image. Exact duplicate images are merged (captions pooled,
the longest is shown). The PathMMU question images are extracted to DATA/datasets/pathmmu/qa_image_cache, which reticle.benchmarks.pathmmu
reads, so the PathMMU steps run in this order: reticle.library.captions -> reticle.benchmarks.pathmmu (questions, text decontamination) ->
reticle.library.encode -> reticle.library.retrieve_captions.
Inputs under DATA/datasets: pathmmu/{data.json, socialpath_mapping.json, images.zip}, quilt_1m/{the release CSV, image folders},
pathgen/{PathGen-2k_samples.json, pathgen_2k_output/}. Output: DATA/library/captions/corpus.jsonl (one record per image, ordered by
image_id = SHA-256: image_path, dataset, upstream_subset, source_group, captions, caption, phash, width, height) and corpus_audit.json.
Difference from the manifest of the reported runs: the train / validation split fields of an earlier retriever-training experiment are not
written.
  python -m reticle.library.captions      (CPU)"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import re
import zipfile

import imagehash
from PIL import Image

from reticle.image_io import configure_image_decoder
from reticle.paths import DATA

SRC = DATA/'datasets'
OUT = DATA/'library/captions'
CORPUS = OUT/'corpus.jsonl'


class BKTree:
    """Perceptual hashes under Hamming distance: query() yields the payloads within `radius` bits."""
    def __init__(self):
        self.root = None

    def add(self, value, payload):
        if self.root is None:
            self.root = [value, [payload], {}]; return
        node = self.root
        while True:
            d = (value ^ node[0]).bit_count()
            if d == 0:
                node[1].append(payload); return
            if d not in node[2]:
                node[2][d] = [value, [payload], {}]; return
            node = node[2][d]

    def query(self, value, radius=4):
        stack = [self.root] if self.root is not None else []
        while stack:
            node = stack.pop(); d = (value ^ node[0]).bit_count()
            if d <= radius:
                yield from node[1]
            stack.extend(c for k, c in node[2].items() if d-radius <= k <= d+radius)


def inspect(path):
    configure_image_decoder()
    try:
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        with Image.open(path) as im:
            im.load(); return dict(image_id=digest, phash=str(imagehash.phash(im.convert('RGB'))), width=im.width, height=im.height)
    except Exception as exc:
        return dict(error=type(exc).__name__)


def main():
    pm = SRC/'pathmmu'; audit, rejected = {}, Counter()
    data = json.loads((pm/'data.json').read_text())
    pm_names = {Path(x['img']).name for subsets in data.values() for xs in subsets.values() if isinstance(xs, list) for x in xs if isinstance(x, dict) and 'img' in x}
    pm_tweets = {str(x['tw_id']) for xs in json.loads((pm/'socialpath_mapping.json').read_text()).values() for x in xs if 'tw_id' in x}
    cache = pm/'qa_image_cache'; cache.mkdir(exist_ok=True)
    with zipfile.ZipFile(pm/'images.zip') as z:  # the PathMMU question images (pixels only; answers are never read here)
        seen = set()
        for m in z.infolist():
            name = Path(m.filename).name
            if m.is_dir() or name not in pm_names:
                continue
            assert name not in seen, ('ambiguous PathMMU image name', name); seen.add(name)
            if not (cache/name).exists() or (cache/name).stat().st_size != m.file_size:
                (cache/name).write_bytes(z.read(m))
    with ThreadPoolExecutor(8) as ex:
        pm_stats = [s for s in ex.map(inspect, sorted(cache.iterdir())) if 'error' not in s]
    pm_sha, pm_hash = {s['image_id'] for s in pm_stats}, BKTree()
    for s in pm_stats:
        pm_hash.add(int(s['phash'], 16), s['image_id'])
    audit.update(pathmmu_image_names=len(pm_names), pathmmu_images_hashed=len(pm_sha), pathmmu_tweets=len(pm_tweets))
    cands = {}
    pg = SRC/'pathgen'
    for r in json.loads((pg/'PathGen-2k_samples.json').read_text()):
        path = str(pg/'pathgen_2k_output'/(r['file_id']+'_'+'_'.join(map(str, r['position']))+'.png'))
        cands[path] = dict(image_path=path, dataset='pathgen', source_group='-'.join(r['wsi_id'].split('-')[:3]), captions=[r['caption']])  # TCGA patient
    quilt = SRC/'quilt_1m'; paths = defaultdict(list)
    for p in quilt.rglob('*'):
        if p.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            paths[p.name].append(p)
    with next(quilt.glob('*.csv')).open(newline='') as f:
        for r in csv.DictReader(f):
            name = Path(r['image_path']).name; cap = r['caption'].strip()
            tweet = re.fullmatch(r'(\d{14,20})_\d+\.(?:jpg|png|jpeg)', name)
            tweet = tweet.group(1) if tweet and r['subset'].lower() == 'openpath' else None
            why = ('missing_or_ambiguous_image' if len(paths.get(name, [])) != 1 else 'not_histology' if r.get('not_histology') in ('1', '1.0', 'True', 'true')
                   else 'short_caption' if len(cap.split()) < 5 or len(cap) < 30 else 'pathmmu_source_or_name' if tweet in pm_tweets or name in pm_names else None)
            if why:
                rejected[why] += 1; continue
            path = str(paths[name][0])
            c = cands.setdefault(path, dict(image_path=path, dataset='quilt_1m', upstream_subset=r['subset'],
                                            source_group='tweet:'+tweet if tweet else 'unknown:'+name, captions=[]))
            if cap not in c['captions']:
                c['captions'].append(cap)
    rows = sorted(cands.values(), key=lambda x: x['image_path'])
    with ThreadPoolExecutor(8) as ex:
        stats = list(ex.map(inspect, [r['image_path'] for r in rows]))
    kept = {}
    for r, s in zip(rows, stats):
        if 'error' in s:
            rejected['unreadable_image'] += 1; continue
        if s['image_id'] in pm_sha or next(pm_hash.query(int(s['phash'], 16)), None) is not None:
            rejected['pathmmu_visual_overlap'] += 1; continue
        r.update(s)
        if s['image_id'] in kept:  # the same image under several names: one record
            k = kept[s['image_id']]; k['captions'] = sorted(set(k['captions']+r['captions'])); rejected['exact_duplicate_merged'] += 1
        else:
            kept[s['image_id']] = r
    OUT.mkdir(parents=True, exist_ok=True)
    with CORPUS.open('w') as f:
        for r in sorted(kept.values(), key=lambda x: x['image_id']):
            f.write(json.dumps(dict(r, caption=max(r['captions'], key=len)), ensure_ascii=False)+'\n')
    audit.update(records=len(kept), by_dataset=dict(Counter(r['dataset'] for r in kept.values())), excluded=dict(rejected))
    (OUT/'corpus_audit.json').write_text(json.dumps(audit, indent=1)+'\n'); print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()
