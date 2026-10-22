"""Case library: the published case reports of MultiCaRe (Zenodo 10.5281/zenodo.20416562, version 3.0.1; open-access PMC articles,
CC BY / CC0), searched by the agent (reticle.agent.library, reticle.agent.consult) and sampled by reticle.benchmarks.multicare_dx.
  download  the release files of configs/library_cases.json into DATA/datasets/multicare (resumable, MD5 checked)
  build     DATA/library/cases/catalog.sqlite: table units (one unit per chunk of a case narrative, <= 2,400 characters with 200 characters
            of overlap, article title, PMC link) with the FTS5 index unit_fts (title, text; porter stemming);
            images.jsonl: pathology figures whose upstream case link is consistent (exactly one case_images.parquet record for the figure,
            naming the same case and article as captions_and_labels.csv), with the caption and a white-letterboxed 384 px PNG view
            (images/<file_id>.png); units.jsonl: the case-narrative units of the articles of these figures
  encode    MUSK vectors of the figure views -> cases_musk.npz (img, ok, file_id; one GPU)
The articles of configs/library_cases.json (excluded_articles) are left out of images.jsonl and units.jsonl; the full-text index keeps every
case, as for the reported runs. Differences from the catalog of the reported runs: its full-text table also held article abstracts, figure
captions, 200 PMC full texts, two NCI PDQ summaries and 10 TCGA reports (the agent searches case narratives only, so they changed only the
BM25 statistics), and it kept the original figure files.
  python -m reticle.library.cases download | build [--limit n] | encode      (CPU; encode: one GPU)"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import sqlite3
import urllib.request
import zipfile

import numpy as np
from PIL import Image, ImageOps

from reticle.paths import CONFIGS, DATA

CFG = json.loads((CONFIGS/'library_cases.json').read_text())
RAW = DATA/'datasets/multicare'
OUT = DATA/'library/cases'
URL = 'https://pmc.ncbi.nlm.nih.gov/articles/{}/'
SCHEMA = '''
CREATE TABLE units(id INTEGER PRIMARY KEY, unit_key TEXT UNIQUE, article_id TEXT NOT NULL, case_id TEXT, kind TEXT NOT NULL, title TEXT,
  text TEXT NOT NULL, source_url TEXT, license TEXT);
CREATE INDEX units_case ON units(case_id, kind);
CREATE VIRTUAL TABLE unit_fts USING fts5(title, text, content='units', content_rowid='id', tokenize='porter unicode61');'''


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, f in CFG['files'].items():
        path = RAW/name
        if path.exists() and path.stat().st_size == f['bytes'] and md5(path) == f['md5']:
            continue
        part = RAW/(name+'.part'); have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(f"https://zenodo.org/api/records/{CFG['zenodo_record']}/files/{name}/content", headers={'Range': f'bytes={have}-'} if have else {})
        with urllib.request.urlopen(req, timeout=120) as r, part.open('ab' if have and r.status == 206 else 'wb') as out:  # restart when the range is ignored
            shutil.copyfileobj(r, out, 8 << 20)
        assert part.stat().st_size == f['bytes'] and md5(part) == f['md5'], ('download mismatch', name)
        part.rename(path); print('downloaded', name, flush=True)


def chunks(text, size, overlap):
    """Exact spans of at most `size` characters, cut at a space in the second half of the span, consecutive spans overlapping by `overlap`."""
    start = 0
    while start < len(text):
        end = min(len(text), start+size)
        if end < len(text):
            cut = text.rfind(' ', start+size//2, end)
            end = cut if cut > start else end
        yield start, text[start:end]
        if end == len(text):
            break
        start = max(start+1, end-overlap)


def letterbox(image, size):
    image = ImageOps.exif_transpose(image).convert('RGBA')
    white = Image.new('RGBA', image.size, 'white'); white.alpha_composite(image)
    return ImageOps.pad(white.convert('RGB'), (size, size), method=Image.Resampling.LANCZOS, color='white')


def build(limit=None):
    import pyarrow.parquet as pq
    rows = lambda name: (r for b in pq.ParquetFile(RAW/name).iter_batches(batch_size=512) for r in b.to_pylist())
    meta = {r['article_id']: r['article_metadata'] for r in rows('metadata.parquet')}
    keep = set(sorted(meta)[:limit]) if limit else set(meta)  # --limit: the first n articles, for a quick check
    OUT.mkdir(parents=True, exist_ok=True); (OUT/'images').mkdir(exist_ok=True)
    (OUT/'catalog.sqlite').unlink(missing_ok=True); db = sqlite3.connect(OUT/'catalog.sqlite'); db.executescript(SCHEMA)
    case_article = {}
    for r in rows('cases.parquet'):
        aid = r['article_id']
        if aid not in keep:
            continue
        m = meta[aid]
        for c in r['cases'] or []:
            cid = c['case_id']; case_article[cid] = aid
            for start, text in chunks(c.get('case_text') or '', CFG['chunk_characters'], CFG['chunk_overlap']):
                if text.strip():
                    db.execute('INSERT INTO units(unit_key, article_id, case_id, kind, title, text, source_url, license) VALUES(?,?,?,?,?,?,?,?)',
                               (f'case:{cid}:{start}', aid, cid, 'case_text', m.get('title') or '', text, URL.format(aid), m.get('license')))
    db.execute("INSERT INTO unit_fts(unit_fts) VALUES('rebuild')"); db.commit()
    links = defaultdict(set)  # figure id -> distinct upstream records of the figure (article, case and figure fields)
    for r in rows('case_images.parquet'):
        if r['article_id'] in keep:
            for c in r['case_images'] or []:
                for im in c['case_image_list'] or []:
                    links[im['image_id']].add(json.dumps(dict(im, article_id=r['article_id'], case_id=c['case_id']), sort_keys=True, default=str))
    where = {}
    for z in sorted(RAW.glob('PMC*.zip')):
        with zipfile.ZipFile(z) as f:
            where.update({Path(i.filename).name: (z.name, i.filename) for i in f.infolist() if not i.is_dir()})
    excluded, picked, count = set(CFG['excluded_articles']), [], Counter()
    with (RAW/'captions_and_labels.csv').open(encoding='utf-8', newline='') as f:
        for im in csv.DictReader(f):
            cid = im['patient_id']; aid = case_article.get(cid)
            if aid is None or im['image_type'] != 'pathology':
                continue
            ls = [json.loads(x) for x in links.get(im['main_image'], ())]
            status = ('missing_figure' if not ls else 'ambiguous_figure' if len(ls) > 1 else
                      'consistent_upstream_link' if (ls[0]['case_id'], ls[0]['article_id']) == (cid, aid) else 'case_figure_mismatch')
            count[status] += 1
            if status != 'consistent_upstream_link' or aid in excluded:
                continue
            if im['file'] not in where:
                count['missing_archive_member'] += 1; continue
            archive, member = where[im['file']]
            picked.append(dict(file=im['file'], file_id=im['file_id'], article_id=aid, upstream_case_id=cid, verified_case_id=cid, case_id=cid,
                               figure_id=im['main_image'], image_type=im['image_type'], image_subtype=im['image_subtype'], license=im['license'],
                               link_status=status, archive=archive, member=member, caption=im['caption']))
    articles = {r['article_id'] for r in picked}; by_archive = defaultdict(list)
    for r in picked:
        by_archive[r['archive']].append(r)

    def views(item):
        archive, rs = item; done = []
        with zipfile.ZipFile(RAW/archive) as z:
            for r in rs:
                try:
                    with Image.open(io.BytesIO(z.read(r['member']))) as im:
                        im.load(); r['original_dimensions'] = list(im.size); v = letterbox(im, CFG['image_size'])
                except (OSError, ValueError, Image.DecompressionBombError):
                    continue
                p = OUT/'images'/f"{r['file_id']}.png"; v.save(p); r['image_path'] = str(p); done.append(r)
        return done
    with ThreadPoolExecutor(6) as ex:
        images = sorted((r for part in ex.map(views, sorted(by_archive.items())) for r in part), key=lambda r: r['file_id'])
    cols = ['id', 'unit_key', 'article_id', 'case_id', 'kind', 'title', 'text', 'source_url', 'license']
    units = [dict(zip(cols, u)) for u in db.execute(f"SELECT {', '.join(cols)} FROM units WHERE kind='case_text' ORDER BY id") if u[2] in articles]
    for name, rs in (('images.jsonl', images), ('units.jsonl', units)):
        (OUT/name).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rs))
    summary = dict(articles=len(keep), cases=len(case_article), case_units=db.execute('SELECT count(*) FROM units').fetchone()[0], pathology_figures=dict(count),
                   images=len(images), image_decode_failures=len(picked)-len(images), image_articles=len(articles), units=len(units), excluded_articles=sorted(excluded))
    db.close(); (OUT/'build_summary.json').write_text(json.dumps(summary, indent=1)+'\n'); print(json.dumps(summary), flush=True)


def encode():
    from reticle.encoders import musk_encoder
    from reticle.library.encode import encode_images
    rows = [json.loads(x) for x in open(OUT/'images.jsonl')]
    vec, ok = encode_images(musk_encoder(), [r['image_path'] for r in rows], 24)
    np.savez(OUT/'cases_musk.npz', img=vec, ok=np.array(ok), file_id=np.array([r['file_id'] for r in rows]))
    print(json.dumps(dict(images=len(rows), failed=len(ok)-sum(ok))), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['download', 'build', 'encode']); ap.add_argument('--limit', type=int, default=None, help='build: only the first n articles')
    a = ap.parse_args()
    {'download': download, 'build': lambda: build(a.limit), 'encode': encode}[a.step]()
