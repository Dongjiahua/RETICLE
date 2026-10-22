"""PathMMU: freeze the val / test queries whose images are available, keep the answers apart (scoring/), and flag library captions
that share a 10-gram with any PathMMU question, option or explanation (text decontamination of the caption library; the image-level
decontamination is in reticle/library/retrieve_captions.py). Runs after reticle.library.captions, which extracts the question images to
datasets/pathmmu/qa_image_cache and builds the caption library.
  python -m reticle.benchmarks.pathmmu      (CPU)
Outputs (under RETICLE_DATA): runs/pathmmu_v3/inputs/{queries_val,queries_test,pathmmu_images}.jsonl, text_contaminated_image_ids.txt,
prepare_audit.json; scoring/pathmmu_v3/gold.jsonl."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from reticle.paths import DATA

BASE = DATA
PATHMMU = BASE/'datasets/pathmmu'
CORPUS = BASE/'library/captions/corpus.jsonl'


def dump(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def option_letters(options):
    letters = [re.match(r'^([A-Z])\)', o) for o in options]
    if not all(letters):
        raise ValueError('Unexpected option format')
    letters = [m.group(1) for m in letters]
    if len(set(letters)) != len(letters):
        raise ValueError('Duplicate option letters')
    return letters


def ngrams(text, n):
    words = re.findall(r'[a-z0-9]+', text.lower())
    return {' '.join(words[i:i+n]) for i in range(len(words)-n+1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=str(BASE/'runs/pathmmu_v3/inputs'))
    p.add_argument('--scoring', default=str(BASE/'scoring/pathmmu_v3'))
    p.add_argument('--ngram', type=int, default=10)
    a = p.parse_args()
    out, scoring = Path(a.out), Path(a.scoring)
    out.mkdir(parents=True, exist_ok=True); scoring.mkdir(parents=True, exist_ok=True)
    assert not (out/'queries_val.jsonl').exists(), 'refusing to overwrite frozen inputs'

    data = json.loads((PATHMMU/'data.json').read_text())
    queries, gold, images, available = {'val': [], 'test': []}, [], {}, Counter()
    leak_grams = set()
    for source in sorted(data):
        for split, rows in data[source].items():
            if not isinstance(rows, list):
                continue
            for ordinal, row in enumerate(rows):
                text = ' '.join([row['question'], *row['options'], row.get('answer', ''), row.get('explanation') or ''])
                leak_grams |= ngrams(text, a.ngram)
                image = PATHMMU/'qa_image_cache'/Path(row['img']).name
                if not image.is_file():
                    continue
                if str(image) not in images:
                    images[str(image)] = sha(image)
                if split not in queries:
                    continue
                qid = f'{source}:{split}:{ordinal}'
                letters = option_letters(row['options'])
                queries[split].append(dict(id=qid, source=source, split=split, image_path=str(image),
                                           image_sha256=images[str(image)], question=row['question'], options=row['options']))
                gold.append(dict(id=qid, source=source, split=split, image_sha256=images[str(image)],
                                 answer=letters[row['options'].index(row['answer'])]))
                available[(source, split)] += 1
    for split, rows in queries.items():
        dump(out/f'queries_{split}.jsonl', rows)
    dump(out/'pathmmu_images.jsonl', [dict(image_path=k, image_sha256=v) for k, v in sorted(images.items())])
    dump(scoring/'gold.jsonl', gold)

    flagged, total = [], 0
    with CORPUS.open() as f:
        for line in f:
            row = json.loads(line); total += 1
            caps = row.get('captions') or [row['caption']]
            if any(ngrams(c, a.ngram) & leak_grams for c in caps):
                flagged.append(row['image_id'])
    (out/'text_contaminated_image_ids.txt').write_text('\n'.join(flagged)+'\n')
    audit = dict(complete=True, queries={k: len(v) for k, v in queries.items()},
                 by_source_split={f'{s}:{sp}': n for (s, sp), n in sorted(available.items())},
                 pathmmu_available_images=len(images), corpus_records=total,
                 text_contaminated=len(flagged), ngram=a.ngram, leak_grams=len(leak_grams),
                 corpus_sha256=sha(CORPUS),
                 inputs_sha256={n: sha(out/n) for n in ['queries_val.jsonl', 'queries_test.jsonl', 'pathmmu_images.jsonl']},
                 gold_sha256=sha(scoring/'gold.jsonl'))
    (out/'prepare_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
