"""MultiCaRe-Dx: case-level consultation benchmark built from MultiCaRe case reports (open-access PMC articles, CC BY / CC0; the case
library of reticle.library.cases).
Item = one published case: an H&E figure (caption hidden) + a clinical presentation written from the case text WITHOUT the pathology result or
diagnosis -> (turn 1) diagnosis with a differential, (turn 2) confirmatory immunohistochemistry / molecular work-up, (turn 3) treatment and
follow-up reported for such cases. Gold for each turn is extracted from the same case text (title + caption for the diagnosis).
One case per article, sampled with a fixed seed from cases with an H&E figure and >= 800 characters of case text (articles of the demo
examples excluded); at answer time the item's
own article and near-duplicate figures (MUSK cosine >= 0.90 to the item figure) are removed from the evidence (reticle/evaluation/consultation.py).
Extraction and leak check: Qwen3.8-27B (no thinking) under a JSON schema; an item is kept only if the model marks it a single patient with a
histopathological diagnosis, the presentation does not name the diagnosis (model flag) and no diagnosis-specific word of the gold appears in
the presentation (lexical check, organ / generic words exempt).
Test: the first candidates (1,000 items); dev: candidates after --offset with test articles excluded (299 items, used for all design
choices).
  python -m reticle.benchmarks.multicare_dx --ports 8931,8932,8933 [--split dev] [--n 1000 --pool 1400]      (CPU + vLLM servers)
Outputs (under RETICLE_DATA): runs/mcdx_v1[/dev]/inputs/items.jsonl (model-visible: id, image_path, presentation) and
scoring/mcdx_v1/gold[_dev].jsonl."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
import random
import re
import threading
import urllib.request

from reticle.paths import CONFIGS, DATA

BASE = DATA
PREP = BASE/'library/cases'  # reticle.library.cases
RUN = BASE/'runs/mcdx_v1'
GOLD = BASE/'scoring/mcdx_v1/gold.jsonl'
SEED = 20261019
SCHEMA = {'type': 'object', 'additionalProperties': False,
          'required': ['single_patient', 'has_histopathology_diagnosis', 'presentation', 'presentation_reveals_diagnosis', 'final_diagnosis',
                       'diagnosis_aliases', 'organ', 'workup', 'treatment', 'follow_up'],
          'properties': {'single_patient': {'type': 'boolean'}, 'has_histopathology_diagnosis': {'type': 'boolean'},
                         'presentation': {'type': 'string'}, 'presentation_reveals_diagnosis': {'type': 'boolean'},
                         'final_diagnosis': {'type': 'string'}, 'diagnosis_aliases': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 5},
                         'organ': {'type': 'string'}, 'workup': {'type': 'string'}, 'treatment': {'type': 'string'}, 'follow_up': {'type': 'string'}}}
PROMPT = """You are preparing a clinical case for a diagnostic exercise. Read the published case report below.

Title: {title}

Case text:
{text}

Return JSON:
- single_patient: true if the text describes one patient.
- has_histopathology_diagnosis: true if the final diagnosis was made or confirmed by histopathology (biopsy or resection).
- presentation: 80-200 words in English describing ONLY what was known before the pathology result: age, sex, history, symptoms, examination,
  laboratory and imaging findings, and which tissue was sampled (e.g. "a biopsy of the mass was taken"). Do NOT name, hint at or paraphrase
  the final diagnosis, the tumour type, its histological features, immunohistochemistry, molecular results, treatment or outcome. Imaging
  descriptions must stay descriptive (e.g. "a 4 cm enhancing mass"), without radiological diagnoses that give the answer away.
- presentation_reveals_diagnosis: true if, despite your effort, the presentation still names or strongly implies the final diagnosis.
- final_diagnosis: the final (histopathological) diagnosis as a short disease name (e.g. "spindle cell oncocytoma of the adenohypophysis").
- diagnosis_aliases: up to 5 synonyms or accepted alternative names of that diagnosis.
- organ: the organ or site of the lesion (one to three words).
- workup: immunohistochemistry, special stains and molecular tests reported, with their results ("" if none).
- treatment: treatment given ("" if not reported).
- follow_up: outcome and follow-up reported ("" if not reported)."""
GENERIC = set('''tumor tumour tumors tumours cancer disease syndrome primary metastatic metastasis malignant benign lesion lesions cell cells type grade high
low left right with without of the and or in on to from a an mass masses neoplasm neoplasms neoplastic secondary multiple solitary giant large small
acute chronic stage invasive infiltrating poorly well moderately differentiated undifferentiated variant subtype associated related due like
patient case rare unusual recurrent recurrence carcinoma adenocarcinoma sarcoma lymphoma infection inflammatory inflammation abscess cyst cystic
nodule nodular lesion benign'''.split()) | set('''renal pulmonary gastric ovarian cutaneous hepatic pancreatic cardiac bone bones tissue tissues cervical rectal breast
testicular peritoneal splenic oral endometrial duodenal mediastinal intestinal mesenteric abdominal bilateral cerebellar cerebral intramedullary conjunctival
gastrointestinal bowel colonic colon endobronchial intracranial vulvar retroperitoneal esophageal oesophageal thyroid adrenal uterine uterus vaginal laryngeal tracheal
endotracheal nasal sinonasal sinus orbital ocular spinal vertebral pleural pericardial bladder vesical prostatic penile scrotal femoral tibial skin lymph nodal node
nodes ascending descending sigmoid transverse biliary gallbladder appendiceal anal ileal jejunal colorectal parotid salivary submandibular gingival lingual tongue
palatal pharyngeal nasopharyngeal oropharyngeal hypopharyngeal axillary inguinal cervicofacial facial scalp hepatobiliary pelvic paratesticular epididymal
mammary chest thoracic lumbar sacral sacrococcygeal soft right-sided left-sided upper lower anterior posterior concurrent synchronous metachronous light
kidney kidneys lung lungs liver stomach ovary ovaries heart pancreas spleen intestine rectum testis testes peritoneum brain spine mediastinum'''.split())


def words(s):
    return [w for w in re.findall(r'[a-z][a-z\-]+', s.lower()) if len(w) >= 4]


def leak(gold, presentation):
    """Diagnosis-specific words (not generic, not part of the organ name) that appear in the presentation."""
    organ = set(words(gold['organ'])); pres = set(words(presentation))
    specific = {w for name in [gold['final_diagnosis'], *gold['diagnosis_aliases']] for w in words(name)} - GENERIC - organ
    return sorted(specific & pres)


def chat(port, text, max_tokens=1500):
    body = dict(model='qwen', messages=[dict(role='user', content=text)], temperature=0.0, max_tokens=max_tokens,
                response_format=dict(type='json_schema', json_schema=dict(name='case', schema=SCHEMA)), chat_template_kwargs=dict(enable_thinking=False))
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(json.load(urllib.request.urlopen(req, timeout=600))['choices'][0]['message']['content'])


def candidates():
    rows = [json.loads(x) for x in open(PREP/'images.jsonl')]
    shown = {r['article_id'] for r in rows if r['file_id'] in set(json.loads((CONFIGS/'demo_published_examples.json').read_text())['file_ids'])}
    text, title = defaultdict(list), {}
    for line in open(PREP/'units.jsonl'):
        u = json.loads(line)
        if u['kind'] == 'case_text' and u['case_id']:
            text[u['case_id']].append((int(u['id']), u['text']))
        if u.get('title'):
            title.setdefault(u['article_id'], u['title'])
    figs = defaultdict(list)
    for r in rows:
        cid = r['verified_case_id'] or r['upstream_case_id']
        if r.get('image_subtype') == 'h&e' and len((r.get('caption') or '').strip()) > 30 and cid in text and r['article_id'] not in shown:
            figs[cid].append(r)
    out, seen = [], set()
    for cid in sorted(figs):
        t = ' '.join(x for _, x in sorted(text[cid]))
        art = figs[cid][0]['article_id']
        if len(t) >= 800 and title.get(art):
            out.append(dict(case_id=cid, article_id=art, title=title[art], text=t, figure=sorted(figs[cid], key=lambda r: r['file_id'])[0]))
    random.Random(SEED).shuffle(out)
    picked = []
    for c in out:  # one case per article
        if c['article_id'] not in seen:
            seen.add(c['article_id']); picked.append(c)
    return picked


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ports', required=True); ap.add_argument('--n', type=int, default=1000); ap.add_argument('--pool', type=int, default=1400)
    ap.add_argument('--split', default='test', choices=['test', 'dev'], help='dev: candidates after the first --offset (the test pool), ids MCDXD_*')
    ap.add_argument('--offset', type=int, default=1800); a = ap.parse_args()
    global RUN, GOLD
    if a.split == 'dev':
        RUN, GOLD = RUN/'dev', GOLD.parent/'gold_dev.jsonl'
    cands = candidates(); print('candidate cases (one per article)', len(cands), flush=True)
    prev = {json.loads(x)['case_id']: json.loads(x) for x in open(RUN/'extraction_all.jsonl')} if (RUN/'extraction_all.jsonl').exists() else {}
    if a.split == 'dev':
        test_articles = {json.loads(x)['article_id'] for x in open(BASE/'scoring/mcdx_v1/gold.jsonl')}
        cands = [c for c in cands[a.offset:] if c['article_id'] not in test_articles]; print('dev candidates after the test pool', len(cands), flush=True)
    ports = itertools.cycle(int(p) for p in a.ports.split(',')); lock = threading.Lock()
    (RUN/'inputs').mkdir(parents=True, exist_ok=True); GOLD.parent.mkdir(parents=True, exist_ok=True)

    def extract(c):
        with lock:
            port = next(ports)
        try:
            g = chat(port, PROMPT.format(title=c['title'], text=c['text'][:12000]))
        except Exception as exc:
            return dict(c, error=str(exc))
        return dict(c, extraction=g, leak_words=leak(g, g['presentation']))
    def get(c):  # reuse an earlier extraction of the same case (the leak check is recomputed)
        if c['case_id'] in prev and prev[c['case_id']].get('extraction'):
            g = prev[c['case_id']]['extraction']; return dict(c, extraction=g, leak_words=leak(g, g['presentation']))
        return extract(c)
    with ThreadPoolExecutor(12*len(a.ports.split(','))) as ex:
        done = list(ex.map(get, cands[:a.pool]))
    keep, reasons = [], defaultdict(int)
    for c in done:
        g = c.get('extraction')
        why = ('error' if g is None else 'not single patient' if not g['single_patient'] else 'no histopathology diagnosis' if not g['has_histopathology_diagnosis']
               else 'model flags leak' if g['presentation_reveals_diagnosis'] else 'lexical leak' if c['leak_words'] else 'short presentation' if len(g['presentation'].split()) < 40 else None)
        reasons[why or 'kept'] += 1
        if why is None and len(keep) < a.n:
            keep.append(c)
    items, gold = [], []
    for k, c in enumerate(keep):
        iid = {'test': f'MCDX_{k:04d}', 'dev': f'MCDXD_{k:04d}'}[a.split]; g = c['extraction']; f = c['figure']
        items.append(dict(id=iid, image_path=f['image_path'], presentation=g['presentation']))
        gold.append(dict(id=iid, split='mcdx_'+a.split, case_id=c['case_id'], article_id=c['article_id'], file_id=f['file_id'], title=c['title'], caption=f['caption'],
                         final_diagnosis=g['final_diagnosis'], diagnosis_aliases=g['diagnosis_aliases'], organ=g['organ'], workup=g['workup'],
                         treatment=g['treatment'], follow_up=g['follow_up'], wsi=c['article_id']))
    (RUN/'inputs/items.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in items)); GOLD.write_text(''.join(json.dumps(x)+'\n' for x in gold))
    (RUN/'extraction_all.jsonl').write_text(''.join(json.dumps({k: v for k, v in c.items() if k != 'text'})+'\n' for c in done))
    summary = dict(candidates=len(cands), extracted=len(done), reasons=dict(reasons), items=len(items), seed=SEED,
                   with_workup=sum(bool(g['workup']) for g in gold), with_treatment=sum(bool(g['treatment']) for g in gold), with_follow_up=sum(bool(g['follow_up']) for g in gold))
    (RUN/'build_summary.json').write_text(json.dumps(summary, indent=2)+'\n'); print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
