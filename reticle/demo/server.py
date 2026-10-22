"""RETICLE demo: a multi-turn pathology consultation in the browser (reticle.agent.consult, the pipeline evaluated on MultiCaRe-Dx).

Start a conversation from an uploaded image (or a file of the --inputs folder), a published case figure (its own article is excluded from
the evidence), or a large-image benchmark example with a RETICLE expert's reading and views (reticle.demo.examples; the reference answer is
revealed on request). An uploaded breast-carcinoma high-power field or colorectal polyp region can be read live by the RETICLE mitosis or
polyp expert (reticle.experts.live, given its um/px). Then ask any number of follow-up questions: diagnosis and differential, work-up,
staging, treatment, follow-up. Every turn retrieves new evidence (published case reports, teaching captions, NCI PDQ summaries), evidence IDs
persist across turns, and every citation is a validated source sentence. The server binds to 127.0.0.1 only.
  python3 -m reticle.demo.server --port 8875 --llm-port 8000        (one GPU for the MUSK index; the VLM served by vLLM on --llm-port)"""
import argparse
import base64
import io
import json
from pathlib import Path
import re
import threading
import time
import uuid

from PIL import Image, ImageOps

from reticle.agent.consult import Library, Session
from reticle.agent import router
from reticle.paths import CONFIGS, DATA

BASE = DATA
EX = BASE/'runs/demo_v4/examples'
WEB = Path(__file__).resolve().parent/'web'
LIVE_SPEC = {  # live experts: (tool class name, benchmark set, case context, first question, source line)
    'mitosis': ('MitosisTool', 'mdg', 'Invasive breast carcinoma, H&E high-power region (uploaded image, live expert reading).',
                'How many mitotic figures are in this region, and where are they?', 'uploaded image, RETICLE mitosis expert run live'),
    'polyp': ('PolypTool', 'u8', 'Colorectal polyp biopsy, H&E, a region of up to 800 µm (uploaded image, live expert reading).',
              'What type of polyp is this? Is the dysplasia low or high grade?', 'uploaded image, RETICLE polyp expert (mixture of seven MUSK experts) run live')}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--port', type=int, default=8875); p.add_argument('--llm-port', type=int, default=8000)
    p.add_argument('--version', type=int, default=11, help='consultation version of the deep mode (reasoning; see reticle.agent.consult.Session)')
    p.add_argument('--fast-version', type=int, default=3, help='consultation version of the fast mode (no reasoning, about 30 s per turn)')
    p.add_argument('--cite-retries', type=int, default=1, help='rewrites requested on follow-up turns when an answer cites no evidence (each costs a full '
                   'generation); none on a first turn answered by an expert')
    p.add_argument('--terms-tokens', type=int, default=256, help='token budget of the search-term call (consult.search_terms; the benchmark runs used 80)')
    p.add_argument('--no-concise', dest='concise', action='store_false', help='answers in the benchmark style (one paragraph) instead of a conclusion + short lines')
    p.add_argument('--cite-check', type=int, default=2, choices=[1, 2], help='citation check: 2 = clause-level (consult._check_citations_v2); '
                   '1 = the check of the benchmark runs (whole sentence, source cut at 400 characters)')
    p.add_argument('--inputs', default=str(BASE/'runs/demo_v4/inputs'), help='folder offered by "Open from folder" (images only, read-only)'); a = p.parse_args()
    lib = Library('cuda'); ix = lib.ix
    html = WEB/'index.html'; log_dir = BASE/'runs/demo_v4/sessions'; log_dir.mkdir(parents=True, exist_ok=True)
    by_file = {r['file_id']: r for r in ix.case_images}
    published = json.loads((CONFIGS/'demo_published_examples.json').read_text())['file_ids']
    large = {m['id']: m for m in (json.loads((d/'meta.json').read_text()) for d in sorted(EX.iterdir()) if (d/'meta.json').exists())} if EX.is_dir() else {}
    sessions, lock = {}, threading.Lock()
    tools, tool_lock, live = {}, threading.Lock(), {}  # live experts (loaded on first use) and their per-session outputs
    LIVE = BASE/'runs/demo_v4/live'; LIVE.mkdir(parents=True, exist_ok=True)
    INPUTS, thumbs = Path(a.inputs), {}

    def input_files():  # the images of the input folder, by name (the only names /inputs/ serves)
        if not INPUTS.is_dir():
            return {}
        return {f.name: f for f in sorted(INPUTS.iterdir()) if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tif', '.tiff')}

    def live_expert(kind, image, um_per_px, sid):
        from reticle.experts import live as experts
        cls, s, context, question, source = LIVE_SPEC[kind]
        with tool_lock:
            tool = tools.setdefault(kind, getattr(experts, cls)()); r = tool.run(image, um_per_px)
        d = LIVE/sid; d.mkdir(exist_ok=True); r['overview'].save(d/'overview.png')
        for k, c in enumerate(r['crops']):
            c.save(d/f'crop_{k}.png')
        meta = dict(id=sid, set=s, live=True, context=context, question=question, expert_text=r['text'], boxes=r['boxes'],
                    options=r['options'], probs=r['probs'], reference=None, source=source)
        live[sid] = meta
        return meta, r['overview'], r['crops']

    def summary(meta):   # one readable line for the page: the expert's calibrated top option (the full expert text stays one click away)
        probs = meta.get('probs') or meta.get('expert_probs'); opts = meta.get('options') or []
        if not probs or len(probs) != len(opts):
            return ''
        k = max(range(len(probs)), key=lambda i: probs[i]); top = opts[k].split(') ', 1)[-1]
        m = re.search(r'extent (\d+) um', meta['expert_text']); n = re.search(r'at the reader threshold [\d.]+: (\d+)', meta['expert_text'])
        lead = f'largest tumour cluster {m.group(1)} µm · ' if m else (f'{n.group(1)} mitotic figures counted · ' if n else '')
        return f'{lead}calibrated reading: {top} ({100*probs[k]:.0f}%)'

    def evidence_view(e):
        v = {k: e.get(k) for k in ('evidence_id', 'kind', 'title', 'similarity', 'match', 'source_url', 'article_id', 'case_id', 'caption', 'turn', 'sentences')}
        v['image'] = f"/case_image/{e['file_id']}" if e.get('file_id') else (f"/library_image/{e['library_index']}" if e.get('library_index') is not None else None)
        return v

    class Handler(__import__('http.server').server.BaseHTTPRequestHandler):
        def send(self, status, body, ctype='application/json; charset=utf-8'):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status); self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store'); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            path = self.path.split('?', 1)[0]
            if path == '/':
                return self.send(200, html.read_bytes(), 'text/html; charset=utf-8')
            if path == '/api/info':
                return self.send(200, dict(cases=len(ix.case_text), case_images=len(ix.case_images), library=len(ix.lib),
                                           published=[dict(file_id=f, caption=by_file[f]['caption']) for f in published],
                                           large=[dict(id=m['id'], set=m['set'], context=m['context'], question=m['question'],
                                                       source=m['source'], crops=len(m['boxes'])) for m in large.values()]))
            if path.startswith('/case_image/') and path[12:] in by_file:
                return self.send(200, Path(by_file[path[12:]]['image_path']).read_bytes(), 'image/png')
            if path.startswith('/library_image/') and path[15:].isdigit() and int(path[15:]) < len(ix.lib):
                return self.send(200, Path(ix.lib[int(path[15:])]['image_path']).read_bytes(), 'image/jpeg')
            if path.startswith('/large/'):
                parts = path.split('/')
                if len(parts) == 4 and parts[3] in {'overview.png', 'crop_0.png', 'crop_1.png', 'crop_2.png'}:
                    root = EX if parts[2] in large else LIVE if parts[2] in live else None
                    if root and (root/parts[2]/parts[3]).exists():
                        return self.send(200, (root/parts[2]/parts[3]).read_bytes(), 'image/png')
            if path.startswith('/anim/'):  # three.js animations of the landing card (web/anim: html, js, png, ttf only)
                rel = path[len('/anim/'):]; f = (WEB/'anim'/rel).resolve(); ctypes = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript',
                                                                                      '.png': 'image/png', '.ttf': 'font/ttf'}
                if '..' not in rel and str(f).startswith(str(WEB/'anim')+'/') and f.is_file() and f.suffix in ctypes:
                    return self.send(200, f.read_bytes(), ctypes[f.suffix])
            if path == '/api/files':
                rows = []
                for n, f in input_files().items():
                    with Image.open(f) as im:
                        rows.append(dict(name=n, bytes=f.stat().st_size, width=im.width, height=im.height, modified=time.strftime('%Y-%m-%d %H:%M', time.localtime(f.stat().st_mtime))))
                return self.send(200, dict(folder=f'…/{INPUTS.parent.name}/{INPUTS.name}', files=rows))
            if path.startswith('/inputs/'):
                parts = path.split('/'); files = input_files()
                if len(parts) == 3 and parts[2] in files:
                    return self.send(200, files[parts[2]].read_bytes(), 'image/png' if parts[2].lower().endswith('.png') else 'image/jpeg')
                if len(parts) == 4 and parts[2] == 'thumb' and parts[3] in files:
                    if parts[3] not in thumbs:
                        im = Image.open(files[parts[3]]).convert('RGB'); im.thumbnail((160, 160)); buf = io.BytesIO(); im.save(buf, 'JPEG', quality=85); thumbs[parts[3]] = buf.getvalue()
                    return self.send(200, thumbs[parts[3]], 'image/jpeg')
            if path.startswith('/api/reveal/') and path[12:] in sessions:
                s = sessions[path[12:]]
                return self.send(200, dict(reference=(s['meta'] or {}).get('reference')))
            return self.send(404, dict(error='not found'))

        def do_POST(self):
            began = time.monotonic()
            try:
                req = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path == '/api/start':
                    image, exclude, meta, crops = None, set(), None, []
                    context = req.get('context', '').strip()[:800]
                    if req.get('large_example') in large:
                        meta = large[req['large_example']]; image = Image.open(EX/meta['id']/'overview.png').convert('RGB')
                        crops = [Image.open(EX/meta['id']/f'crop_{k}.png').convert('RGB') for k in range(len(meta['boxes'])) if (EX/meta['id']/f'crop_{k}.png').exists()]
                        context = context or meta['context']
                    elif req.get('example_file_id') in by_file:
                        row = by_file[req['example_file_id']]; exclude = {row['article_id']}; image = Image.open(row['image_path']).convert('RGB')
                    elif req.get('image_base64'):
                        image = ImageOps.exif_transpose(Image.open(io.BytesIO(base64.b64decode(req['image_base64'])))).convert('RGB')
                    sid = uuid.uuid4().hex[:12]; tool, routed = req.get('expert_tool'), None
                    if tool == 'auto' and image is not None and not meta:  # the router picks the expert (or none) from the image and the case information
                        tool = routed = router.route(a.llm_port, image, context or router.GENERIC)
                    if tool in LIVE_SPEC and image is not None and not meta:
                        mpp = float(req.get('um_per_px') or (0.25 if tool == 'mitosis' else 0.44))
                        if not 0.1 <= mpp <= 2.0:
                            raise ValueError('µm/px must be between 0.1 and 2.0.')
                        meta, image, crops = live_expert(tool, image, mpp, sid); context = context or meta['context']
                        if routed:
                            meta['source'] += ', chosen by the router'
                    s = Session(lib, a.llm_port, image, context=context, retrieval=True, exclude=exclude, version=a.version if req.get('think') else a.fast_version,
                                cite_retries=a.cite_retries, terms_tokens=a.terms_tokens, concise=a.concise, cite_check=a.cite_check,
                                expert_text=(meta['expert_text']+'\nCategories used by the expert'+('' if meta.get('live') else ' and the reference')+' (option letters): '+' | '.join(meta['options'])) if meta else '',
                                expert_images=crops)
                    with lock:
                        sessions[sid] = dict(session=s, meta=meta, created=time.time())
                        for k in [k for k, v in sessions.items() if time.time()-v['created'] > 4*3600]:
                            sessions.pop(k, None)
                    return self.send(200, dict(session_id=sid, routed=routed, large=dict(id=meta['id'], expert_text=meta['expert_text'], summary=summary(meta), crops=len(crops), boxes=meta['boxes'],
                                                                          question=meta['question'], source=meta['source'], live=bool(meta.get('live'))) if meta else None))
                if self.path == '/api/ask':
                    entry = sessions.get(req.get('session_id', ''))
                    if not entry:
                        raise ValueError('The session has expired; please start again.')
                    question = req.get('question', '').strip()[:2000]
                    if not question:
                        raise ValueError('Please enter a question.')
                    s = entry['session']
                    with entry.setdefault('lock', threading.Lock()):
                        s.cite_retries = 0 if (s.expert_text and not s.turns) else a.cite_retries
                        turn = s.ask(question) if s.think else s.ask(question, max_tokens=2000)  # fast mode: caps a runaway answer before the concise retry
                    ev = {e['evidence_id']: e for e in s.evidence}
                    resp = dict(turn=len(s.turns), question=question, answer=turn['answer'], search_terms=turn['search_terms'],
                                new_evidence=[evidence_view(ev[i]) for i in turn['new_evidence']], seconds=round(time.monotonic()-began, 1))
                    (log_dir/f"{req['session_id']}.json").write_text(json.dumps(dict(meta=entry['meta']['id'] if entry['meta'] else None,
                                                                                     turns=[dict(question=t['question'], answer=t['answer']) for t in s.turns]), ensure_ascii=False))
                    return self.send(200, resp)
                return self.send(404, dict(error='not found'))
            except Exception as exc:
                return self.send(400, dict(error=str(exc)))

        def log_message(self, *args):
            pass

    print(f'RETICLE demo on 127.0.0.1:{a.port}', flush=True)
    __import__('http.server').server.ThreadingHTTPServer(('127.0.0.1', a.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
