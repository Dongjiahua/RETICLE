"""Evidence index of the RETICLE agent: published case reports (MultiCaRe figures and case narratives) and teaching image-caption pairs.

Case figures and captions are searched with MUSK embeddings, case narratives with SQLite full-text search (catalog.sqlite).
Built by reticle.library.cases (case catalog and figure vectors), reticle.library.captions and reticle.library.encode (caption library)."""
from collections import defaultdict
import contextlib
import json
from pathlib import Path
import re
import sqlite3
import threading

import numpy as np
import torch

from reticle.encoders import musk_encoder
from reticle.library.encode import TRANSFORM
from reticle.paths import DATA

PREP = DATA/'library/cases'                          # case catalog: images.jsonl, units.jsonl, catalog.sqlite (reticle.library.cases)
LIB = DATA/'library/captions/corpus.jsonl'           # teaching image-caption pairs, Quilt-1M and PathGen (reticle.library.captions)
CASE_VECTORS = PREP/'cases_musk.npz'                 # MUSK embeddings of the case figures (reticle.library.cases encode)
CAPTION_BLOCKS = DATA/'runs/pathmmu_v3/musk'         # MUSK image/text embeddings of the caption library (block_*.npz)


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().split('\n') if x]


def sentences(text, limit=40):
    """Split a source text into the numbered sentences an answer can cite."""
    parts = [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z0-9(])', text.replace('\n', ' ')) if s.strip()]
    return parts[:limit]


class Index:
    def __init__(self, device):
        self.device = device
        self.enc = musk_encoder()
        # published case reports: figures linked to case narratives
        self.case_images = read_jsonl(PREP/'images.jsonl')
        cases = np.load(CASE_VECTORS)
        assert list(cases['file_id']) == [r['file_id'] for r in self.case_images]
        self.case_vec = torch.from_numpy(cases['img']).to(device)
        self.case_text = defaultdict(list); self.titles = {}
        for u in read_jsonl(PREP/'units.jsonl'):
            if u['kind'] == 'case_text' and u['case_id']:
                self.case_text[u['case_id']].append(u)
            self.titles.setdefault(u['article_id'], u['title'])
        # teaching image-caption library
        self.lib = read_jsonl(LIB)
        parts = [np.load(b) for b in sorted(CAPTION_BLOCKS.glob('block_*.npz'))]
        self.lib_img = torch.from_numpy(np.concatenate([p['img'] for p in parts])).to(device)
        self.lib_txt = torch.from_numpy(np.concatenate([p['txt'] for p in parts])).to(device)
        assert len(self.lib_img) == len(self.lib)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(f'file:{PREP}/catalog.sqlite?mode=ro', uri=True, check_same_thread=False)
        self.case_figure = {}
        for r in self.case_images:
            self.case_figure.setdefault(r['verified_case_id'] or r['upstream_case_id'], r)

    @torch.inference_mode()
    def encode(self, image):
        """MUSK embedding (float16, L2-normalised) of a letterboxed PIL image."""
        with self.lock:
            x = TRANSFORM(image).unsqueeze(0).to(self.device)
            cast = torch.autocast('cuda', dtype=torch.float16) if str(self.device).startswith('cuda') else contextlib.nullcontext()
            with cast:
                v = self.enc.model(image=x, return_global=True, with_head=True, out_norm=False, ms_aug=False)[0]
            return torch.nn.functional.normalize(v.float(), dim=-1).half()[0]
