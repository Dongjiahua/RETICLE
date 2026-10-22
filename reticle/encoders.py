"""Frozen MUSK image/text encoder (the backbone of the evidence index and of the RETICLE experts).

The model record comes from configs/models.json (Hugging Face id, revision, file hashes). Weights are read from RETICLE_MUSK_PATH if set,
otherwise from the Hugging Face snapshot of id@revision (HF_HOME; downloaded on first use, or offline with HF_HUB_OFFLINE=1). The MUSK model
code is imported from MUSK_SRC (a checkout of https://github.com/lilab-stanford/MUSK; the Dockerfile puts it in /opt/MUSK), or from
third_party/MUSK in this repository if MUSK_SRC is unset. RETICLE_DEVICE=cpu runs the encoder on the CPU (default: cuda)."""
import json
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

from reticle.paths import CONFIGS

DEV = os.environ.get('RETICLE_DEVICE', 'cuda')


def model_record(name='musk'):
    """The pinned record of a model in configs/models.json."""
    return next(m for m in json.loads((CONFIGS/'models.json').read_text())['models'] if m['name'] == name)


def model_path(model):
    """Local directory of the pinned weights: RETICLE_MUSK_PATH, or the Hugging Face snapshot of id@revision."""
    if os.environ.get('RETICLE_MUSK_PATH'):
        return Path(os.environ['RETICLE_MUSK_PATH'])
    from huggingface_hub import snapshot_download
    try:  # already in the cache (works offline)
        return Path(snapshot_download(model['id'], revision=model['revision'], local_files_only=True))
    except Exception:
        return Path(snapshot_download(model['id'], revision=model['revision'], allow_patterns=list(model.get('files', []))))


class MuskTokenizer:
    """XLM-R SentencePiece mapping used by MUSK (independent of the Transformers tokenizer classes)."""
    bos_token_id = 0
    pad_token_id = 1
    eos_token_id = 2
    unk_token_id = 3

    def __init__(self, path):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor(model_file=str(path))
        assert self.sp.unk_id() == 0 and self.sp.bos_id() == 1 and self.sp.eos_id() == 2
        assert self.sp.get_piece_size() > 10000

    def encode(self, text, add_special_tokens=True):
        ids = [3 if i == 0 else i + 1 for i in self.sp.encode(text, out_type=int)]
        return [0] + ids + [2] if add_special_tokens else ids


class BaseEncoder:
    """MUSK encoder: `.model` is the MUSK network (images are fed to it directly by the callers); `encode_text(texts)` returns
    L2-normalised float32 text vectors (with the projection head)."""
    def __init__(self, model, root):
        assert model['family'] == 'musk', model['family']
        path = model_path(model)
        self.tokenizer = MuskTokenizer(path/'tokenizer.spm')
        sys.path.insert(0, os.environ.get('MUSK_SRC') or str(Path(root)/'third_party/MUSK'))
        from musk import modeling, utils  # noqa: F401  (registers the timm model)
        from timm.models import create_model
        from safetensors.torch import load_file
        self.model = create_model('musk_large_patch16_384')
        self.model.load_state_dict(load_file(str(path/'model.safetensors')), strict=True)
        self.model.to(DEV); self.musk_tokenize = utils.xlm_tokenizer
        self.model.eval()

    @torch.inference_mode()
    def encode_text(self, items):
        tokens = [self.musk_tokenize(item, self.tokenizer, max_len=100) for item in items]
        ids = torch.tensor([x[0] for x in tokens], device=DEV, dtype=torch.long)
        pad = torch.tensor([x[1] for x in tokens], device=DEV, dtype=torch.long)
        vectors = self.model(text_description=ids, padding_mask=pad, return_global=True, with_head=True, out_norm=False, ms_aug=False)[1]
        assert vectors.ndim == 2 and len(vectors) == len(items) and torch.isfinite(vectors).all()
        assert (vectors.float().norm(dim=-1) > 0).all()
        return F.normalize(vectors.float(), dim=-1)


def musk_encoder(root=None):
    """The pinned MUSK encoder (configs/models.json)."""
    from reticle.paths import REPO
    return BaseEncoder(model_record('musk'), root or REPO)
