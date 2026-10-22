"""Convert an original-format LLaVA-1.5 checkpoint (LlavaLlamaForCausalLM: model.*, model.mm_projector.*, lm_head) into the transformers
LlavaForConditionalGeneration format that vLLM loads, following transformers' convert_llava_weights_to_hf.py: language model keys under
language_model, projector under multi_modal_projector, the frozen CLIP vision tower from its own checkpoint, an added <image> token and a
<pad> token with the embeddings padded to 32064 (new rows = mean of the old ones). Used for the Quilt-LLaVA baseline.
  python -m reticle.evaluation.baselines.convert_llava_to_hf <llava repo dir> <clip vision repo dir> <out dir>"""
import glob
import json

import torch
from safetensors.torch import load_file
from transformers import (AddedToken, AutoTokenizer, CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel, LlamaConfig, LlavaConfig,
                          LlavaForConditionalGeneration, LlavaProcessor)


def load_state(src):
    sd = {}
    for f in sorted(glob.glob(f'{src}/*.safetensors')):
        sd.update(load_file(f))
    for f in sorted(glob.glob(f'{src}/pytorch_model*.bin'))+sorted(glob.glob(f'{src}/mm_projector.bin')):
        sd.update(torch.load(f, map_location='cpu', weights_only=True))
    return sd


def fix_vision_keys(out):
    """transformers >= 5 saves the CLIP tower as [model.]vision_tower.<embeddings|encoder|...>; vLLM's Llava loader expects the older
    [model.]vision_tower.vision_model.<...> layout, so the tower keys are rewritten in place."""
    from safetensors.torch import save_file

    def new_key(k):
        for pre in ('model.vision_tower.', 'vision_tower.'):
            if k.startswith(pre) and not k.startswith(pre+'vision_model.'):
                return pre+'vision_model.'+k[len(pre):]
        return k
    for f in sorted(glob.glob(f'{out}/*.safetensors')):
        sd = load_file(f); moved = {k: new_key(k) for k in sd if new_key(k) != k}
        if moved:
            save_file({moved.get(k, k): v for k, v in sd.items()}, f, metadata={'format': 'pt'}); print('renamed', len(moved), 'vision keys in', f, flush=True)
    for idx in glob.glob(f'{out}/model.safetensors.index.json'):
        d = json.load(open(idx)); d['weight_map'] = {new_key(k): v for k, v in d['weight_map'].items()}; json.dump(d, open(idx, 'w'), indent=2)


def main(src, clip, out):
    cfg = json.load(open(f'{src}/config.json'))
    tok = AutoTokenizer.from_pretrained(src, use_fast=False)
    tok.add_tokens(AddedToken('<image>', special=True, normalized=False), special_tokens=True); tok.add_special_tokens({'pad_token': '<pad>'})
    text_cfg = LlamaConfig(**{k: v for k, v in cfg.items() if k in LlamaConfig().to_dict() and k != 'architectures'}, architectures=['LlamaForCausalLM'])
    vis_cfg = CLIPVisionConfig.from_pretrained(clip)
    conf = LlavaConfig(text_config=text_cfg, vision_config=vis_cfg, image_token_index=tok.convert_tokens_to_ids('<image>'), pad_token_id=tok.pad_token_id,
                       vision_feature_layer=cfg.get('mm_vision_select_layer', -2), vision_feature_select_strategy='default', projector_hidden_act='gelu')
    with torch.device('meta'):
        model = LlavaForConditionalGeneration(conf)
    keys = set(model.state_dict()); sd = load_state(src); new = {}
    lm = 'model.language_model.' if 'model.language_model.embed_tokens.weight' in keys else 'language_model.model.'   # transformers >= 4.52 vs older
    head = 'lm_head.weight' if 'lm_head.weight' in keys else 'language_model.lm_head.weight'
    proj = 'model.multi_modal_projector.' if any(k.startswith('model.multi_modal_projector.') for k in keys) else 'multi_modal_projector.'
    vt = 'model.vision_tower.' if any(k.startswith('model.vision_tower.') for k in keys) else 'vision_tower.'
    for k, v in sd.items():
        if 'vision_tower' in k or k.endswith('rotary_emb.inv_freq'):
            continue  # the frozen CLIP tower is taken from its own checkpoint; rotary buffers are recomputed
        if k.startswith('model.mm_projector.'):
            idx = int(k.split('.')[2]); new[f"{proj}linear_{1 if idx == 0 else 2}.{k.split('.')[-1]}"] = v
        elif k.startswith('model.'):
            new[lm+k[len('model.'):]] = v
        elif k == 'lm_head.weight':
            new[head] = v
    vis = CLIPVisionModel.from_pretrained(clip, dtype=torch.float16).state_dict()
    new.update({vt+k: v for k, v in vis.items()})
    unknown = [k for k in new if k not in keys]
    assert not unknown, unknown[:5]
    model.load_state_dict(new, strict=False, assign=True)
    emb = model.get_input_embeddings().weight.data; n_old = emb.shape[0]; mu = emb.float().mean(0)
    model.resize_token_embeddings(32064)
    for w in (model.get_input_embeddings().weight, model.get_output_embeddings().weight):
        w.data[n_old:] = mu.to(w.dtype)
    proc = LlavaProcessor(image_processor=CLIPImageProcessor.from_pretrained(clip), tokenizer=tok, patch_size=vis_cfg.patch_size,
                          vision_feature_select_strategy='default',
                          chat_template="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. {% for m in messages %}{% if m['role'] == 'user' %}USER: {% for c in m['content'] %}{% if c['type'] == 'image' %}<image>\n{% else %}{{ c['text'] }}{% endif %}{% endfor %} {% else %}ASSISTANT: {{ m['content'] if m['content'] is string else m['content'][0]['text'] }}</s>{% endif %}{% endfor %}{% if add_generation_prompt %}ASSISTANT:{% endif %}")
    missing = [k for k, v in model.state_dict().items() if v.is_meta]
    assert not missing, missing[:5]
    model.to(torch.float16).save_pretrained(out, safe_serialization=True); proc.save_pretrained(out)
    fix_vision_keys(out)
    print('saved', out, 'params', sum(p.numel() for p in model.parameters()), flush=True)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Convert an original-format LLaVA-1.5 checkpoint to the transformers Llava format.')
    ap.add_argument('src', help='LLaVA checkpoint directory (or, with --fix-only, a converted directory)')
    ap.add_argument('clip', nargs='?', help='CLIP vision tower directory'); ap.add_argument('out', nargs='?', help='output directory')
    ap.add_argument('--fix-only', action='store_true', help='only rename the vision-tower keys of an already converted checkpoint')
    a = ap.parse_args()
    fix_vision_keys(a.src) if a.fix_only else main(a.src, a.clip, a.out)
