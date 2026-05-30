#!/usr/bin/env python3
"""
generate_lora.py — Inferencia con adapter LoRA de TrovaMUZ
==========================================================
Uso:
  python generate_lora.py --adapter models/TrovaMUZ_V1_LoRA_v3/best
  python generate_lora.py --adapter_pt path/to/lora_adapter.pt --rank 32
  python generate_lora.py --adapter models/TrovaMUZ_V1_LoRA_v3/best --style amargue
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import math

sys.path.insert(0, str(Path("repositories/audiocraft")))
from audiocraft.models import MusicGen

PROMPTS = {
    "bachata": (
        "musicgau_adn style, TrovaMUZ_V1 Dominican bachata romantica, "
        "syncopated bicheo rhythm, requinto guitar with expressive bends, "
        "116 BPM, A minor, romantic melancholic mood"
    ),
    "bolero": (
        "musicgau_adn style, TrovaMUZ_V1 bolero romantico, "
        "warm nylon-string requinto guitar, gentle bongo, intimate strings, "
        "100 BPM, C major, heartfelt emotional"
    ),
    "salsa": (
        "musicgau_adn style, TrovaMUZ_V1 salsa tropical, "
        "clave rhythm, brass section, piano guajeo, tumbao bass, "
        "180 BPM, F major, high energy danceable"
    ),
    "merengue": (
        "musicgau_adn style, TrovaMUZ_V1 merengue tipico, "
        "accordion lead, tambora drum, guira, fast pambiche rhythm, "
        "160 BPM, G major, festive tropical"
    ),
}


class LoRALinear(nn.Module):
    def __init__(self, base, r=16, alpha=16.0, dropout=0.0):
        super().__init__()
        self.base = base
        self.r = r
        self.scale = alpha / r
        d_in, d_out = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.empty(r, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.base(x) + self.scale * (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T)


def apply_lora(model, r=16, alpha=16.0, targets=("out_proj", "linear1", "linear2")):
    for param in model.parameters():
        param.requires_grad_(False)
    for full_name, module in list(model.named_modules()):
        parts = full_name.split(".")
        if parts[-1] not in targets or not isinstance(module, nn.Linear):
            continue
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], LoRALinear(module, r=r, alpha=alpha))
    return model


def load_adapter(model, adapter_pt, r=32, alpha=32.0):
    apply_lora(model.lm, r=r, alpha=alpha)
    weights = torch.load(adapter_pt, map_location="cpu", weights_only=False)
    missing, unexpected = model.lm.load_state_dict(weights, strict=False)
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected[:3]}")
    print(f"[lora] Loaded {len(weights)} tensors from {adapter_pt}")
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default=None,
                   help="Directorio del adapter (contiene lora_adapter.pt)")
    p.add_argument("--adapter_pt", default=None,
                   help="Ruta directa al archivo lora_adapter.pt")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--alpha", type=float, default=32.0)
    p.add_argument("--model_id", default="melody")
    p.add_argument("--style", default="bachata", choices=list(PROMPTS.keys()))
    p.add_argument("--prompt", default=None)
    p.add_argument("--duration", type=float, default=15.0)
    p.add_argument("--top_k", type=int, default=250)
    p.add_argument("--cfg_coef", type=float, default=3.0)
    p.add_argument("--output", default="trovamuz_lora_output.wav")
    args = p.parse_args()

    # Resolver path del adapter
    if args.adapter_pt:
        adapter_pt = args.adapter_pt
    elif args.adapter:
        adapter_pt = str(Path(args.adapter) / "lora_adapter.pt")
    else:
        print("ERROR: usa --adapter <dir> o --adapter_pt <file>")
        sys.exit(1)

    if not Path(adapter_pt).exists():
        print(f"ERROR: no encontrado: {adapter_pt}")
        sys.exit(1)

    prompt = args.prompt or PROMPTS[args.style]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Blackwell SDPA patch (RTX 5070 Ti)
    try:
        from xformers import ops as _xops
        from torch.nn import functional as _F

        def _sdpa(q, k, v, attn_bias=None, p=0.0, scale=None, **kw):
            q_t, k_t, v_t = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
            is_causal, mask = False, None
            if attn_bias is not None:
                try:
                    from xformers.ops.fmha.attn_bias import LowerTriangularMask
                    if isinstance(attn_bias, LowerTriangularMask):
                        is_causal = True
                except Exception:
                    pass
            out = _F.scaled_dot_product_attention(q_t, k_t, v_t,
                attn_mask=mask, dropout_p=p, is_causal=is_causal)
            return out.transpose(1, 2)

        _xops.memory_efficient_attention = _sdpa
        print("[compat] xFormers → PyTorch SDPA")
    except Exception as e:
        print(f"[compat] patch skipped: {e}")

    print(f"\nCargando MusicGen '{args.model_id}' en {device}...")
    model = MusicGen.get_pretrained(args.model_id, device=device)

    print(f"Cargando LoRA adapter (rank={args.rank})...")
    model = load_adapter(model, adapter_pt, r=args.rank, alpha=args.alpha)
    model.lm = model.lm.to(device)
    model.lm.eval()

    CHUNK = 30
    overlap = 5

    print(f"\nPrompt: {prompt[:100]}...")

    chunks = []
    n_chunks = math.ceil(args.duration / CHUNK)

    with torch.no_grad():
        for chunk_num in range(n_chunks):
            this_duration = min(CHUNK, args.duration - chunk_num * CHUNK)
            print(f"\nGenerando chunk {chunk_num+1}/{n_chunks} ({this_duration}s)...")
            model.set_generation_params(
                duration=this_duration,
                use_sampling=True,
                top_k=args.top_k,
                cfg_coef=args.cfg_coef,
            )
            wav = model.generate([prompt], progress=True)
            chunks.append(wav[0, 0].cpu().numpy())

    final = np.clip(np.concatenate(chunks), -1.0, 1.0).astype(np.float32)
    sf.write(args.output, final, model.sample_rate)
    print(f"\n✓ Guardado → {args.output}  ({len(final)/model.sample_rate:.1f}s)")


if __name__ == "__main__":
    main()
