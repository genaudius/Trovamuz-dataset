#!/usr/bin/env python3
"""
generate_trovamuz.py — Test generation for TrovaMUZ trained checkpoints
========================================================================
Genera 30 segundos de audio usando el modelo fine-tuneado.

Uso rápido (usa lm_final.pt por defecto):
  python generate_trovamuz.py

Especificar checkpoint:
  python generate_trovamuz.py --checkpoint models/lm_50.0.pt

Prompt personalizado:
  python generate_trovamuz.py --style amargue

Listar todos los checkpoints disponibles:
  python generate_trovamuz.py --list
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path("repositories/audiocraft")))
from audiocraft.models import MusicGen

# ── Prompts de referencia (basados en los captions del training) ──────────────
PROMPTS = {
    "bolero": (
        "musicgau_adn style, TrovaMUZ_V1 bachata romántica bolero, "
        "devotional romantic heartfelt, warm nylon-string requinto guitar lead "
        "with gentle vibrato, rhythmic guitar chord strumming, slap bass on beat 4, "
        "soft bongo open tones, gentle guira sixteenth notes, "
        "116 BPM, C major, intimate studio recording, high fidelity audio"
    ),
    "amargue": (
        "musicgau_adn style, TrovaMUZ_V1 bachata de amargue, "
        "melancholic bitter heartbreak, sharp requinto guitar with expressive bends, "
        "syncopated rhythm guitar, punching bongo martillo, bright guira, "
        "minor key dark emotional, 124 BPM, D minor, raw studio sound"
    ),
    "moderna": (
        "musicgau_adn style, TrovaMUZ_V1 bachata moderna romántica, "
        "dark romantic passionate, electric requinto guitar dark melodic lead, "
        "slap bass brooding syncopated groove, bongo deep open tones, "
        "guira steady sixteenth texture, 116 BPM, C minor, warm intimate studio mix"
    ),
    "instrumental": (
        "musicgau_adn style, TrovaMUZ_V1 bachata instrumental, "
        "authentic bolero bachata, high energy, bright syncopated guira, "
        "punching bongo martillo, driving tropical rhythm, "
        "warm nylon-string requinto, commercial studio mix"
    ),
}


def list_checkpoints(models_dir: Path) -> list:
    pts = sorted(models_dir.glob("lm_*.pt"))
    return pts


def generate(args):
    models_dir = Path("models")

    # ── --list ────────────────────────────────────────────────────────────────
    if args.list:
        ckpts = list_checkpoints(models_dir)
        if not ckpts:
            print("No checkpoints found in models/")
        else:
            print(f"\nCheckpoints disponibles en models/:")
            for c in ckpts:
                size_mb = c.stat().st_size / 1_048_576
                print(f"  {c.name:<25}  {size_mb:.0f} MB")
        return

    # ── Seleccionar checkpoint ────────────────────────────────────────────────
    if args.checkpoint:
        ckpt = Path(args.checkpoint)
    else:
        ckpt = models_dir / "lm_final.pt"
        if not ckpt.exists():
            # Fallback: último checkpoint numérico
            ckpts = list_checkpoints(models_dir)
            if not ckpts:
                print("ERROR: No hay checkpoints en models/. Ejecuta el training primero.")
                sys.exit(1)
            ckpt = ckpts[-1]
            print(f"lm_final.pt no encontrado — usando {ckpt.name}")

    if not ckpt.exists():
        print(f"ERROR: Checkpoint no encontrado: {ckpt}")
        sys.exit(1)

    # ── Prompt ────────────────────────────────────────────────────────────────
    if args.prompt:
        prompt = args.prompt
    else:
        prompt = PROMPTS.get(args.style, PROMPTS["bolero"])

    # ── Dispositivo ───────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Output path ───────────────────────────────────────────────────────────
    out_path = Path(args.output)

    print(f"\n{'='*58}")
    print(f"  TrovaMUZ Generator")
    print(f"{'='*58}")
    print(f"  Checkpoint : {ckpt}")
    print(f"  Model      : {args.model_id}")
    print(f"  Device     : {device}")
    print(f"  Duration   : {args.duration}s")
    print(f"  Style      : {args.style}")
    print(f"  Prompt     : {prompt[:80]}...")
    print(f"  Output     : {out_path}")
    print(f"{'='*58}\n")

    # ── Cargar modelo base ────────────────────────────────────────────────────
    print("Cargando modelo base MusicGen...")
    model = MusicGen.get_pretrained(args.model_id, device=device)

    # ── Cargar pesos entrenados ───────────────────────────────────────────────
    print(f"Cargando checkpoint {ckpt.name}...")
    state = torch.load(str(ckpt), map_location=device)
    model.lm.load_state_dict(state)
    model.lm.eval()

    # ── Parámetros de generación ──────────────────────────────────────────────
    model.set_generation_params(
        duration=args.duration,
        use_sampling=True,
        top_k=args.top_k,
        top_p=0.0,           # usar solo top_k
        temperature=args.temperature,
        cfg_coef=args.cfg_coef,
    )

    # ── Generar ───────────────────────────────────────────────────────────────
    print("Generando audio...")
    with torch.no_grad():
        wav = model.generate([prompt], progress=True)  # [1, 1, T]

    # ── Guardar ───────────────────────────────────────────────────────────────
    wav_np = wav[0, 0].cpu().numpy()
    wav_np = np.clip(wav_np, -1.0, 1.0).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), wav_np, model.sample_rate)

    duration_real = len(wav_np) / model.sample_rate
    print(f"\n✓ Audio guardado → {out_path}  ({duration_real:.1f}s @ {model.sample_rate}Hz)")
    print(f"  Abre el archivo para escuchar el resultado.\n")


def parse_args():
    p = argparse.ArgumentParser(
        description="TrovaMUZ — generación de audio con modelo entrenado",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", default=None,
                   help="Ruta al checkpoint .pt (default: models/lm_final.pt)")
    p.add_argument("--model_id", default="melody",
                   choices=["small", "medium", "large", "melody"],
                   help="Modelo base (debe coincidir con el que se entrenó)")
    p.add_argument("--style", default="bolero",
                   choices=list(PROMPTS.keys()),
                   help="Estilo de prompt predefinido")
    p.add_argument("--prompt", default=None,
                   help="Prompt personalizado (sobreescribe --style)")
    p.add_argument("--duration", type=float, default=30.0,
                   help="Duración en segundos")
    p.add_argument("--top_k", type=int, default=250)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--cfg_coef", type=float, default=3.0,
                   help="Classifier-free guidance (3-5 recomendado)")
    p.add_argument("--output", default="trovamuz_output.wav",
                   help="Archivo WAV de salida")
    p.add_argument("--list", action="store_true",
                   help="Listar checkpoints disponibles y salir")
    return p.parse_args()


if __name__ == "__main__":
    generate(parse_args())
