#!/usr/bin/env python3
"""
finetune_musicgen.py — TrovaMUZ_V1 LoRA Fine-Tuner
===================================================
Fine-tunes MusicGen Melody (1.5B) with low-rank adapters (LoRA) on a curated
Latin-music dataset with per-track enriched captions.

Architecture
------------
- LoRA rank=16, alpha=16 applied to out_proj + linear1 + linear2 in every
  transformer layer (~1.5 M trainable vs 1,500 M frozen params).
- Optional Demucs vocal separation before encoding (--demucs flag).
- Per-track enriched captions read from individual .txt files.
- bf16 autocast + gradient accumulation for H100 efficiency.
- Saves only the LoRA adapter weights (~50-100 MB), not the full model.

Usage
-----
  # Standard fine-tune (H100 recommended):
  python finetune_musicgen.py \\
      --dataset training/datasets/combined \\
      --model_id melody \\
      --epochs 5 \\
      --lr 1e-4 \\
      --batch_size 4 \\
      --grad_acc 4 \\
      --lora_rank 16 \\
      --output models/TrovaMUZ_V1_LoRA

  # With Demucs vocal separation:
  python finetune_musicgen.py --dataset training/datasets/combined --demucs

  # Quick smoke-test (CPU, 1 epoch):
  python finetune_musicgen.py --dataset training/datasets/bolero_bachata \\
      --model_id small --epochs 1 --batch_size 1 --grad_acc 1

Loading the adapter later
-------------------------
  from finetune_musicgen import load_lora_adapter
  model = MusicGen.get_pretrained('melody')
  model = load_lora_adapter(model, 'models/TrovaMUZ_V1_LoRA')
  # then use model.generate(...) normally
"""

import argparse
import contextlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

# ── Repo path setup ────────────────────────────────────────────────────────────
_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_root / "repositories" / "audiocraft"))
sys.path.insert(0, str(_root / "repositories"))

from audiocraft.models import MusicGen
from audiocraft.modules.conditioners import ClassifierFreeGuidanceDropout
from transformers import get_scheduler


# ══════════════════════════════════════════════════════════════════════════════
#  LoRA implementation
# ══════════════════════════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with low-rank adapters.

    Forward: base(x) + (alpha/r) * dropout(x) @ A^T @ B^T
    A is initialized with kaiming_uniform, B is zeros → delta starts at 0.
    """

    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 16.0,
                 dropout: float = 0.05):
        super().__init__()
        self.base = base
        self.r = r
        self.scale = alpha / r
        d_in, d_out = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.empty(r, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + self.scale * lora_out

    def extra_repr(self) -> str:
        return (f"in={self.base.in_features}, out={self.base.out_features}, "
                f"r={self.r}, scale={self.scale:.3f}")


def apply_lora(
    model: nn.Module,
    r: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.05,
    targets: Tuple[str, ...] = ("out_proj", "linear1", "linear2"),
) -> Tuple[nn.Module, int]:
    """Replace all target Linear layers with LoRALinear, freeze everything else.

    Returns (model, num_lora_params).
    """
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad_(False)

    replaced = 0
    # Walk the named modules and replace matching Linear layers
    for full_name, module in list(model.named_modules()):
        parts = full_name.split(".")
        leaf = parts[-1]
        if leaf not in targets:
            continue
        if not isinstance(module, nn.Linear):
            continue

        # Navigate to the parent module
        parent: nn.Module = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        lora_layer = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
        setattr(parent, leaf, lora_layer)
        replaced += 1

    num_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] Applied to {replaced} layers — "
          f"{num_lora:,} trainable / {num_total:,} total params "
          f"({100 * num_lora / num_total:.3f}%)")
    return model, num_lora


def save_lora_adapter(
    model: nn.Module,
    save_dir: str,
    config: dict,
) -> None:
    """Save only the LoRA adapter weights + config to save_dir."""
    os.makedirs(save_dir, exist_ok=True)
    adapter: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            adapter[name] = param.detach().cpu()

    adapter_path = os.path.join(save_dir, "lora_adapter.pt")
    torch.save(adapter, adapter_path)

    config_path = os.path.join(save_dir, "lora_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    size_mb = os.path.getsize(adapter_path) / 1_048_576
    print(f"[save] LoRA adapter saved → {adapter_path}  ({size_mb:.1f} MB, "
          f"{len(adapter)} tensors)")


def load_lora_adapter(musicgen_model: "MusicGen", adapter_dir: str) -> "MusicGen":
    """Load LoRA adapter weights into a freshly loaded MusicGen model.

    Usage:
        model = MusicGen.get_pretrained('melody')
        model = load_lora_adapter(model, 'models/TrovaMUZ_V1_LoRA')
        audio = model.generate(['bachata melody'])
    """
    config_path = os.path.join(adapter_dir, "lora_config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    apply_lora(
        musicgen_model.lm,
        r=cfg["lora_rank"],
        alpha=cfg["lora_alpha"],
        dropout=cfg.get("lora_dropout", 0.0),
        targets=tuple(cfg["lora_targets"]),
    )

    adapter_path = os.path.join(adapter_dir, "lora_adapter.pt")
    adapter_weights = torch.load(adapter_path, map_location="cpu")
    missing, unexpected = musicgen_model.lm.load_state_dict(
        adapter_weights, strict=False
    )
    if unexpected:
        print(f"[load_lora] Unexpected keys: {unexpected[:5]}")
    print(f"[load_lora] Loaded {len(adapter_weights)} LoRA tensors from {adapter_path}")
    return musicgen_model


# ══════════════════════════════════════════════════════════════════════════════
#  Demucs vocal separation
# ══════════════════════════════════════════════════════════════════════════════

def separate_vocals(wav_path: Path, cache_dir: Path) -> Path:
    """Run Demucs htdemucs --two-stems=vocals on wav_path.

    Returns the no_vocals.wav stem path, or the original wav_path if Demucs
    is not installed or separation fails.
    """
    stem_dir = cache_dir / "htdemucs" / wav_path.stem
    no_vocals = stem_dir / "no_vocals.wav"

    if no_vocals.exists():
        return no_vocals

    try:
        import demucs  # noqa: F401 — just check availability
    except ImportError:
        print(f"  [demucs] Not installed — skipping separation for {wav_path.name}")
        return wav_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [demucs] Separating vocals from {wav_path.name} …")
    result = subprocess.run(
        [sys.executable, "-m", "demucs", "--two-stems=vocals",
         "-o", str(cache_dir), str(wav_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not no_vocals.exists():
        print(f"  [demucs] WARN separation failed: {result.stderr[:300]}")
        return wav_path  # graceful fallback to original

    print(f"  [demucs] ✓ {wav_path.name} → no_vocals.wav")
    return no_vocals


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset
# ══════════════════════════════════════════════════════════════════════════════

class TrovaMuzDataset(Dataset):
    """Loads (wav_path, caption) pairs from a dataset directory.

    Each WAV must have a matching .txt caption file.
    If use_demucs=True, audio is separated before encoding.
    """

    def __init__(self, data_dir: str, use_demucs: bool = False):
        self.pairs: List[Tuple[Path, str]] = []
        self.use_demucs = use_demucs
        self.demucs_cache = Path(data_dir) / ".demucs_cache"

        missing_captions = []
        root = Path(data_dir)
        # Support both flat directories and nested genre subfolders
        all_wavs = sorted(root.rglob("*.wav"))
        # Skip files inside hidden cache directories and the legacy Master_Wav folder
        all_wavs = [w for w in all_wavs
                    if ".demucs_cache" not in w.parts and "Master_Wav" not in w.parts]
        for wav in all_wavs:
            txt = wav.with_suffix(".txt")
            if not txt.exists():
                missing_captions.append(wav.name)
                continue
            caption = txt.read_text(encoding="utf-8").strip()
            if not caption:
                print(f"  [WARN] Empty caption for {wav.name} — skipping")
                continue
            self.pairs.append((wav, caption))

        if missing_captions:
            print(f"[dataset] WARN: {len(missing_captions)} WAVs have no caption "
                  f"and will be skipped: {missing_captions[:5]}")

        if not self.pairs:
            raise ValueError(f"No valid WAV+caption pairs found in {data_dir}")

        print(f"[dataset] {len(self.pairs)} tracks loaded from {data_dir}")
        if use_demucs:
            print(f"[dataset] Demucs vocal separation is ON (cache: {self.demucs_cache})")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        wav_path, caption = self.pairs[idx]
        if self.use_demucs:
            wav_path = separate_vocals(wav_path, self.demucs_cache)
        return str(wav_path), caption


# ══════════════════════════════════════════════════════════════════════════════
#  Audio preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_audio(
    audio_path: str,
    model: "MusicGen",
    duration: int = 30,
    device: torch.device = torch.device("cpu"),
) -> Optional[torch.Tensor]:
    """Load, resample, and encode audio → codebook tokens [1, K, T].

    Returns None if the file is shorter than `duration` seconds.
    """
    p = Path(audio_path)
    try:
        if p.suffix.lower() == ".wav":
            data, sr = sf.read(str(p), always_2d=True)
            wav = torch.from_numpy(data.T).to(torch.float32)
        else:
            wav, sr = torchaudio.load(str(p))
    except Exception as e:
        print(f"  [audio] ERROR loading {p.name}: {e}")
        return None

    # Resample to model sample rate (32 kHz for MusicGen)
    if sr != model.sample_rate:
        wav = torchaudio.functional.resample(wav, sr, model.sample_rate)

    # Mono
    wav = wav.mean(dim=0, keepdim=True)

    # Skip clips shorter than required duration
    min_samples = model.sample_rate * duration
    if wav.shape[1] < min_samples:
        print(f"  [audio] SKIP {p.name}: too short "
              f"({wav.shape[1] / model.sample_rate:.1f}s < {duration}s)")
        return None

    # Random crop to exactly `duration` seconds (data augmentation)
    import random
    start = random.randrange(0, max(wav.shape[1] - min_samples, 1))
    wav = wav[:, start: start + min_samples]

    wav = wav.to(device).unsqueeze(0)  # [1, 1, T]

    with torch.no_grad():
        codes, scale = model.compression_model.encode(wav)
        assert scale is None, "Unexpected non-None scale from EnCodec"

    return codes  # [1, K, T]


# ══════════════════════════════════════════════════════════════════════════════
#  Training
# ══════════════════════════════════════════════════════════════════════════════

def train(args: argparse.Namespace) -> None:
    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = (device.type == "cuda")
    print(f"\n{'='*60}")
    print(f"  TrovaMUZ_V1 LoRA Trainer")
    print(f"{'='*60}")
    print(f"  Device    : {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"  Model     : {args.model_id}")
    print(f"  Dataset   : {args.dataset}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  LR        : {args.lr}")
    print(f"  Batch     : {args.batch_size} × {args.grad_acc} grad_acc "
          f"= {args.batch_size * args.grad_acc} effective")
    print(f"  LoRA rank : {args.lora_rank}  alpha={args.lora_alpha}")
    print(f"  BF16      : {use_bf16}")
    print(f"  Demucs    : {args.demucs}")
    print(f"  Output    : {args.output}")
    print(f"{'='*60}\n")

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset = TrovaMuzDataset(args.dataset, use_demucs=args.demucs)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0,  # avoid pickling issues with Demucs paths
        drop_last=False,
    )

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[model] Loading MusicGen '{args.model_id}' …")
    model = MusicGen.get_pretrained(args.model_id, device=device)
    model.compression_model = model.compression_model.to(device)
    # Keep LM in float32 initially; bf16 autocast handles precision in forward
    model.lm = model.lm.to(torch.float32).to(device)
    model.lm.train()

    # ── Blackwell (SM 12.0+) xFormers compatibility patch ────────────────────────
    # xFormers 0.0.35 operators (fa3, cutlass) require capability ≤ 9.0.
    # Monkey-patch ops.memory_efficient_attention with PyTorch SDPA which supports
    # all CUDA architectures and is equally memory-efficient (FlashAttention-style).
    try:
        from xformers import ops as _xops
        from torch.nn import functional as _F

        def _sdpa_compat(q, k, v, attn_bias=None, p: float = 0.0, scale=None, **kw):
            """Drop-in for xFormers MEA using PyTorch SDPA. Inputs: [B,T,H,D]."""
            q_t = q.transpose(1, 2)   # → [B, H, T, D]
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)
            is_causal, mask = False, None
            if attn_bias is not None:
                try:
                    from xformers.ops.fmha.attn_bias import LowerTriangularMask
                    if isinstance(attn_bias, LowerTriangularMask):
                        is_causal = True
                    else:
                        Tq, Tk = q.shape[1], k.shape[1]
                        mask = attn_bias.materialize((Tq, Tk), dtype=q.dtype, device=q.device)
                except Exception:
                    pass
            out = _F.scaled_dot_product_attention(
                q_t, k_t, v_t,
                attn_mask=mask, dropout_p=p, is_causal=is_causal,
            )
            return out.transpose(1, 2)  # → [B, T, H, D]

        _xops.memory_efficient_attention = _sdpa_compat
        print("[compat] xFormers MEA → PyTorch SDPA (Blackwell SM 12.0 support)")
    except Exception as _e:
        print(f"[compat] SDPA patch skipped: {_e}")

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    model.lm, _ = apply_lora(
        model.lm,
        r=args.lora_rank,
        alpha=float(args.lora_alpha),
        dropout=args.lora_dropout,
        targets=("out_proj", "linear1", "linear2"),
    )
    # Move LoRA params (created on CPU) to the correct device
    model.lm = model.lm.to(device)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    if args.resume:
        adapter_path = os.path.join(args.resume, "lora_adapter.pt")
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"[resume] Adapter not found: {adapter_path}")
        print(f"[resume] Loading LoRA weights from {adapter_path}")
        adapter_weights = torch.load(adapter_path, map_location=device)
        missing, unexpected = model.lm.load_state_dict(adapter_weights, strict=False)
        if unexpected:
            print(f"[resume] Unexpected keys (first 3): {unexpected[:3]}")
        print(f"[resume] Loaded {len(adapter_weights)} tensors — continuing from epoch {args.start_epoch}")

    # ── Optimizer (only LoRA params) ──────────────────────────────────────────
    lora_params = [p for p in model.lm.parameters() if p.requires_grad]
    assert lora_params, "No trainable LoRA parameters found!"

    try:
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(
            lora_params,
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )
        print("[optimizer] AdamW8bit (bitsandbytes)")
    except ImportError:
        optimizer = AdamW(
            lora_params,
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )
        print("[optimizer] AdamW (bitsandbytes not available)")

    remaining_epochs = args.epochs - args.start_epoch + 1
    total_steps = remaining_epochs * math.ceil(len(dataloader) / args.grad_acc)
    scheduler = get_scheduler(
        "cosine",
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    criterion = nn.CrossEntropyLoss()
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else contextlib.nullcontext()
    )

    os.makedirs(args.output, exist_ok=True)

    # ── LoRA config (for saving) ───────────────────────────────────────────────
    lora_config = {
        "model_id": args.model_id,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_targets": ["out_proj", "linear1", "linear2"],
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_acc": args.grad_acc,
        "dataset": args.dataset,
        "trigger_word": "musicgau_adn style, TrovaMUZ_V1",
    }

    # ── R2 upload helper (crash protection) ──────────────────────────────────
    def _upload_best_to_r2(output_dir: str) -> None:
        try:
            import boto3 as _boto3
            _aid = os.environ.get('R2_ACCOUNT_ID')
            _ak  = os.environ.get('R2_ACCESS_KEY')
            _sk  = os.environ.get('R2_SECRET_KEY')
            _bkt = os.environ.get('R2_BUCKET_NAME') or os.environ.get('R2_BUCKET')
            if not all([_aid, _ak, _sk, _bkt]):
                print("[r2] R2 env vars missing — skipping mid-training upload")
                return
            _s3 = _boto3.client('s3',
                endpoint_url=f'https://{_aid}.r2.cloudflarestorage.com',
                aws_access_key_id=_ak, aws_secret_access_key=_sk, region_name='auto')
            best_dir = Path(output_dir) / 'best'
            for _f in best_dir.glob('*'):
                _key = f'TrovaMUZ_V1/adapters/{output_dir}/best/{_f.name}'
                _s3.upload_file(str(_f), _bkt, _key)
            print(f"[r2] ✓ best adapter uploaded to R2 ({output_dir}/best/)")
        except Exception as _e:
            print(f"[r2] Upload skipped (non-fatal): {_e}")

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step = 0
    optimizer_step = 0
    best_loss = float("inf")
    running_loss = 0.0
    loss_count = 0

    for epoch in range(args.start_epoch, args.epochs + 1):
        epoch_loss = 0.0
        epoch_batches = 0

        for batch_idx, (audio_paths, labels) in enumerate(dataloader):
            # Encode audio → codebook tokens
            all_codes: List[torch.Tensor] = []
            texts: List[str] = []

            for path, caption in zip(audio_paths, labels):
                codes = preprocess_audio(
                    path, model, duration=args.duration, device=device
                )
                if codes is None:
                    continue
                all_codes.append(codes)
                texts.append(caption)

            if not all_codes:
                continue

            codes = torch.cat(all_codes, dim=0)  # [B, K, T]

            # Caption dropout for classifier-free guidance training
            import random as _random
            texts_cfgd = [
                "" if _random.random() < args.caption_dropout else t
                for t in texts
            ]

            # Prepare conditioning
            attributes, _ = model._prepare_tokens_and_attributes(texts_cfgd, None)
            tokenized = model.lm.condition_provider.tokenize(attributes)
            condition_tensors = model.lm.condition_provider(tokenized)

            # Forward pass
            with autocast_ctx:
                lm_output = model.lm.compute_predictions(
                    codes=codes,
                    conditions=[],
                    condition_tensors=condition_tensors,
                )
                logits = lm_output.logits  # [B, K, T, card]
                mask = lm_output.mask       # [B, K, T]

                masked_logits = logits[mask]   # [N, card]
                masked_codes = codes[mask]      # [N]

                # Normalise by grad_acc so accumulated loss equals mean-of-batch
                loss = criterion(masked_logits, masked_codes) / args.grad_acc

            loss.backward()
            global_step += 1

            loss_val = loss.item() * args.grad_acc  # un-normalise for logging
            running_loss += loss_val
            loss_count += 1
            epoch_loss += loss_val
            epoch_batches += 1

            print(
                f"  ep {epoch}/{args.epochs} | "
                f"batch {batch_idx + 1}/{len(dataloader)} | "
                f"loss {loss_val:.4f}",
                flush=True,
            )

            # Optimizer step every grad_acc micro-batches
            if global_step % args.grad_acc == 0:
                torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_step += 1

                avg_loss = running_loss / loss_count
                running_loss = 0.0
                loss_count = 0
                print(
                    f"  >> optimizer step {optimizer_step} | "
                    f"avg loss {avg_loss:.4f} | "
                    f"lr {scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )

                # Checkpoint every N optimizer steps
                if args.save_step and optimizer_step % args.save_step == 0:
                    ckpt_dir = os.path.join(args.output, f"checkpoint-{optimizer_step}")
                    save_lora_adapter(model.lm, ckpt_dir, lora_config)

                # Track best
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_lora_adapter(
                        model.lm,
                        os.path.join(args.output, "best"),
                        lora_config,
                    )

        # Flush remaining gradient at epoch end (if batch count isn't a multiple of grad_acc)
        if global_step % args.grad_acc != 0:
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_epoch = epoch_loss / max(epoch_batches, 1)
        print(f"\n[epoch {epoch}] avg loss: {avg_epoch:.4f}  (best: {best_loss:.4f})\n")

        # Save after each epoch
        epoch_dir = os.path.join(args.output, f"epoch-{epoch}")
        save_lora_adapter(model.lm, epoch_dir, lora_config)

        # Upload best adapter to R2 every N epochs (crash protection)
        if args.upload_every and epoch % args.upload_every == 0:
            _upload_best_to_r2(args.output)

        # Generate audio sample every N epochs
        if args.sample_every and epoch % args.sample_every == 0 and args.test_prompt:
            samples_dir = os.path.join(args.output, "samples")
            os.makedirs(samples_dir, exist_ok=True)
            sample_path = os.path.join(samples_dir, f"epoch_{epoch:04d}.wav")
            print(f"[sample] Generating epoch {epoch} sample...", flush=True)
            model.lm.eval()
            model.set_generation_params(duration=10, use_sampling=True, top_k=250)
            with torch.no_grad():
                audio = model.generate([args.test_prompt])
            wav_cpu = audio.squeeze(0).cpu()
            try:
                torchaudio.save(sample_path, wav_cpu, model.sample_rate)
            except (ImportError, RuntimeError):
                import soundfile as _sf
                import numpy as _np
                _sf.write(sample_path, wav_cpu.numpy().T, model.sample_rate)
            model.lm.train()
            print(f"[sample] Saved → {sample_path}", flush=True)

    # ── Final save ────────────────────────────────────────────────────────────
    final_dir = os.path.join(args.output, "final")
    save_lora_adapter(model.lm, final_dir, lora_config)

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best loss    : {best_loss:.4f}")
    print(f"  Adapter saved: {final_dir}/lora_adapter.pt")
    print(f"{'='*60}\n")

    # ── Quick generation test ─────────────────────────────────────────────────
    if args.test_prompt:
        print(f"[test] Generating sample with prompt: '{args.test_prompt}'")
        model.lm.eval()
        model.set_generation_params(duration=10, use_sampling=True, top_k=250)
        with torch.no_grad():
            audio = model.generate([args.test_prompt])
        out_wav = os.path.join(args.output, "test_generation.wav")
        wav_cpu = audio.squeeze(0).cpu()
        try:
            torchaudio.save(out_wav, wav_cpu, model.sample_rate)
        except (ImportError, RuntimeError):
            import soundfile as sf
            import numpy as np
            sf.write(out_wav, wav_cpu.numpy().T, model.sample_rate)
        print(f"[test] Sample saved → {out_wav}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TrovaMUZ_V1 — LoRA fine-tuner for MusicGen Melody",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset / model
    p.add_argument("--dataset", required=True,
                   help="Path to dataset folder (must contain .wav + .txt pairs)")
    p.add_argument("--model_id", default="melody",
                   choices=["small", "medium", "large", "melody"],
                   help="Base MusicGen model to fine-tune")
    p.add_argument("--duration", type=int, default=30,
                   help="Audio clip duration in seconds for training")
    # LoRA
    p.add_argument("--lora_rank", type=int, default=16,
                   help="LoRA rank (r). Higher = more capacity, more VRAM")
    p.add_argument("--lora_alpha", type=float, default=16.0,
                   help="LoRA alpha. Scale = alpha/rank. Default keeps scale=1.0")
    p.add_argument("--lora_dropout", type=float, default=0.05,
                   help="Dropout applied to LoRA input path")
    # Training hyperparams
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_acc", type=int, default=4,
                   help="Gradient accumulation steps (effective batch = batch_size × grad_acc)")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=20)
    p.add_argument("--caption_dropout", type=float, default=0.05,
                   help="Probability to drop captions for CFG training")
    p.add_argument("--resume", default=None,
                   help="Adapter dir to resume from (contains lora_adapter.pt)")
    p.add_argument("--start_epoch", type=int, default=1,
                   help="Starting epoch number (use with --resume)")
    p.add_argument("--save_step", type=int, default=None,
                   help="Save a checkpoint every N optimizer steps (default: off)")
    p.add_argument("--sample_every", type=int, default=None,
                   help="Generate a test audio sample every N epochs and save to output/samples/")
    p.add_argument("--upload_every", type=int, default=None,
                   help="Upload best adapter to R2 every N epochs (crash protection)")
    # Demucs
    p.add_argument("--demucs", action="store_true",
                   help="Run Demucs vocal separation before encoding "
                        "(requires: pip install demucs)")
    # Output
    p.add_argument("--output", default="models/TrovaMUZ_V1_LoRA",
                   help="Directory to save LoRA adapter weights")
    p.add_argument("--test_prompt", default="musicgau_adn style, TrovaMUZ_V1, bolero bachata",
                   help="Generate a test clip with this prompt after training "
                        "(set to empty string to skip)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
