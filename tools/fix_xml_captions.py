#!/usr/bin/env python3
"""
fix_xml_captions.py — Converts ACE-Step XML captions to plain MusicGen format
==============================================================================
Fixes TXT files that look like:
  <CAPTION>TROVAMUZ style, bachata track, by Artist</CAPTION><BPM>125</BPM><KEYSCALE>A minor</KEYSCALE>...

Produces clean MusicGen captions:
  musicgau_adn style, TrovaMUZ_V1 bachata track, by Artist, 118 BPM, D minor, professional studio sound

Usage:
  python tools/fix_xml_captions.py --folder training/datasets/bachata
  python tools/fix_xml_captions.py --all            # all folders with XML captions
  python tools/fix_xml_captions.py --all --dry-run  # preview only
"""

import argparse
import re
import sys
from pathlib import Path

TRIGGER = "musicgau_adn style, TrovaMUZ_V1"

GENRE_DESCRIPTORS = {
    "bachata": "syncopated bicheo rhythm, requinto guitar lead with bends, rhythmic segunda guitar, slap bass on beat 4, bongo martillo, bright guira, intimate studio sound",
    "bachata-de-amargue": "bitter melancholic bachata de amargue, sharp requinto bends, punching bongo martillo, bright guira, dark emotional raw studio sound",
    "merengue": "driving merengue rhythm, brass section, accordion, tambora drum, güira, energetic dance groove, commercial studio mix",
    "merengue clasico": "classic merengue, brass section, accordion, tambora, güira, festive tropical energy, vintage studio recording",
}


def try_librosa(wav_path: Path):
    """Return (bpm, key_str) or (None, None) if librosa unavailable / fails."""
    try:
        import librosa
        import numpy as np
        y, sr = librosa.load(str(wav_path), sr=None, mono=True, duration=60)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_val = tempo.item() if hasattr(tempo, "item") else float(tempo)
        bpm = int(round(tempo_val))

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        root_idx = int(np.argmax(chroma_mean))
        root = pitch_classes[root_idx]

        # Simple major/minor heuristic: compare 3rd (index+4) vs minor 3rd (index+3)
        major_3rd = chroma_mean[(root_idx + 4) % 12]
        minor_3rd = chroma_mean[(root_idx + 3) % 12]
        mode = "major" if major_3rd >= minor_3rd else "minor"
        return bpm, f"{root} {mode}"
    except Exception:
        return None, None


def parse_xml_caption(text: str):
    """Extract fields from ACE-Step XML caption string."""
    caption = re.search(r"<CAPTION>(.*?)</CAPTION>", text)
    bpm     = re.search(r"<BPM>(.*?)</BPM>", text)
    key     = re.search(r"<KEYSCALE>(.*?)</KEYSCALE>", text)
    return (
        caption.group(1).strip() if caption else "",
        bpm.group(1).strip()     if bpm     else "",
        key.group(1).strip()     if key     else "",
    )


def is_xml_caption(text: str) -> bool:
    return "<CAPTION>" in text


def fix_trigger(caption_text: str) -> str:
    """Replace wrong trigger with correct one."""
    fixed = re.sub(r"TROVAMUZ\s+style", TRIGGER, caption_text, flags=re.IGNORECASE)
    if not fixed.startswith("musicgau_adn"):
        fixed = f"{TRIGGER} {fixed}"
    # Remove comma immediately after TrovaMUZ_V1 so it reads "TrovaMUZ_V1 genre..." not "TrovaMUZ_V1, genre..."
    fixed = re.sub(r"(TrovaMUZ_V1),\s*", r"\1 ", fixed)
    return fixed


def build_clean_caption(caption_text: str, xml_bpm: str, xml_key: str,
                        wav_path: Path, genre_folder: str, use_librosa: bool) -> str:
    """Build the final plain-text MusicGen caption."""
    # Fix trigger word
    core = fix_trigger(caption_text)

    # Get BPM and key
    if use_librosa:
        bpm, key = try_librosa(wav_path)
        bpm = bpm or xml_bpm
        key = key or xml_key
    else:
        bpm, key = xml_bpm, xml_key

    # Add genre descriptors if available
    descriptor = GENRE_DESCRIPTORS.get(genre_folder, "")

    parts = [core]
    if descriptor:
        parts.append(descriptor)
    if bpm:
        parts.append(f"{bpm} BPM")
    if key:
        parts.append(key)
    if "professional studio sound" not in core and "studio sound" not in core:
        parts.append("professional studio sound")

    return ", ".join(parts)


def process_folder(folder: Path, dry_run: bool, use_librosa: bool) -> tuple[int, int]:
    genre = folder.name
    txt_files = sorted(folder.glob("*.txt"))
    xml_files = [f for f in txt_files if is_xml_caption(f.read_text(encoding="utf-8", errors="replace"))]

    if not xml_files:
        print(f"  [{genre}] — no XML captions found, skipping")
        return 0, 0

    fixed = 0
    for txt in xml_files:
        raw = txt.read_text(encoding="utf-8", errors="replace")
        caption_text, xml_bpm, xml_key = parse_xml_caption(raw)

        if not caption_text:
            print(f"    SKIP {txt.name} — cannot parse CAPTION tag")
            continue

        wav = txt.with_suffix(".wav")
        new_caption = build_clean_caption(
            caption_text, xml_bpm, xml_key, wav, genre, use_librosa and wav.exists()
        )

        if dry_run:
            print(f"    DRY  {txt.name}")
            print(f"         OLD: {raw[:80]}...")
            print(f"         NEW: {new_caption}")
        else:
            txt.write_text(new_caption, encoding="utf-8")
            fixed += 1

    return len(xml_files), fixed


def main():
    p = argparse.ArgumentParser(description="Fix ACE-Step XML captions → MusicGen plain text")
    p.add_argument("--folder", default=None, help="Single genre folder to fix")
    p.add_argument("--all", action="store_true", help="Fix all folders under training/datasets/")
    p.add_argument("--datasets-dir", default="training/datasets",
                   help="Root datasets directory (default: training/datasets)")
    p.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    p.add_argument("--no-librosa", action="store_true",
                   help="Skip librosa BPM/key analysis (use XML values instead)")
    args = p.parse_args()

    if not args.folder and not args.all:
        p.print_help()
        sys.exit(1)

    use_librosa = not args.no_librosa
    if use_librosa:
        try:
            import librosa  # noqa: F401
            print("librosa detected — will measure real BPM and key from audio")
        except ImportError:
            print("librosa not installed — using BPM/key from XML tags")
            use_librosa = False

    datasets_dir = Path(args.datasets_dir)
    total_found = total_fixed = 0

    if args.folder:
        folders = [Path(args.folder)]
    else:
        folders = sorted(datasets_dir.iterdir())
        folders = [f for f in folders if f.is_dir()]

    for folder in folders:
        print(f"\n>> {folder.name}")
        found, fixed = process_folder(folder, args.dry_run, use_librosa)
        total_found += found
        total_fixed += fixed
        if not args.dry_run and fixed:
            print(f"   Fixed {fixed}/{found} caption(s)")

    print(f"\n{'DRY RUN — no files changed' if args.dry_run else f'Done: {total_fixed}/{total_found} captions fixed'}")


if __name__ == "__main__":
    main()
