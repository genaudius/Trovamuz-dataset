import argparse
from pathlib import Path
import subprocess
import sys
from shutil import copy2

try:
    import soundfile as sf
    import torch
    import torchaudio
    from torchaudio import functional as F
except ImportError as exc:
    print("Missing required packages. Install soundfile, torch, and torchaudio in the venv.")
    raise SystemExit(1) from exc

AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}


def format_duration(seconds: float) -> str:
    return f"{int(seconds // 60)}m{int(seconds % 60)}s"


def scan_dataset(dataset_dir: Path):
    audio_files = [p for p in sorted(dataset_dir.iterdir()) if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    text_files = [p for p in sorted(dataset_dir.iterdir()) if p.is_file() and p.suffix.lower() == '.txt']
    return audio_files, text_files


def validate_dataset(dataset_dir: Path, min_duration: float = 30.0):
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    audio_files, text_files = scan_dataset(dataset_dir)

    if not audio_files:
        print('No audio files found in dataset. Add WAV files and matching TXT captions.')
        return 1

    text_map = {p.stem: p for p in text_files}
    audio_map = {p.stem: p for p in audio_files}

    errors = 0
    for audio_path in audio_files:
        key = audio_path.stem
        if key not in text_map:
            print(f"[MISSING CAPTION] {audio_path.name} has no {audio_path.stem}.txt")
            errors += 1

    for text_path in text_files:
        key = text_path.stem
        if key not in audio_map:
            print(f"[MISSING AUDIO] {text_path.name} has no {text_path.stem}.wav/mp3")
            errors += 1

    if errors:
        print(f"Found {errors} missing pair(s). Fix the filenames first.")

    for audio_path in audio_files:
        try:
            try:
                info = sf.info(str(audio_path))
                duration = info.frames / info.samplerate
            except Exception:
                waveform, sr = torchaudio.load(str(audio_path))
                duration = waveform.shape[-1] / sr
        except Exception as exc:
            print(f"[ERROR] Could not read audio info for {audio_path.name}: {exc}")
            errors += 1
            continue
        if duration < min_duration:
            print(f"[TOO SHORT] {audio_path.name} is {format_duration(duration)}, needs at least {format_duration(min_duration)}")
            errors += 1

    if errors == 0:
        print(f"Dataset is valid: {len(audio_files)} audio files, {len(text_files)} captions.")
    else:
        print(f"Dataset validation completed with {errors} issue(s). See messages above.")
    return errors


def convert_audio_file(source_path: Path, target_path: Path, sample_rate: int = 32000):
    if source_path.suffix.lower() == '.wav':
        data, sr = sf.read(str(source_path), always_2d=True)
        if sr != sample_rate:
            waveform = torch.from_numpy(data.T)
            waveform = F.resample(waveform, orig_freq=sr, new_freq=sample_rate)
            audio_np = waveform.cpu().numpy().T
        else:
            audio_np = data
    else:
        waveform, sr = torchaudio.load(str(source_path))
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if sr != sample_rate:
            waveform = F.resample(waveform, orig_freq=sr, new_freq=sample_rate)
        audio_np = waveform.cpu().numpy().T

    if audio_np.ndim == 1:
        sf.write(str(target_path), audio_np, sample_rate)
    else:
        sf.write(str(target_path), audio_np, sample_rate)
    print(f"Converted {source_path.name} -> {target_path.name} @ {sample_rate} Hz")


def create_dataset(dataset_name: str):
    root = Path(__file__).resolve().parent / 'datasets'
    target = root / dataset_name
    target.mkdir(parents=True, exist_ok=True)
    readme = target / 'README.md'
    if not readme.exists():
        readme.write_text(
            "# Dataset: {}\n\n".format(dataset_name)
            + "Place WAV audio files and TXT caption files here.\n"
            + "Each audio file must have a corresponding text file with the same base name.\n"
            + "Example: `segment_000.wav` and `segment_000.txt`.\n"
            + "Audio files must be at least 30 seconds long for training.\n"
        )
    print(f"Created dataset folder: {target}")
    print("Put WAV audio and TXT caption files in that folder.")
    return target


def main():
    parser = argparse.ArgumentParser(description="Prepare and validate MusicGen training datasets.")
    sub = parser.add_subparsers(dest='command', required=True)

    init = sub.add_parser('init', help='Create a new dataset folder in training/datasets/')
    init.add_argument('name', help='Dataset name')

    validate = sub.add_parser('validate', help='Validate a dataset folder in training/datasets/')
    validate.add_argument('dataset_path', nargs='?', default=None, help='Relative path to dataset folder')
    validate.add_argument('--min-duration', type=float, default=30.0, help='Minimum audio duration in seconds')

    convert = sub.add_parser('convert', help='Convert a source audio folder into WAV files for training')
    convert.add_argument('source', help='Source folder containing audio files')
    convert.add_argument('target', help='Target dataset folder under training/datasets/')
    convert.add_argument('--sample-rate', type=int, default=32000, help='Output sample rate for WAV files')

    args = parser.parse_args()
    root = Path(__file__).resolve().parent

    if args.command == 'init':
        create_dataset(args.name)
    elif args.command == 'validate':
        if args.dataset_path:
            path = Path(args.dataset_path)
            if not path.is_absolute():
                candidate = (root / args.dataset_path)
                path = candidate if candidate.exists() else path.resolve()
        else:
            path = root
        return_code = validate_dataset(path, args.min_duration)
        sys.exit(return_code)
    elif args.command == 'convert':
        source_dir = Path(args.source).resolve()
        target_dir = (root / 'datasets' / args.target).resolve()
        if not source_dir.exists():
            raise FileNotFoundError(f"Source folder not found: {source_dir}")
        target_dir.mkdir(parents=True, exist_ok=True)

        source_files = sorted(source_dir.iterdir())
        audio_files = [p for p in source_files if p.suffix.lower() in AUDIO_EXTS]
        text_files = {p.stem: p for p in source_files if p.is_file() and p.suffix.lower() == '.txt'}

        if not audio_files:
            print(f"No supported audio files found in {source_dir}")
            return

        for source_file in audio_files:
            out_file = target_dir / (source_file.stem + '.wav')
            convert_audio_file(source_file, out_file, args.sample_rate)
            caption = text_files.get(source_file.stem)
            if caption:
                copy2(caption, target_dir / caption.name)
                print(f"Copied caption {caption.name}")
            else:
                print(f"[WARNING] No caption text found for {source_file.name}")

        missing_captions = [t.name for stem, t in text_files.items() if stem not in {a.stem for a in audio_files}]
        for caption_name in missing_captions:
            print(f"[UNUSED CAPTION] {caption_name} has no matching audio file")

        print(f"Converted files to {target_dir}")


if __name__ == '__main__':
    main()
