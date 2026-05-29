#!/usr/bin/env python3
"""
Sube el dataset TrovaMUZ_V1 a HuggingFace Hub (repositorio privado).
Uso: python tools/upload_dataset_hf.py --repo GenAudius/TrovaMUZ_V1_dataset
"""
import argparse
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="HuggingFace repo ID, ej: GenAudius/TrovaMUZ_V1_dataset")
    p.add_argument("--dataset_dir", default="training/datasets", help="Carpeta local del dataset")
    p.add_argument("--token", default=None, help="HF token (o usa HF_TOKEN env var)")
    args = p.parse_args()

    from huggingface_hub import HfApi, create_repo

    token = args.token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    # Crear repo privado si no existe
    try:
        create_repo(args.repo, repo_type="dataset", private=True, token=token, exist_ok=True)
        print(f"Repo: https://huggingface.co/datasets/{args.repo}")
    except Exception as e:
        print(f"Repo ya existe o error: {e}")

    # Subir todos los WAV y TXT
    dataset_path = Path(args.dataset_dir)
    files = sorted(list(dataset_path.rglob("*.wav")) + list(dataset_path.rglob("*.txt")))

    print(f"Subiendo {len(files)} archivos...")
    for i, f in enumerate(files, 1):
        rel = f.relative_to(dataset_path)
        print(f"  [{i}/{len(files)}] {rel}", end="\r", flush=True)
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=str(rel),
            repo_id=args.repo,
            repo_type="dataset",
            token=token,
        )

    print(f"\nDataset subido a: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
