#!/usr/bin/env python3
"""
Sube el dataset TrovaMUZ_V1 a Cloudflare R2 (S3-compatible).
Uso: python tools/upload_dataset_r2.py

Requiere en .env:
  R2_ACCOUNT_ID=tu_account_id
  R2_ACCESS_KEY=tu_access_key
  R2_SECRET_KEY=tu_secret_key
  R2_BUCKET=nombre_del_bucket
"""
import os
import sys
from pathlib import Path


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", default="training/datasets")
    p.add_argument("--prefix", default="TrovaMUZ_V1/dataset", help="Prefijo dentro del bucket")
    p.add_argument("--ext", default=".wav,.txt", help="Extensiones a subir")
    args = p.parse_args()

    # Cargar .env
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    account_id = os.environ["R2_ACCOUNT_ID"]
    access_key = os.environ["R2_ACCESS_KEY"]
    secret_key = os.environ["R2_SECRET_KEY"]
    bucket     = os.environ.get("R2_BUCKET_NAME") or os.environ["R2_BUCKET"]
    endpoint   = f"https://{account_id}.r2.cloudflarestorage.com"

    try:
        import boto3
    except ImportError:
        print("Instalando boto3...")
        os.system(f"{sys.executable} -m pip install boto3 -q")
        import boto3

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )

    extensions = set(args.ext.split(","))
    dataset_path = Path(args.dataset_dir)
    files = [f for f in sorted(dataset_path.rglob("*"))
             if f.is_file() and f.suffix in extensions]

    print(f"Subiendo {len(files)} archivos a R2 bucket '{bucket}'...")
    print(f"Endpoint: {endpoint}")

    ok = 0
    for i, f in enumerate(files, 1):
        rel = f.relative_to(dataset_path)
        key = f"{args.prefix}/{rel}".replace("\\", "/")
        print(f"  [{i:3d}/{len(files)}] {rel}", end="\r", flush=True)
        s3.upload_file(str(f), bucket, key)
        ok += 1

    print(f"\n✓ {ok}/{len(files)} archivos subidos a R2")
    print(f"  Prefijo: s3://{bucket}/{args.prefix}/")


if __name__ == "__main__":
    main()
