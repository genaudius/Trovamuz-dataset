#!/bin/bash
# ============================================================
#  TrovaMUZ — RunPod H100 Setup Script
#  Run this once when you launch a new pod:
#    bash setup_runpod.sh
# ============================================================

set -e

REPO_URL="${TROVAMUZ_REPO:-https://github.com/genaudius/Trovamuz-dataset.git}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   TrovaMUZ RunPod H100 Setup             ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── System packages ───────────────────────────────────────────────────────────
echo "[1/6] Installing system packages..."
apt-get update -qq && apt-get install -y -qq \
    git ffmpeg libsndfile1 python3-pip python3-venv wget curl screen \
    pkg-config libavformat-dev libavcodec-dev libavdevice-dev \
    libavutil-dev libswscale-dev libswresample-dev > /dev/null
echo "      ✓ System packages ready"

# ── Clone or update the repo ──────────────────────────────────────────────────
echo "[2/6] Setting up TrovaMUZ repo..."
if [ -z "$REPO_URL" ]; then
    echo "ERROR: Set TROVAMUZ_REPO env var to your GitHub repo URL"
    echo "       Example: export TROVAMUZ_REPO=https://github.com/youruser/TrovaMUZ"
    exit 1
fi

if [ -n "$GITHUB_TOKEN" ]; then
    AUTH_URL=$(echo "$REPO_URL" | sed "s|https://|https://${GITHUB_TOKEN}@|")
else
    AUTH_URL="$REPO_URL"
fi

if [ -d "TrovaMUZ/.git" ]; then
    echo "      Repo exists — pulling latest..."
    cd TrovaMUZ && git pull && cd ..
else
    git clone "$AUTH_URL" TrovaMUZ
fi
cd TrovaMUZ
echo "      ✓ Repo ready"

# ── Python virtual environment ────────────────────────────────────────────────
echo "[3/6] Creating Python venv..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
echo "      ✓ venv ready"

# ── PyTorch with CUDA 12.1 (H100 compatible) ─────────────────────────────────
echo "[4/6] Installing PyTorch + CUDA 12.1..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
python -c "import torch; print(f'      ✓ PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}')"

# ── Project requirements ──────────────────────────────────────────────────────
echo "[5/6] Installing project requirements..."
pip install -r requirements.txt -q
pip install -r requirements_tools.txt -q
pip install transformers wandb demucs boto3 bitsandbytes -q

# ── Audiocraft v1.3.0 (must use tag — main branch has JASCO/spacy conflict) ──
if [ ! -d "repositories/audiocraft/.git" ]; then
    echo "      Cloning audiocraft v1.3.0..."
    mkdir -p repositories
    git clone https://github.com/facebookresearch/audiocraft.git repositories/audiocraft -q
    cd repositories/audiocraft && git checkout v1.3.0 -q && cd ../..
fi
pip install -e repositories/audiocraft -q

# ── Reinstall torch after audiocraft (audiocraft may downgrade it) ────────────
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 --upgrade -q
pip install "numpy<2.0" --force-reinstall -q

echo "      ✓ All packages installed"

# ── Dataset desde Cloudflare R2 ───────────────────────────────────────────────
echo "[6/6] Descargando dataset desde Cloudflare R2..."
pip install boto3 -q
python -c "
import boto3, os
from pathlib import Path

account_id = os.environ['R2_ACCOUNT_ID']
access_key = os.environ['R2_ACCESS_KEY']
secret_key = os.environ['R2_SECRET_KEY']
bucket     = os.environ.get('R2_BUCKET_NAME') or os.environ['R2_BUCKET']
prefix     = os.environ.get('R2_PREFIX', 'TrovaMUZ_V1/dataset')
endpoint   = f'https://{account_id}.r2.cloudflarestorage.com'

s3 = boto3.client('s3', endpoint_url=endpoint,
    aws_access_key_id=access_key, aws_secret_access_key=secret_key, region_name='auto')

paginator = s3.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
objects = [obj for page in pages for obj in page.get('Contents', [])]
print(f'  {len(objects)} archivos en R2...')

for obj in objects:
    key = obj['Key']
    rel = key[len(prefix):].lstrip('/')
    local = Path('training/datasets') / rel
    local.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, str(local))

wav_count = len(list(Path('training/datasets').rglob('*.wav')))
print(f'  {wav_count} WAV files descargados OK')
"
echo "      ✓ Dataset listo"

# ── GPU check ─────────────────────────────────────────────────────────────────
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null \
    | awk '{print "      " $0}' || echo "      (nvidia-smi not available)"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   Setup complete! Next steps:            ║"
echo "║                                          ║"
echo "║   1. Upload your WAV datasets:               ║"
echo "║      scp -r training/datasets/ \\            ║"
echo "║        root@POD_IP:/workspace/TrovaMUZ/training/ ║"
echo "║                                              ║"
echo "║   2. Create your .env file:                  ║"
echo "║      cp .env.example .env && nano .env        ║"
echo "║                                              ║"
echo "║   3. Enrich captions (needs API keys):       ║"
echo "║      python tools/enrich_captions.py --all   ║"
echo "║                                              ║"
echo "║   4. Start LoRA training:                    ║"
echo "║      bash train_trovamuz.sh                  ║"
echo "║      # With vocal separation:                ║"
echo "║      bash train_trovamuz.sh --demucs         ║"
echo "╚══════════════════════════════════════════╝"
echo ""
