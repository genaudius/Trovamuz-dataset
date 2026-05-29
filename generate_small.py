import sys
from pathlib import Path
import numpy as np
import soundfile as sf
sys.path.insert(0, str(Path('repositories/audiocraft')))
from audiocraft.models import MusicGen

print('Loading small model...')
model = MusicGen.get_pretrained('small', device='cpu')
model.set_generation_params(duration=6, top_k=120, top_p=0.95, temperature=1.0, cfg_coef=3.0)
print('Generating prompt...')
wav = model.generate(['una canción alegre con ritmo de rock y sintetizadores'], progress=False)
print('Saving...')
wav_np = wav[0].cpu().numpy()
if wav_np.ndim == 2:
    wav_np = wav_np.T
wav_np = np.clip(wav_np, -1.0, 1.0).astype(np.float32)
sf.write('test_small.wav', wav_np, model.sample_rate)
print('saved test_small.wav')
