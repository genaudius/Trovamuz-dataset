import sys
from pathlib import Path
import numpy as np
import soundfile as sf
sys.path.insert(0, str(Path('repositories/audiocraft')))
from audiocraft.models import MusicGen

print('Loading debug model...')
model = MusicGen.get_pretrained('debug', device='cpu')
model.set_generation_params(duration=4)
print('Generating...')
wav = model.generate(['alegre rock'], progress=False)
print('Saving...')
wav_np = wav[0].cpu().numpy()
if wav_np.ndim == 2:
    wav_np = wav_np.T
wav_np = np.clip(wav_np, -1.0, 1.0).astype(np.float32)
sf.write('test_debug.wav', wav_np, model.sample_rate)
print('saved test_debug.wav')
