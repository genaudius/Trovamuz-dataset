import sys
from pathlib import Path
import argparse
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path('repositories/audiocraft')))
from audiocraft.models import MusicGen
import torch

parser = argparse.ArgumentParser(description='Generate audio from a locally trained MusicGen checkpoint.')
parser.add_argument('--prompt', type=str, required=True)
parser.add_argument('--weights_path', type=str, required=True)
parser.add_argument('--model_id', type=str, default='debug')
parser.add_argument('--save_path', type=str, default='trained_sample.wav')
parser.add_argument('--duration', type=float, default=10.0)
parser.add_argument('--sample_loops', type=int, default=2)
parser.add_argument('--top_k', type=int, default=120)
parser.add_argument('--top_p', type=float, default=0.95)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--cfg_coef', type=float, default=3.0)
args = parser.parse_args()

print('Loading model', args.model_id)
model = MusicGen.get_pretrained(args.model_id, device='cpu')
print('Loading LM checkpoint from', args.weights_path)
model.lm.load_state_dict(torch.load(args.weights_path, map_location='cpu'))

attributes, prompt_tokens = model._prepare_tokens_and_attributes([args.prompt], None)
print('Prompt tokens shape:', prompt_tokens.shape if prompt_tokens is not None else None)

model.generation_params = {
    'max_gen_len': int(args.duration * model.frame_rate),
    'use_sampling': True,
    'temp': args.temperature,
    'top_k': args.top_k,
    'top_p': args.top_p,
    'cfg_coef': args.cfg_coef,
    'two_step_cfg': False,
}

total = []
for loop in range(args.sample_loops):
    print(f'Generating loop {loop+1}/{args.sample_loops}')
    with model.autocast:
        gen_tokens = model.lm.generate(prompt_tokens, attributes, callback=None, **model.generation_params)
        total.append(gen_tokens[..., prompt_tokens.shape[-1] if prompt_tokens is not None else 0:])
        prompt_tokens = gen_tokens[..., -gen_tokens.shape[-1] // 2:]

gen_tokens = torch.cat(total, -1)
print('Generated tokens shape:', gen_tokens.shape)
with torch.no_grad():
    gen_audio = model.compression_model.decode(gen_tokens, None)

wav_np = gen_audio[0].cpu().numpy()
if wav_np.ndim == 2:
    wav_np = wav_np.T
wav_np = np.clip(wav_np, -1.0, 1.0).astype(np.float32)
sf.write(args.save_path, wav_np, model.sample_rate)
print('Saved generated audio to', args.save_path)
