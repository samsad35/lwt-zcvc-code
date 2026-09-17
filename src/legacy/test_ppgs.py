import ppgs
from pathlib import Path
from une import WavLM
import torch

audio_file = "/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac"

audio = ppgs.load.audio(audio_file)
print(f"Audio shape: {audio.shape}")

gpu = 0 if torch.cuda.is_available() else None
p = ppgs.from_audio(audio, ppgs.SAMPLE_RATE, gpu=gpu)
print(f"PPGs shape: {p.shape}")

wavlm = WavLM(model_name="microsoft/wavlm-base", device="cuda" if gpu==0 else "cpu")
with torch.no_grad():
    hw = wavlm(audio_file)[12]
print(f"WavLM shape: {hw.shape}")
