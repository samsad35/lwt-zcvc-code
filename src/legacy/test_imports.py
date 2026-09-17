import sys
import os
import faulthandler
import signal
faulthandler.enable()
faulthandler.register(signal.SIGUSR1)

print("1. Importing torch...", flush=True)
import torch
torch.cuda.is_available = lambda: False
print("Mocked torch.cuda.is_available to return False.", flush=True)
print("2. Importing torchaudio...", flush=True)
import torchaudio
print("3. Importing numpy...", flush=True)
import numpy as np
print("4. Importing transformers...", flush=True)
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
print("5. Importing speechbrain...", flush=True)
from speechbrain.inference.speaker import EncoderClassifier
print("All imports completed successfully!", flush=True)
