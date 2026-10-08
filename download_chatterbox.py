import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using:", device)
print("Loading Chatterbox...")

model = ChatterboxMultilingualTTS.from_pretrained(
    device=device
)

print("Chatterbox is ready!")