import torch
import torchaudio as ta

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using:", device)
print("Loading Chatterbox...")

model = ChatterboxMultilingualTTS.from_pretrained(
    device=device
)

print("Model loaded!")

text = "Hello! This is my first locally generated voice clone."

print("Generating speech...")

wav = model.generate(
    text=text,
    language_id="en",
    audio_prompt_path="audio/my_voice.wav"
)

ta.save(
    "audio/cloned_voice.wav",
    wav,
    model.sr
)

print("Done!")
print("Saved to audio/cloned_voice.wav")