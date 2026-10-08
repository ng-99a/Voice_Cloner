import torch
import torchaudio as ta

from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

#INPUT_AUDIO = "audio/test_speech.wav"
REFERENCE_AUDIO = "audio/my_voice.wav"
OUTPUT_AUDIO = "audio/generated.wav"

WHISPER_MODEL = "openai/whisper-large-v3-turbo"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

dtype = (
    torch.float16
    if torch.cuda.is_available()
    else torch.float32
)

print("Loading Whisper...")

whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    WHISPER_MODEL,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)

whisper_model.to(device)

processor = AutoProcessor.from_pretrained(
    WHISPER_MODEL
)

whisper = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=dtype,
    device=0 if torch.cuda.is_available() else -1,
)

print("Whisper ready!")

print("Loading Chatterbox...")

tts = ChatterboxMultilingualTTS.from_pretrained(
    device="cuda" if torch.cuda.is_available() else "cpu"
)

print("Chatterbox ready!")

print("Transcribing...")

'''result = whisper(
    INPUT_AUDIO,
    generate_kwargs={
        "task": "transcribe"
    }
)'''

#text = result["text"]
text = "Namaste, aap kaise hain?"

print("\nTranscription:")
print(text)

print("\nGenerating cloned speech...")

wav = tts.generate(
    text=text,
    language_id="en",
    audio_prompt_path=REFERENCE_AUDIO
)

ta.save(
    OUTPUT_AUDIO,
    wav,
    tts.sr
)

print("\nSaved:")
print(OUTPUT_AUDIO)