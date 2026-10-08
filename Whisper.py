from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

model_id = "openai/whisper-large-v3-turbo"

print("Downloading Whisper Large-v3-Turbo...")

processor = AutoProcessor.from_pretrained(model_id)

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype="auto",
    use_safetensors=True,
)

print("Whisper Large-v3-Turbo downloaded successfully!")