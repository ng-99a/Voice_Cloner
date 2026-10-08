import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)
from langchain_ollama import ChatOllama

MODEL = "openai/whisper-large-v3-turbo"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print("Loading Whisper...")

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)

model.to(device)

processor = AutoProcessor.from_pretrained(MODEL)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=dtype,
    device=0 if torch.cuda.is_available() else -1,
)

print("Whisper ready!")

result = pipe(
    "audio/my_voice.wav",
    generate_kwargs={
        "language": "hi",
        "task": "transcribe"
    },
    return_timestamps=True
)


print("\nYou said:")
print(result["text"])


llm = ChatOllama(
    model="qwen3:8b",
    temperature=0
)

prompt = f"""
Convert the following Hindi transcription into natural Romanized Hinglish.

Rules:
- Preserve the meaning exactly.
- Use Roman/English letters only.
- Do not translate Hindi into formal English.
- Keep common English words in English.
- Do not add explanations.

Hindi:
{result["text"]}
"""

response = llm.invoke(prompt)

print(response.content)