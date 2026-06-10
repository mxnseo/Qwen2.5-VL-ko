# Qwen2.5-VL-3B + SenseVoice-Small (STT) pipeline
# Voice input → SenseVoice STT → ko text → Qwen2.5-VL inference → Answer

# conda env setup (get_started.py)
"""
    conda activate qwen25vl

    pip install funasr
    pip install torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    pip install sounddevice

    sudo apt-get install libportaudio2

"""

import torch
import time
import numpy as np
import sounddevice as sd
import torchaudio
from funasr import AutoModel as STTModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# settting
SAMPLE_RATE = 16000
RECORD_SEC = 10
IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"

# Inference Time Check
def measure_inference(func):
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        result = func(*args, **kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.perf_counter()

        elapsed = t_end - t_start
        print(f"\n[{func.__name__}] Inference Time: {elapsed:.3f} s")
        if torch.cuda.is_available():
            print(f"VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        return result
    return wrapper

model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
    attn_implementation="sdpa",
    device_map="auto"
)

# ---

stt_model = STTModel(
    model="iic/SenseVoiceSmall",
    trust_remote_code=True,
    device="cuda"
)


# Voice record
def record_audio(duration=RECORD_SEC, sample_rate=SAMPLE_RATE):
    print(f"\n{duration}s say")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    print("________")
    return audio.flatten()


# STT inference
@measure_inference
def run_stt(audio_array):
    result = stt_model.generate(
        input=audio_array,
        cache={},
        language="ko",
        use_itn=True,
        batch_size_s=60,
    )
    raw = result[0]["text"]
    text = raw.split(">")[-1].strip() if ">" in raw else raw.strip()
    return text


# VLM Inference
@measure_inference
def run_vlm(question, image_url):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_url},
                {"type": "text",  "text": question},
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=128)
    generated_ids_trimmed = [
        out[len(in_):] for in_, out in zip(inputs.input_ids, generated_ids)
    ]
    answer = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return answer


"""
def pipeline(image_url=IMAGE_URL):
    print("\n-- start --\n")

    t_total = time.perf_counter()
    audio = record_audio()

    print("\n[STT] Voice → Text Inference...")
    question = run_stt(audio)
    print(f"Text: '{question}'")

    if not question:
        print("retry")
        return

    print("\n[VLM] Inference...")
    answer = run_vlm(question, image_url)
    print(f"Answer: {answer}")

    total = time.perf_counter() - t_total
    print(f"\nTime: {total:.3f}s")

"""

# Audio file test

def pipeline_from_file(audio_path, image_url=IMAGE_URL):
    # wav file test
    waveform, sr = torchaudio.load(audio_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    audio = waveform.squeeze().numpy()

    print(f"\n[STT] '{audio_path}' file Inference...")
    question = run_stt(audio)
    print(f"Text: '{question}'")

    print("\n[VLM] Inference...")
    answer = run_vlm(question, image_url)
    print(f"Answer: {answer}")

pipeline_from_file("test_audio.wav")


if __name__ == "__main__":
    # pipeline()
    pipeline_from_file("test_audio.wav")