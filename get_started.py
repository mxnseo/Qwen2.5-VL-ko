# huggingface - Qwen2.5-VL-3B-Instruct // Get Started
# https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
# git - mxnseo

# conda env 
"""
    -- RTX 4070 PC --

    conda create -n qwen25vl python=3.10 -y
    conda activate qwen25vl

    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

    # transformers (pip version -> qwen2_5_vl KeyError)
    pip install git+https://github.com/huggingface/transformers accelerate

    # Qwen VL util
    pip install qwen-vl-utils[decord]==0.0.8

"""

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import torch
import time

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


# Simple Inference (eng)

@measure_inference
def simple_inference():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
                {"type": "text", "text": "Can you describe this image?"},
            ]
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=256)
    generated_ids_trimmed = [
        out[len(in_):] for in_, out in zip(inputs.input_ids, generated_ids)
    ]
    generated_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(generated_texts[0])

simple_inference()



# Simple Inference (ko)

@measure_inference
def korean_inference():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
                {"type": "text", "text": "이 이미지에 무엇이 있나요? "},
            ]
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=256)
    generated_ids_trimmed = [
        out[len(in_):] for in_, out in zip(inputs.input_ids, generated_ids)
    ]
    generated_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(generated_texts[0])

korean_inference()




# Object recognition (ko, for project)

@measure_inference
def object_recognition():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": "hard.jpg"},
                {"type": "text", "text": "이게 뭔가요? 물건 이름과 어떤 상황인가요?"},
            ]
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=16)
    generated_ids_trimmed = [
        out[len(in_):] for in_, out in zip(inputs.input_ids, generated_ids)
    ]
    generated_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(generated_texts[0])

object_recognition()



# Multi-image Inference (ko)

@measure_inference
def multi_image_inference():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "두 이미지의 공통점은 무엇인가요?"},
                {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
                {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"},
            ]
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=256)
    generated_ids_trimmed = [
        out[len(in_):] for in_, out in zip(inputs.input_ids, generated_ids)
    ]
    generated_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(generated_texts[0])

multi_image_inference()
