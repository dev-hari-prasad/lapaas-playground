import time
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_ID = "apple/FastVLM-0.5B"
IMAGE_TOKEN_INDEX = -200

_model = None
_tokenizer = None
_loaded_at = None


def load_model():
    global _model, _tokenizer, _loaded_at
    if _model is None:
        t0 = time.time()
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.float32,
            device_map={"": "cpu"},
            trust_remote_code=True,
        )
        _loaded_at = time.time()
        print(f"FastVLM loaded in {round(_loaded_at - t0, 2)}s")
    return _model, _tokenizer, _loaded_at


def describe_image(image_path: str, prompt: str = "Describe this image in detail."):
    model, tokenizer, loaded_at = load_model()

    total_start = time.time()

    # Prepare message and tokenize around the image token
    messages = [{"role": "user", "content": f"<image>\n{prompt}"}]
    rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    pre, post = rendered.split("<image>", 1)
    pre_ids = tokenizer(pre, return_tensors="pt", add_special_tokens=False).input_ids
    post_ids = tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids

    img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
    input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)

    img = Image.open(image_path).convert("RGB")
    px = model.get_vision_tower().image_processor(images=img, return_tensors="pt")["pixel_values"]
    px = px.to(model.device, dtype=model.dtype)

    with torch.no_grad():
        pred_start = time.time()
        out = model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            images=px,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
        )
        pred_end = time.time()

    response = tokenizer.decode(out[0], skip_special_tokens=True)

    return {
        "text": response,
        "total_time": round(time.time() - total_start, 2),
        "prediction_time": round(pred_end - pred_start, 2),
        "model_load_time": round((_loaded_at - total_start), 2) if _loaded_at and _loaded_at >= total_start else None,
        "model": "FastVLM-0.5B",
    }


