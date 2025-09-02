import time
import torch
from PIL import Image
import open_clip


DEFAULT_MODEL = "MobileCLIP-S2"
DEFAULT_PRETRAINED = "datacompdr"

_model = None
_preprocess = None
_tokenizer = None


def load_model(model_name: str = DEFAULT_MODEL, pretrained: str = DEFAULT_PRETRAINED):
    global _model, _preprocess, _tokenizer
    if _model is None:
        print("Loading MobileCLIP ...")
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device="cpu",
            precision="fp32",
        )
        _model.eval()
        _tokenizer = open_clip.get_tokenizer(model_name)
        print("MobileCLIP ready")
    return _model, _preprocess, _tokenizer


def classify(image_path: str, labels: list[str]):
    model, preprocess, tok = load_model()
    total_start = time.time()

    img = Image.open(image_path).convert("RGB")
    image = preprocess(img).unsqueeze(0)
    text = tok(labels)

    with torch.no_grad():
        pred_start = time.time()
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        pred_end = time.time()

    probs = text_probs[0].tolist()
    ranked = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)
    return {
        "labels": [r[0] for r in ranked],
        "probs": [float(r[1]) for r in ranked],
        "top1": {"label": ranked[0][0], "prob": float(ranked[0][1])},
        "total_time": round(time.time() - total_start, 2),
        "prediction_time": round(pred_end - pred_start, 2),
        "model": DEFAULT_MODEL,
    }


