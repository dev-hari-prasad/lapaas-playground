from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
import tempfile
import time
import open_clip
from services.fastvlm_service import describe_image as vlm_describe
from services.mobileclip_service import classify as mc_classify

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# FastVLM model configuration
MID = "apple/FastVLM-0.5B"
IMAGE_TOKEN_INDEX = -200

# Global model variables (will be loaded on first request)
model = None
tokenizer = None
model_loaded_at = None
mc_model = None
mc_preprocess = None
mc_tokenizer = None

def load_model():
    """Load FastVLM model and tokenizer"""
    global model, tokenizer, model_loaded_at
    if model is None:
        print("Loading FastVLM model...")
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(MID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MID,
            dtype=torch.float32,
            device_map={"": "cpu"},
            trust_remote_code=True,
        )
        model_loaded_at = time.time()
        print(f"Model loaded successfully in {round(model_loaded_at - t0, 2)}s!")
    return model, tokenizer

def load_mobileclip(model_name: str = "MobileCLIP-S2", pretrained: str = "datacompdr"):
    """Load MobileCLIP model and preprocess."""
    global mc_model, mc_preprocess, mc_tokenizer
    if mc_model is None:
        print("Loading MobileCLIP model...")
        mc_model, _, mc_preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device="cpu",
            precision="fp32",
        )
        mc_model.eval()
        mc_tokenizer = open_clip.get_tokenizer(model_name)
        print("MobileCLIP loaded!")
    return mc_model, mc_preprocess, mc_tokenizer

@app.route('/')
def index():
    """Render the home page"""
    return render_template('home.html')

# New pages
@app.route('/fastvlm')
def page_fastvlm():
    return render_template('fastvlm.html')

@app.route('/mobileclip')
def page_mobileclip():
    return render_template('mobileclip.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """Analyze uploaded image with FastVLM"""
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        if not file.filename.lower().endswith(tuple('.' + ext for ext in allowed_extensions)):
            return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name

        try:
            # Load model if not already loaded
            model, tokenizer = load_model()

            # Start timing
            total_start = time.time()

            # Process image
            img = Image.open(temp_path).convert("RGB")

            # Prepare the prompt
            messages = [
                {"role": "user", "content": "<image>\nDescribe this image in detail."}
            ]

            # Apply template
            rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            pre, post = rendered.split("<image>", 1)
            pre_ids = tokenizer(pre, return_tensors="pt", add_special_tokens=False).input_ids
            post_ids = tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids

            # Insert image token
            img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
            input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(model.device)
            attention_mask = torch.ones_like(input_ids, device=model.device)

            # Preprocess image
            px = model.get_vision_tower().image_processor(images=img, return_tensors="pt")["pixel_values"]
            px = px.to(model.device, dtype=model.dtype)

            # Generate response
            with torch.no_grad():
                pred_start = time.time()
                out = model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    images=px,
                    max_new_tokens=300,  # Shorter for web interface
                    do_sample=True,
                    temperature=0.7,
                )
                pred_end = time.time()

            # Decode response
            response = tokenizer.decode(out[0], skip_special_tokens=True)

            # Calculate times
            total_time = time.time() - total_start
            prediction_time = pred_end - pred_start

            # Clean up temp file
            os.unlink(temp_path)

            return jsonify({
                'success': True,
                'description': response,
                'total_time': round(total_time, 2),
                'prediction_time': round(prediction_time, 2),
                'model_load_time': round((model_loaded_at - total_start), 2) if model_loaded_at and model_loaded_at >= total_start else None,
                'model': 'FastVLM-0.5B'
            })

        except Exception as e:
            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return jsonify({'error': f'Processing error: {str(e)}'}), 500

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.get('/healthz')
def health():
    return jsonify({'ok': True}), 200

# New API endpoints using modular services
@app.route('/api/fastvlm', methods=['POST'])
def api_fastvlm():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
        try:
            result = vlm_describe(temp_path)
            os.unlink(temp_path)
            return jsonify({
                'success': True,
                'description': result['text'],
                'total_time': result['total_time'],
                'prediction_time': result['prediction_time'],
                'model_load_time': result['model_load_time'],
                'model': result['model']
            })
        except Exception as e:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return jsonify({'error': f'Processing error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/mobileclip', methods=['POST'])
def mobileclip_classify():
    """Zero-shot classify image with MobileCLIP; returns probs and timing."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        labels = request.form.get('labels', '')
        if not labels.strip():
            return jsonify({'error': 'No labels provided'}), 400
        labels_list = [x.strip() for x in labels.split(',') if x.strip()]
        if not labels_list:
            return jsonify({'error': 'No valid labels'}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            request.files['image'].save(temp_file.name)
            temp_path = temp_file.name

        try:
            model, preprocess, tok = load_mobileclip()
            total_start = time.time()

            img = Image.open(temp_path).convert('RGB')
            image = preprocess(img).unsqueeze(0)
            text = tok(labels_list)

            with torch.no_grad():
                pred_start = time.time()
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                pred_end = time.time()

            total_time = time.time() - total_start
            probs = text_probs[0].tolist()
            ranked = sorted(zip(labels_list, probs), key=lambda x: x[1], reverse=True)
            os.unlink(temp_path)
            return jsonify({
                'success': True,
                'labels': [r[0] for r in ranked],
                'probs': [round(float(r[1]), 4) for r in ranked],
                'top1': {'label': ranked[0][0], 'prob': round(float(ranked[0][1]), 4)},
                'total_time': round(total_time, 2),
                'prediction_time': round(pred_end - pred_start, 2),
                'model': 'MobileCLIP-S2'
            })
        except Exception as e:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return jsonify({'error': f'Processing error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/mobileclip', methods=['POST'])
def api_mobileclip():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        labels = request.form.get('labels', '')
        if not labels.strip():
            return jsonify({'error': 'No labels provided'}), 400
        labels_list = [x.strip() for x in labels.split(',') if x.strip()]
        if not labels_list:
            return jsonify({'error': 'No valid labels'}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            request.files['image'].save(temp_file.name)
            temp_path = temp_file.name

        try:
            result = mc_classify(temp_path, labels_list)
            os.unlink(temp_path)
            return jsonify({
                'success': True,
                'labels': result['labels'],
                'probs': [round(float(p), 4) for p in result['probs']],
                'top1': {'label': result['top1']['label'], 'prob': round(float(result['top1']['prob']), 4)},
                'total_time': result['total_time'],
                'prediction_time': result['prediction_time'],
                'model': result['model']
            })
        except Exception as e:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return jsonify({'error': f'Processing error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
