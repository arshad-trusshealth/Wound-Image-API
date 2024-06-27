from flask import Flask, request, jsonify
from PIL import Image, ImageEnhance
import numpy as np
import torch
import torchvision.transforms as transforms
import os
import io
import base64
import gdown
from flask_cors import CORS
 
app = Flask(__name__)
CORS(app)
device = "cuda" if torch.cuda.is_available() else "cpu"
 
file_id = '1lSAWay1qKiHNu0EiON1ROxmEwd5GgRhT'
output = 'model_trained_deeplab_resnet101.pt'
 
gdrive_url = f"https://drive.google.com/uc?id={file_id}"
 
if not os.path.exists(output):
    gdrive_url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(gdrive_url, output, quiet=False)
else:
    print(f"{output} already exists. Skipping download.")
 
trained_deeplab = torch.load(output, map_location=device)
 
def encode_image(image):
    encoded = base64.b64encode(image).decode('utf-8')
    return encoded
 
def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[0] - window_size[1] + 1, step_size):
        for x in range(0, image.shape[1] - window_size[0] + 1, step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])
 
def pad_image(image, window_size):
    pad_height = (window_size[0] - image.shape[0] % window_size[0]) % window_size[0]
    pad_width = (window_size[1] - image.shape[1] % window_size[1]) % window_size[1]
    padded_image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
    return padded_image
 
def segment_image(model, image, window_size=(512, 512), step_size=256):
    model.eval()
    image_np = np.array(image)
    image_np_padded = pad_image(image_np, window_size)
   
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4843, 0.3917, 0.3575], std=[0.2620, 0.2456, 0.2405])
    ])
   
    segmented_image = np.zeros((image_np_padded.shape[0], image_np_padded.shape[1]), dtype=np.uint8)
   
    for (x, y, patch) in sliding_window(image_np_padded, step_size, window_size):
        patch = transform(patch).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(patch)['out'].cpu().numpy().squeeze()
            # output = (output > 0.5).astype(np.uint8)
        segmented_image[y:y + window_size[1], x:x + window_size[0]] = output
   
    return segmented_image[:image_np.shape[0], :image_np.shape[1]]
 
def overlay_mask_onto_image(mask, image):
    mask_rgb = np.zeros_like(np.array(image))
    mask_rgb[:, :, 0] = (mask * 255)
   
    mask_pil = Image.fromarray(mask_rgb)
    mask_pil = ImageEnhance.Brightness(mask_pil).enhance(1)
    overlay = Image.blend(image, mask_pil, alpha=0.5)
    return overlay
 
@app.route('/analyze_image', methods=['POST'])
def analyze_wound():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
   
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
 
    try:
        image = Image.open(file.stream).convert('RGB')
        window_size = (512, 512)
        step_size = 256
        segmented_image = segment_image(trained_deeplab, image, window_size, step_size)
       
        overlay_image_mask = overlay_mask_onto_image(segmented_image, image)
        buffer = io.BytesIO()
        overlay_image_mask.save(buffer, format="PNG")
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.read()).decode('utf-8')
       
        return jsonify({"segmented_image": encoded_image})
   
    except Exception as e:
        return jsonify({"error": str(e)}), 500
 
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
 
