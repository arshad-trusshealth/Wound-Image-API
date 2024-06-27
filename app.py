from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import numpy as np
import torch
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
import os
import base64
import gdown
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
device = "cpu"

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
    # image_bytes = image.read()
    encoded = base64.b64encode(image).decode('utf-8')
    return encoded

def sliding_window(image, step_size, window_size):
    # Slide a window across the image
    for y in range(0, image.shape[0] - window_size[1] + 1, step_size):
        for x in range(0, image.shape[1] - window_size[0] + 1, step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


def pad_image(image, window_size):
    # Pad the image to make sure dimensions are multiples of window_size
    pad_height = (window_size[0] - image.shape[0] % window_size[0]) % window_size[0]
    pad_width = (window_size[1] - image.shape[1] % window_size[1]) % window_size[1]
    padded_image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
    return padded_image


def segment_image(model, image, window_size=(512, 512), step_size = 256):
    model.eval()
    # Load and preprocess the image
    # image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    image_np_padded = pad_image(image_np, window_size)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.4843, 0.3917, 0.3575], std = [0.2620, 0.2456, 0.2405])
    ])
    
    # Initialize the segmented image
    segmented_image = np.zeros((image_np_padded.shape[0], image_np_padded.shape[1]), dtype=np.uint8)
    
    # Process patches
    for (x, y, patch) in sliding_window(image_np, step_size, window_size):
        patch = transform(patch).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(patch)['out'].cpu().numpy()
            # output = torch.sigmoid(output).squeeze().cpu().numpy()
        segmented_image[y:y + window_size[1], x:x + window_size[0]] = output
    
    return segmented_image[:image.size[1], :image.size[0]]


def overlay_mask_onto_image(mask, image):
    # image = Image.open(image_path).convert("RGB")
    mask_rgb = np.zeros_like(np.array(image))
    mask_rgb[:, :, 0] = (mask * 255)
    
    mask_pil = Image.fromarray(mask_rgb)
    mask_pil = ImageEnhance.Brightness(mask_pil).enhance(1)
    overlay = Image.blend(image, mask_pil, alpha = 0.5)
    return overlay



# @app.route("/")
# def index():
#     return render_template("index.html")

@app.route('/analyze_image', methods=['GET', 'POST'])
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
        overlay_image_mask_enc = overlay_image_mask.tobytes()
        encoded_image = base64.b64encode(overlay_image_mask_enc).decode()
        # plt.imshow(overlay_image_mask)
        # plt.title('Wound Segmented Image')
        # plt.axis('off')
        # plt.show()
        
        return jsonify({"segmented_image": encoded_image})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug = True , host='0.0.0.0' , port=8080)
