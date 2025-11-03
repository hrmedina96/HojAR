from flask import Flask, render_template, request, jsonify
import torch
from torchvision import models, transforms
from PIL import Image
import io, os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'leaf_model.pth')

# Lazy load del modelo
model, class_names = None, None

def load_model():
    global model, class_names
    if model is None:
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        class_names = checkpoint['class_names']
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    return model, class_names

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No se envió ninguna imagen'})

        file = request.files['file']
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        model, class_names = load_model()

        img_t = transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_t)
            _, pred = torch.max(outputs, 1)
            label = class_names[pred.item()]

        return jsonify({'prediction': label})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
