from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
from torchvision import transforms
import io, os, gc

app = Flask(__name__)

# Ruta al modelo TorchScript
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'leaf_model.pt')

# Transformaciones de imagen
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
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img_t = transform(img).unsqueeze(0)

        # Cargar modelo TorchScript en CPU
        model = torch.jit.load(MODEL_PATH, map_location='cpu')
        model.eval()

        with torch.no_grad():
            outputs = model(img_t)
            pred = torch.argmax(outputs, 1).item()

        # 🔹 Mapeo de clases (según tu modelo)
        class_names = ['Chivato', 'Lapacho', 'Maracuyá', 'Pata de Vaca']

        # Enviar respuesta
        return jsonify({'prediction': class_names[pred]})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
