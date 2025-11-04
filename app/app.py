from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
from torchvision import transforms
import io, os, gc

app = Flask(__name__)

# Ruta al modelo TorchScript
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'leaf_model_ultra.pt')



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
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_t = transform(img).unsqueeze(0).half()  # 🧠 Convertir imagen a float16

        # ✅ Cargar modelo TorchScript optimizado
        model = torch.jit.load(MODEL_PATH, map_location='cpu')
        model.eval()

        # Inferencia
        with torch.no_grad():
            outputs = model(img_t)
            pred = torch.argmax(outputs, 1).item()

        # Limpieza de memoria
        del model, img_t, outputs
        torch.cuda.empty_cache()

        # 🌿 Etiquetas de las clases
        labels = ["Chivato", "Lapacho", "Maracuyá", "Pata de Vaca"]
        predicted_label = labels[pred] if pred < len(labels) else "Desconocida"

        return jsonify({'prediction': predicted_label})

    except Exception as e:
        return jsonify({'error': str(e)})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
