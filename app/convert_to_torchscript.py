import torch
from torchvision import models

# Ruta al modelo original
MODEL_PATH = "model/leaf_model.pth"

# Cargar checkpoint
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
class_names = checkpoint["class_names"]

# Crear el mismo modelo base
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Convertir a TorchScript (compilado)
scripted_model = torch.jit.script(model)

# Guardar el modelo compilado
torch.jit.save(scripted_model, "model/leaf_model.pt")

print("✅ Modelo convertido a TorchScript y guardado en model/leaf_model.pt")
