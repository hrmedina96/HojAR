import torch

# Cargar tu modelo actual optimizado
model = torch.jit.load("app/model/leaf_model_optimized.pt", map_location='cpu')
model.eval()

# Reducir aún más: half precision + optimización para inferencia
model = model.half()
optimized = torch.jit.optimize_for_inference(model)

# Guardar como modelo ultra liviano
optimized.save("app/model/leaf_model_ultra.pt")
print("✅ Modelo ultra liviano guardado como leaf_model_ultra.pt")
