# Imagen base liviana de Python
FROM python:3.10-slim

# Evitar archivos cache
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Crear directorio de la app
WORKDIR /app

# Instalar dependencias necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copiar solo archivos esenciales
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de la app
COPY . .

# Exponer puerto
EXPOSE 5000

# Comando para ejecutar Flask
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app.app:app"]
