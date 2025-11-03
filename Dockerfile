# Imagen base compatible con Python 3.13
FROM python:3.13-slim-bookworm

# Evitar archivos cache y mejorar logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Crear directorio de la app
WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copiar solo los requirements e instalarlos
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copiar el resto de la app
COPY . .

# Exponer el puerto de Flask
EXPOSE 5000

# Comando de arranque
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app.app:app"]
