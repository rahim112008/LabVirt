FROM python:3.10-slim

WORKDIR /app

# Installation des dépendances système de base
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copie des fichiers du projet
COPY . .

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Exposition du port requis par Hugging Face
EXPOSE 7860

# Lancement de l'application Streamlit (remplacez app.py par le nom de votre fichier principal si nécessaire)
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.maxUploadSize=10240", "--server.maxMessageSize=10240"]
