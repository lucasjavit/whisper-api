# 🐍 Imagem base leve com Python
FROM python:3.10-slim

# 🛠️ Instala dependências do sistema necessárias para ffmpeg e soundfile
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 📁 Define diretório de trabalho
WORKDIR /app

# 📦 Copia e instala dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 🧠 Copia o código principal
COPY whisper-v3.py .

# 🔥 Expõe a porta padrão do FastAPI
EXPOSE 8000

# 🚀 Comando de inicialização
CMD ["uvicorn", "whisper-v3:app", "--host", "0.0.0.0", "--port", "8000"]
