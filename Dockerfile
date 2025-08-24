# Imagem base leve com Python
FROM python:3.10-slim

# Instala dependências do sistema (ffmpeg e libsndfile para soundfile)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Define diretório de trabalho
WORKDIR /app

# Copia os arquivos do projeto
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY whisper-v3.py .

# Define porta padrão
EXPOSE 8000

# Comando de inicialização
CMD ["uvicorn", "whisper-v3:app", "--host", "0.0.0.0", "--port", "8000"]
