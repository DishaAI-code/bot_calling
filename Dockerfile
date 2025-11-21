# 1️⃣ Use official Python image
FROM python:3.10-slim

# 2️⃣ Set work directory
WORKDIR /app

# 3️⃣ Install system dependencies (important for audio + STT/TTS + ffmpeg)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4️⃣ Install pip packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5️⃣ Copy all backend files
COPY . .

# 6️⃣ Expose FastAPI port
EXPOSE 8000

# 7️⃣ Load .env automatically
ENV PYTHONUNBUFFERED=1

# 8️⃣ Run the application (same as your local: python app.py api)
CMD ["python", "app.py", "api"]
