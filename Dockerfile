FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libreoffice \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .

EXPOSE 10000
CMD ["streamlit", "run", "app.py", \
     "--server.port=10000", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
