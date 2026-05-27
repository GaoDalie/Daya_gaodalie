FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libreoffice libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY apps.py .

EXPOSE 10000
CMD ["uvicorn", "apps:app", "--host", "0.0.0.0", "--port", "10000"]
