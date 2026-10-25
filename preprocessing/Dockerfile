FROM python:3.12-slim

RUN pip install --no-cache-dir numpy==2.5.4

WORKDIR /app
COPY preprocess.py .

CMD ["python", "preprocess.py"]
