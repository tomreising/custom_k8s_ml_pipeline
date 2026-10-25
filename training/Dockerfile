FROM python:3.12-slim

RUN pip install --no-cache-dir scikit-learn==1.6.1

WORKDIR /app
COPY train.py .

CMD ["python", "train.py"]
