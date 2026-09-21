# Runs the Flask agent on Hugging Face Spaces (Docker SDK, port 7860).
FROM python:3.12-slim

WORKDIR /app

# Force UTF-8 so Chinese text doesn't hit "ascii codec can't encode" in the slim image.
ENV PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV PORT=7860
EXPOSE 7860
CMD ["python", "app.py"]
