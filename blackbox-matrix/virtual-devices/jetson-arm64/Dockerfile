# Simulates an NVIDIA Jetson / Edge AI Gateway
FROM arm64v8/python:3.10-slim

WORKDIR /app
COPY ../common/actor.py /app/actor.py
CMD ["python3", "-u", "/app/actor.py"]