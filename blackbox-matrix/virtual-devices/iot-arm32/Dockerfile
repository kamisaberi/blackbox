# Simulates a Raspberry Pi or low-power IoT device
FROM arm32v7/python:3.10-alpine

WORKDIR /app

# Install dependencies (none needed for standard lib, but good practice)
# RUN pip install requests 

# Copy the actor script
COPY ../common/actor.py /app/actor.py

# Entry point
CMD ["python3", "-u", "/app/actor.py"]