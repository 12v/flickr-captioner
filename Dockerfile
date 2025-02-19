FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install torch first so it doesn't include CUDA
RUN pip install --no-cache-dir torch torchvision --index-url=https://download.pytorch.org/whl/cpu

# Copy the app-specific requirements file
COPY app-requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r app-requirements.txt

# Copy the weights to the weights directory
COPY weights/decoder_gpu.pth weights/decoder_gpu.pth
COPY index.html index.html

COPY *.py .
COPY data/*.py ./data/
COPY model/*.py ./model/
COPY index.html .

# Expose the port the app runs on
EXPOSE 60606

# Command to run the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "60606"]
