# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies that might be needed by some Python packages
# (e.g., for pandas, numpy, or other scientific libraries if they have C extensions)
# This is a common set, might need adjustment based on specific errors during build.
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    # Add other system dependencies if needed
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's source code from your context 
# (the directory where the Dockerfile is) into the container at /app
COPY . .

# Expose port 8000 to the outside world
EXPOSE 8000

# Define the command to run your app using uvicorn
# This will run the FastAPI app defined in src/api/main.py
# Ensure your API_CHECKPOINT_TO_LOAD_PATH in src/api/main.py is either:
#   a) Set to a path accessible *within the container* (if you COPY checkpoints into the image),
#   b) Loaded via environment variables and mounted volumes at runtime, or
#   c) Fetched from a remote store like S3 on startup.
# For simplicity now, we assume it might be baked in or handled at runtime deployment.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"] 