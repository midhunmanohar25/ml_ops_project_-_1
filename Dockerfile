# # Use an official Python runtime as a parent image
# FROM python:3.12-slim

# # Set the working directory to /app
# WORKDIR /app

# # Copy the required files and directory into the container at /app
# COPY app.py /app/app.py
# COPY pipeline.joblib /app/pipeline.joblib
# COPY src/ /app/src/
# COPY requirements.txt /app/requirements.txt

# # Install any needed packages specified in requirements.txt
# RUN pip install -r requirements.txt

# # Copy files from S3 inside docker
# # RUN mkdir /app/models
# # RUN aws s3 cp s3://creditcard-project/models/model.joblib /app/models/model.joblib

# # Port
# EXPOSE 8000

# # Run app.py when the container launches
# CMD ["python", "app.py"]

# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory to /app
WORKDIR /app

# Install system dependencies if required (e.g., curl for health checks)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker caching layers
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire project structure into the container
# This guarantees Website/, src/, and app.py are all present
COPY . /app/

# Expose ports (8000 for FastAPI, 8501 for Streamlit default)
EXPOSE 8000
EXPOSE 8501