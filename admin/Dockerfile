# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Install boto3 for AWS Secrets Manager
RUN pip install --no-cache-dir boto3

# Copy the rest of the application code
COPY . .

# Set environment variables for AWS and Secret Name
ENV SECRET_NAME=apgpg-pgvector-secret

# Expose the port the app runs on
EXPOSE 80

# Command to run the Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=80", "--server.address=0.0.0.0"]