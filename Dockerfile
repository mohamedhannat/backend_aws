# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the requirements file into the container at /usr/src/app
COPY requirements.txt ./

# Install the necessary dependencies and packages
RUN apt-get update && apt-get install -y \
    gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-dev \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN apt install -y libpq-dev
# Copy the rest of the working directory contents into the container at /usr/src/app
COPY . .

# Expose port 5000 for the Flask app
EXPOSE 5000

# Run app.py when the container launches
CMD ["python3", "app.py"]

