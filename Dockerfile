# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# RUN apt-get update && \
#     apt-get install -y python3 python3-pip curl && \
#     rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install the required packages
RUN pip3 install --no-cache-dir -r requirements.txt
# Copy the rest of the application into the container
COPY . .

# Set the environment variable for Flask
ENV FLASK_APP=app.py

# Run the application
CMD ["flask", "run", "app.py", "--host=0.0.0.0", "--port=8080"]