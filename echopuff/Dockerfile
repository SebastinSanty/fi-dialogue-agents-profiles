# Dockerfile for Echopuff
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code into the container
COPY app/ .

# Expose the port your agent is listening on (ensure it's the same in your code)
EXPOSE 5380

# Run the service
CMD ["python", "main.py"]
