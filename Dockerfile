# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
# --no-cache-dir: Disables the cache, which reduces the image size
# --trusted-host pypi.python.org: Can help avoid SSL issues in some networks
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . /app

# The command to run the application using Gunicorn
# Gunicorn is a production-ready WSGI server
# -w 4: Number of worker processes
# -k uvicorn.workers.UvicornWorker: The worker class for ASGI applications (FastAPI)
# -b 0.0.0.0:8000: Bind to all network interfaces on port 8000
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "src.api.main:app"]