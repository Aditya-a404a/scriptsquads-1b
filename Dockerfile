# Use the specified official Python base image
FROM --platform=linux/amd64 python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy the entire 'app' directory and the requirements file into the container
COPY ./app /app
COPY ./requirements.txt /app

# Install necessary OS-level dependencies for libraries like OpenCV and PyMuPDF
# This prevents common errors during library installation or runtime
RUN apt-get update
RUN apt-get install -y --no-install-recommends libgl1-mesa-glx 
RUN rm -rf /var/lib/apt/lists/*
# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install the Python dependencies from requirements.txt
RUN pip install --timeout=600 -r requirements.txt
RUN pip install "numpy<2"

# Expose port 8000 to allow communication with the Uvicorn server
EXPOSE 8000

# --- MODIFIED: Replaced entrypoint.sh with a direct command ---
# This command starts the Uvicorn server in the background (&), waits 20 seconds
# for it to initialize, and then runs the utils.py script.
CMD ["/bin/sh", "-c", "uvicorn cold_start:app --host 0.0.0.0 --port 8000 & sleep 5 && python utils.py"]
