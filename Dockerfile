# Use a lightweight Python image
FROM python:3.9-slim
# Set the working directory
WORKDIR /C:\\Users\\hp\\projects
# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0
# Copy only requirements first (to use Docker caching)
COPY requirement.txt .
# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirement.txt

# Copy the rest of the project files
COPY . .

# Expose API port
EXPOSE 8000

# Command to run the FastAPI server
CMD ["uvicorn", "crowd:app", "--host", "0.0.0.0", "--port", "8000"]
