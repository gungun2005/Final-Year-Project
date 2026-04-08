# Use Python 3.11
FROM python:3.11

# Set the working directory
WORKDIR /app

# Copy your files into the container
COPY . .

# Install libraries
RUN pip install --no-cache-dir -r requirements.txt

# Start the app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]