FROM python:3.9-slim

WORKDIR /code

# Copy requirements first (for better caching)
COPY requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all application files
COPY . /code

# Expose port 7860 (Hugging Face standard)
EXPOSE 7860

# Run the Flask app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "app:app"]