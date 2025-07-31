# Gunakan image Python resmi sebagai base image
FROM python:3.9-slim-buster # Sesuaikan versi Python Anda (misal: 3.8-slim-buster, 3.10-slim-buster)

# Atur direktori kerja di dalam container
WORKDIR /app

# Salin requirements.txt dan install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file aplikasi ke dalam container
COPY . .

# Tetapkan variabel lingkungan PORT yang diharapkan oleh Flask
ENV PORT 7860

# Perintah untuk menjalankan aplikasi Flask menggunakan Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]