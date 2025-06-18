# Bagian 1: Base Image
# Gunakan base image Python yang ringan dan efisien
FROM python:3.9-slim

# Bagian 2: Setup Lingkungan
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Nginx digunakan sebagai reverse proxy yang tangguh di depan server model
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Bagian 3: Instalasi Dependensi Model
# Pertama, salin file requirements dari model yang sudah diunduh
COPY downloaded_model/requirements.txt /app/requirements.txt

# Kedua, instal dependensi tersebut menggunakan pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Bagian 4: Salin Artefak Model
# Salin seluruh isi folder model yang sudah diunduh ke dalam image
COPY downloaded_model/ /app/model/

# Bagian 5: Konfigurasi & Eksekusi
# Expose port yang akan digunakan oleh server
EXPOSE 8080

# Jalankan model server MLflow saat container dimulai
# Perintah ini akan menyajikan model yang ada di direktori /app/model
CMD ["mlflow", "models", "serve", "-m", "/app/model", "-h", "0.0.0.0", "-p", "8080"]