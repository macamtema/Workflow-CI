# Bagian 1: Base Image
# Gunakan base image Python yang sesuai dengan versi di workflow Anda (misal: 3.11)
FROM python:3.12-slim

# Bagian 2: Setup Lingkungan
WORKDIR /app

# Bagian 3: Instalasi Dependensi
# Salin file requirements.txt dari LOKASI YANG BENAR di dalam build context
COPY artifacts_temp/model/requirements.txt .

# Gunakan pip untuk menginstal semua dependensi dari file tersebut
RUN pip install --no-cache-dir -r requirements.txt

# Bagian 4: Salin Artefak Model
# Salin seluruh isi dari sub-folder model
COPY artifacts_temp/model/ /app/model/

# Bagian 5: Konfigurasi & Eksekusi
# Expose port yang akan digunakan oleh server
EXPOSE 8080

# Perintah default untuk menjalankan container
CMD ["mlflow", "models", "serve", "-m", "/app/model", "-h", "0.0.0.0", "-p", "8080", "--env-manager", "local"]