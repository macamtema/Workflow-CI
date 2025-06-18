# Bagian 1: Base Image
# Kita mulai dari image miniconda yang sudah memiliki Conda terinstal
FROM continuumio/miniconda3

# Bagian 2: Buat Lingkungan dari conda.yaml
WORKDIR /app

# Salin file conda.yaml dari path yang benar sesuai bukti dari log
COPY environment_files/environment/conda.yaml .

# Gunakan Conda untuk membuat lingkungan persis seperti saat training
RUN conda env create -n model-env -f conda.yaml && conda clean -a

# Bagian 3: Salin Artefak Model
# Salin sub-folder model dari path yang benar sesuai bukti dari log
COPY downloaded_model/model/ /app/model/

# Bagian 4: Konfigurasi & Eksekusi
# Expose port yang akan digunakan oleh server
EXPOSE 8080

# Perintah default untuk menjalankan container
# 1. Aktifkan lingkungan conda 'model-env' yang baru kita buat
# 2. Jalankan mlflow models serve untuk menyajikan model
CMD ["conda", "run", "-n", "model-env", "mlflow", "models", "serve", "-m", "/app/model", "-h", "0.0.0.0", "-p", "8080"]