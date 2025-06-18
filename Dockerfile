# Bagian 1: Base Image
FROM continuumio/miniconda3

# Bagian 2: Buat Lingkungan dari conda.yaml
WORKDIR /app

# Salin file conda.yaml dari folder artefak yang sudah kita unduh semua
COPY artifacts/conda.yaml .

# Gunakan Conda untuk membuat lingkungan persis seperti saat training
RUN conda env create -n model-env -f conda.yaml && conda clean -a

# Bagian 3: Salin Artefak Model
# Salin sub-folder model dari dalam folder artefak ke dalam image
COPY artifacts/model/ /app/model/

# Bagian 4: Konfigurasi & Eksekusi
# Expose port yang akan digunakan oleh server
EXPOSE 8080

# Perintah default untuk menjalankan container
# 1. Aktifkan lingkungan conda 'model-env' yang baru kita buat
# 2. Jalankan mlflow models serve untuk menyajikan model
CMD ["conda", "run", "-n", "model-env", "mlflow", "models", "serve", "-m", "/app/model", "-h", "0.0.0.0", "-p", "8080"]