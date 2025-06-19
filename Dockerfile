# Bagian 1: Base Image
FROM continuumio/miniconda3

# Bagian 2: Buat Lingkungan dari conda.yaml
WORKDIR /app

# Salin file conda.yaml dari folder unduhan 'environment'
COPY environment_files/environment/conda.yaml .

# Gunakan Conda untuk membuat lingkungan persis seperti saat training
RUN conda env create -n model-env -f conda.yaml && conda clean -a

# Bagian 3: Salin Artefak Model
# Salin sub-folder model dari path yang benar
COPY downloaded_model/model/ /app/model/

# Bagian 4: Konfigurasi & Eksekusi
# Expose port yang akan digunakan oleh server
EXPOSE 8080

# =====================================================================
# === PERUBAHAN KUNCI DI SINI =========================================
# =====================================================================
# Perintah default untuk menjalankan container
# Tambahkan flag '--env-manager local' untuk memberitahu MLflow agar tidak membuat virtual env baru
CMD ["conda", "run", "-n", "model-env", "mlflow", "models", "serve", "-m", "/app/model", "-h", "0.0.0.0", "-p", "8080", "--env-manager", "local"]