# Base image Miniconda
FROM continuumio/miniconda3

WORKDIR /app

# Salin file conda.yaml dari folder unduhan 'environment'
COPY environment_files/conda.yaml .

# Buat lingkungan Conda dari file tersebut
RUN conda env create -n model-env -f conda.yaml && conda clean -a

# Salin folder model yang telah diunduh
COPY downloaded_model/ /app/model/

# Expose port
EXPOSE 8080

# Perintah default untuk menjalankan container
CMD ["conda", "run", "-n", "model-env", "mlflow", "models", "serve", "-m", "/app/model", "-h", "0.0.0.0", "-p", "8080"]