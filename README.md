# Proyek CI/CD End-to-End untuk Model Klasifikasi Beras


## Ringkasan Proyek

Proyek ini adalah implementasi dari pipeline MLOps (Machine Learning Operations) yang menerapkan prinsip **Continuous Integration/Continuous Deployment (CI/CD)** untuk sebuah model klasifikasi gambar. Tujuannya adalah untuk mengotomatisasi seluruh siklus hidup model, mulai dari training, logging, versioning, hingga pembuatan aset deployment (image Docker) yang siap digunakan.

Setiap kali ada pembaruan pada kode sumber atau data, pipeline ini akan berjalan secara otomatis untuk memastikan bahwa versi model terbaru selalu terlatih, teruji, dan siap untuk di-deploy tanpa memerlukan intervensi manual.

## Fitur Utama

  - **Otomatisasi Training**: Proses training model dijalankan secara otomatis di lingkungan yang bersih dan terisolasi menggunakan GitHub Actions.
  - **Pelacakan Eksperimen**: Terintegrasi penuh dengan **MLflow** dan **DagsHub** untuk mencatat semua parameter, metrik, dan artefak dari setiap sesi training secara terpusat.
  - **Versioning Artefak di Git**: Setelah setiap training yang sukses, artefak model (file `MLmodel`, bobot, dll.) secara otomatis disimpan kembali ke dalam repositori melalui Pull Request (PR) yang dibuat oleh bot. Ini memungkinkan versioning model yang transparan langsung di Git.
  - **Dockerisasi Otomatis**: Model yang telah dilatih secara otomatis di-package ke dalam sebuah image Docker yang ringan dan efisien menggunakan `Dockerfile` kustom.
  - **Deployment Otomatis ke Registry**: Image Docker yang telah berhasil dibuat langsung di-push ke **Docker Hub**, membuatnya tersedia untuk ditarik dan dijalankan di lingkungan produksi, staging, atau lokal.

## Struktur Proyek

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── mlflow_ci.yml         # File workflow utama CI/CD
├── MLProject/
│   ├── modelling.py              # Skrip utama untuk training model
│   └── data_split/               # Folder berisi data latih, validasi, dan tes
│       ├── train/
│       ├── val/
│       └── test/
└── Dockerfile                    # Blueprint untuk membangun image Docker server model
```

## Alur Kerja Otomatis (CI/CD Pipeline)

Pipeline ini diatur dalam file `.github/workflows/mlflow_ci.yml` dan berjalan secara otomatis. Berikut adalah tahapan lengkapnya:

```mermaid
graph TD
    A[Push ke branch 'main'] --> B{Workflow Terpicu};
    B --> C[1. Setup Lingkungan Python & Instalasi Dependensi];
    C --> D[2. Menjalankan Training Model (`modelling.py`)];
    D --> E[3. Hasil (Metrik & Artefak) Tercatat di DagsHub];
    E --> F[4. Mengunduh Artefak dari Run Terbaru];
    F --> G[5. Membuat PR untuk Menyimpan Artefak ke Git];
    F --> H[6. Membangun Image Docker dari Artefak];
    H --> I[7. Mendorong Image ke Docker Hub];
```

1.  **Pemicu**: Workflow akan aktif setiap kali ada `push` ke branch `main`.
2.  **Setup Lingkungan**: GitHub Actions menyiapkan runner, menginstal Python `3.11`, dan menginstal semua dependensi yang dibutuhkan (`tensorflow`, `mlflow`, dll.) menggunakan `pip`.
3.  **Training & Logging**: Skrip `modelling.py` dieksekusi. Berkat `mlflow.autolog()`, semua parameter (seperti jumlah epoch), metrik (seperti akurasi dan loss), dan artefak (model yang telah dilatih, `requirements.txt`) secara otomatis dicatat ke DagsHub.
4.  **Pengambilan Artefak**: Setelah training selesai, workflow mengambil `run_id` dari eksekusi tersebut dan mengunduh semua artefak yang baru saja dibuat ke dalam runner.
5.  **Membuat Pull Request**: Workflow kemudian menyiapkan folder artefak dan menggunakan action `peter-evans/create-pull-request` untuk secara otomatis membuat PR. PR ini berisi versi terbaru dari artefak model untuk digabungkan ke dalam folder `MLProject/model_artifact/` di repositori.
6.  **Membangun Image Docker**: Menggunakan `Dockerfile` kustom, workflow membangun sebuah image Docker. `Dockerfile` ini menggunakan `requirements.txt` dari artefak untuk menciptakan lingkungan Python yang ringan dan konsisten, lalu menyalin file model ke dalamnya.
7.  **Push ke Docker Hub**: Image yang telah berhasil dibangun kemudian diberi tag `latest` dan didorong ke akun Docker Hub yang telah ditentukan, siap untuk digunakan.

## Cara Menggunakan

Untuk mereplikasi atau menjalankan alur kerja ini di repositori Anda sendiri, ikuti langkah-langkah berikut:

### 1\. Prasyarat

Pastikan Anda telah mengatur semua *secrets* yang diperlukan di repositori GitHub Anda di bawah `Settings > Secrets and variables > Actions`.

  - `DAGSHUB_USERNAME`: Username DagsHub Anda.
  - `DAGSHUB_TOKEN`: Token akses dari DagsHub.
  - `DOCKERHUB_USERNAME`: Username Docker Hub Anda.
  - `DOCKERHUB_TOKEN`: Token akses dari Docker Hub.
  - `PAT`: Personal Access Token GitHub dengan scope `repo` dan `workflow`, digunakan untuk membuat Pull Request secara otomatis.

### 2\. Memicu Workflow

Cukup lakukan `git push` ke branch `main`. Workflow akan berjalan secara otomatis. Anda dapat memantau progresnya di tab "Actions" pada repositori GitHub Anda.

### 3\. Hasil

Setelah workflow berhasil, Anda akan melihat:

  - Sebuah **Pull Request baru** yang dibuat oleh bot, siap untuk Anda review dan merge.
  - Sebuah **image Docker baru** dengan tag `latest` di akun Docker Hub Anda.

-----

*Proyek ini disusun dan dikonfigurasi oleh Tema Anggara.*
