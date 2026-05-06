# 🔬 Computer Vision 101: Interactive Lab

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://interactiv-lab.streamlit.app/)
[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Computer Vision 101: Interactive Lab** adalah sebuah *platform* web interaktif edukasional yang mendemonstrasikan konsep dasar hingga lanjutan di bidang pemrosesan citra digital, *Machine Learning*, dan *Deep Learning*. Aplikasi ini dikembangkan menggunakan bahasa pemrograman Python dan kerangka kerja *Streamlit*, memfokuskan antarmuka visual yang modern, bersih, dan intuitif.

Aplikasi dirancang sebagai purwarupa untuk mahasiswa pascasarjana Magister Teknik Informatika guna melakukan *hands-on learning* pada model Neural Network berkinerja tinggi langsung melalui peramban (browser) web, tanpa perlu persiapan lingkungan instalasi GPU kompleks.

dapat dilihat pada https://interactiv-lab.streamlit.app

---

## 🚀 Fitur Utama (Modul Pembelajaran)

Aplikasi ini dibagi menjadi 4 modul interaktif utama, di mana pengguna dapat mengunggah gambar dan secara *real-time* bereksperimen dengan parameter-parameter komputasi.

### 📐 Modul 1: Dasar & Sinyal Citra
*   **Manipulasi Spasial:** Transformasi geometri seperti skala dimensi (*resize*), matriks rotasi 2D, dan ekstraksi area sentral (*center crop*).
*   **Pipeline Filter:** 
    *   *Gaussian Blur:* Pelembutan sinyal untuk reduksi _noise_.
    *   *Sobel Edge Detection:* Pencarian kontur dan tepi gambar berdasarkan gradien intensitas.
    *   *Fourier Transform (FFT):* Representasi _magnitude spectrum_ citra ke dalam domain frekuensi (frekuensi spasial).

### 🤖 Modul 2: Machine Learning Klasik
*   **Color Quantization (K-Means Clustering):** Demonstrasi pembelajaran tak terarah (*unsupervised learning*) untuk mengompresi representasi rentang warna piksel gambar ke dalam palet 3D RGB centroid terbatas (nilai *K* yang dapat diatur pengguna).

### 🧠 Modul 3: CNN & Klasifikasi Arsitektur
*   **Inferensi Klasifikasi Gambar:** Mampu menebak kelas objek di dalam gambar dengan referensi 1000 kelas format dataset ImageNet.
*   **Arsitektur ResNet-50:** Mengimplementasikan jaringan *Residual Network* dalam 50 layer, sekaligus menjabarkan komparasi teori secara tekstual terkait bagaimana inovasi matriks *Skip Connection* berhasil menyelesaikan anomali matematik *Vanishing Gradient* yang terjadi pada AlexNet dan GoogLeNet.

### 🎯 Modul 4: Deteksi Objek Real-Time
*   **YOLOv8 Inference:** Pemrosesan Deteksi Multi-Objek berbasis arsitektur *Single-Stage Detector* dengan varian "Nano" untuk kecepatan ultra.
*   **Sistem Saringan Deteksi:** Pengguna dapat memodifikasi batas nilai probabilitas akurasi minimum (*Confidence Threshold*) dan kapasitas maksimum deteksi. Semua entitas yang lolos validasi disajikan ke dalam tabel (*Dataframe*) secara mendetail.

---

## 🛠️ Stack Teknologi & Pustaka Dependensi

Proyek ini dibangun secara modular menggunakan pustaka *Open Source* pilihan:

*   **Front-end & UI Routing:** [Streamlit](https://streamlit.io/)
*   **Komputasi & Manipulasi Tensor:** [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/)
*   **Operasi Dasar Visi Komputer:** [OpenCV (Headless)](https://opencv.org/), [Pillow (PIL)](https://python-pillow.org/)
*   **Mesin Algoritma Pemelajaran Klasik:** [Scikit-Learn](https://scikit-learn.org/)
*   **Jaringan Saraf Tiruan & Komputasi Mendalam (Deep Learning):** [PyTorch](https://pytorch.org/), [Torchvision](https://pytorch.org/vision/stable/index.html)
*   **Integrasi Model Pre-trained YOLOv8:** [Ultralytics](https://github.com/ultralytics/ultralytics)

---

## 💻 Instalasi Lokal (Environment Setup)

Apabila Anda ingin menjalankan, memodifikasi, atau berkontribusi pada proyek ini secara lokal, silakan ikuti petunjuk pengaturan konfigurasi berikut:

### 1. Prasyarat Sistem
*   Python 3.9 ke atas
*   Git 

### 2. Kloning Repositori
Kloning repositori proyek ini ke mesin lokal Anda menggunakan Git:
```bash
git clone https://github.com/delsonmanurung/interactiv-lab.git
cd interactiv-lab
```

### 3. Pembuatan Virtual Environment (Opsional namun sangat direkomendasikan)
Buat _virtual environment_ untuk mengisolasi dependensi proyek:
```bash
# Untuk MacOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# Untuk Windows Command Prompt
python -m venv .venv
.venv\Scripts\activate
```

### 4. Instalasi Pustaka
Instalasi semua *requirements* yang disyaratkan oleh aplikasi. 
*(Pastikan Anda menggunakan versi `opencv-python-headless` jika _deployment_ diarahkan ke server cloud).*
```bash
pip install -r requirements.txt
```

### 5. Menjalankan Aplikasi
Mulai peladen peladen lokal aplikasi (*localhost*):
```bash
streamlit run cv_app.py
```
Aplikasi akan secara otomatis meluncur di peramban sistem Anda via alamat `http://localhost:8501`.

---

## ☁️ Deployment (Streamlit Community Cloud)

Proyek ini disiapkan untuk mendukung penerapan kontinu (*Continuous Deployment*) ke **Streamlit Cloud**:
1. Hubungkan repositori GitHub ini dengan akun Streamlit Anda.
2. Di dalam dasbor Streamlit Cloud, klik **New App**.
3. Atur parameter jalur utama (*Main file path*) menuju berkas `cv_app.py`.
4. Tekan **Deploy!**

---

## 👤 Pengembang & Penghargaan
Aplikasi laboratorium ini direkayasa sebagai bagian dari kewajiban/eksplorasi akademis di Program Studi Magister Teknik Informatika (S2). 

---
> *Lisensi diatur secara terbuka di bawah atribusi MIT License. Segala modifikasi dan komersialisasi diizinkan.*
