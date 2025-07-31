---
title: NutriPredict - Deteksi Obesitas & Rekomendasi Diet Personal
emoji: 🍎
colorFrom: purple
colorTo: purple # Pastikan ini salah satu warna yang diizinkan (red, yellow, green, blue, indigo, purple, pink, gray)
sdk: static     # <--- INI YANG BENAR UNTUK FLASK DI HUGGING FACE SPACES!
app_file: app.py
python_version: 3.9 # Sesuaikan dengan versi Python Anda (misal 3.8, 3.10, 3.11)
command: gunicorn --bind 0.0.0.0:7860 app:app
---

# NutriPredict

Sistem Prediksi Risiko Obesitas dengan Rekomendasi Diet Personal menggunakan Machine Learning.

## Fitur:
- **Deteksi BMI Cepat:** Klasifikasi status gizi berdasarkan BMI standar.
- **Rekomendasi Diet Personal:** Saran menu diet berdasarkan kebiasaan nutrisi yang dianalisis oleh model LightGBM.
- **Estimasi Nutrisi:** Detail makronutrien estimasi untuk setiap rekomendasi.
- **Antarmuka Intuitif:** Desain modern dan alur dua tahap untuk pengalaman pengguna yang mudah.

## Cara Menggunakan:
1. Masukkan data diri dasar untuk mendapatkan status BMI Anda.
2. Lanjutkan untuk memberikan estimasi kebiasaan makan Anda.
3. Dapatkan prediksi status gizi dan rekomendasi diet yang dipersonalisasi.

## Model
- **Algoritma:** LightGBM Classifier
- **Fitur:** 10 fitur terpenting yang relevan dengan nutrisi dan demografi.

## Data
- Dataset nutrisi dan kebiasaan makan.
- Dataset rekomendasi menu.