# app.py

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib 

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Muat model Random Forest yang sudah dilatih
try:
    # Cek NAMA FILE INI. Harus PERSIS sama dengan nama file model yang kamu simpan.
    model = joblib.load('rf_model.pkl') # <-- PASTIkan NAMA FILE INI BENAR
except FileNotFoundError:
    print("File model 'rf_model.pkl' tidak ditemukan. Pastikan file berada di direktori yang sama dengan app.py.")
    model = None
except Exception as e: # <-- Tambahkan ini untuk menangkap error lain
    print(f"Terjadi error saat memuat model: {e}")
    model = None

# Muat dataset rekomendasi makanan (PASTIKAN PATH FILE INI BENAR)
try:
    # Ganti dengan nama file & path yang benar dari dataset makanan Anda
    meal_suggestion_df = pd.read_csv("rekomendasi_menu.csv") 
except FileNotFoundError:
    print("File rekomendasi makanan ('rekomendasi_menu.csv') tidak ditemukan.")
    meal_suggestion_df = None


@app.route('/')
def home():
    """Merender halaman utama dengan form input."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Menerima input dari form, melakukan prediksi, dan menampilkan hasil."""
    if model is None:
        return "Model tidak berhasil dimuat. Silakan cek log server."

    try:
        # --- KUMPULKAN DATA DARI FORM ---
        # Data dasar untuk kalkulasi BMI
        age = int(request.form['age'])
        weight = float(request.form['weight'])
        height_cm = float(request.form['height'])

        # Data lain yang dibutuhkan oleh model
        daily_calorie_target = float(request.form['daily_calorie_target'])
        calories = float(request.form['calories'])
        lunch_calories = float(request.form['lunch_calories'])
        breakfast_fats = float(request.form['breakfast_fats'])
        dinner_carbs = float(request.form['dinner_carbohydrates_1']) # Sesuaikan nama dengan di HTML
        sugar = float(request.form['sugar'])
        breakfast_protein = float(request.form['breakfast_protein'])
        lunch_protein = float(request.form['lunch_protein'])

        # --- FEATURE ENGINEERING (seperti di notebook) ---
        # Hitung BMI dari tinggi dan berat 
        height_m = height_cm / 100
        bmi = weight / (height_m ** 2) if height_m > 0 else 0

        # --- SUSUN FITUR SESUAI URUTAN UNTUK MODEL ---
        # Urutan harus SAMA PERSIS dengan yang Anda berikan
        features_list = [
            bmi,
            daily_calorie_target,
            calories,
            age,
            lunch_calories,
            weight,
            breakfast_fats,
            dinner_carbs,
            sugar,
            breakfast_protein,
            lunch_protein
        ]
        
        final_features = [np.array(features_list)]

        # Lakukan prediksi
        prediction_encoded = model.predict(final_features)
        
        # Decode hasil prediksi ke label yang mudah dibaca (sesuaikan mapping ini jika perlu)
        obesity_status_map = {0: 'Normal', 1: 'Obesitas', 2: 'Kelebihan Berat Badan', 3: 'Kekurangan Berat Badan'}
        obesity_status = obesity_status_map.get(prediction_encoded[0], "Tidak Diketahui")

        # Berikan rekomendasi diet
        diet_recommendation = {}
        if meal_suggestion_df is not None:
            # Ambil satu baris acak sebagai rekomendasi
            recommendation = meal_suggestion_df.sample(1).to_dict(orient='records')[0]
            diet_recommendation = {
                'breakfast': recommendation.get('Breakfast Suggestion', 'Tidak ada saran'),
                'lunch': recommendation.get('Lunch Suggestion', 'Tidak ada saran'),
                'dinner': recommendation.get('Dinner Suggestion', 'Tidak ada saran'),
                'snack': recommendation.get('Snack Suggestion', 'Tidak ada saran')
            }

        # Render halaman hasil dengan membawa data
        return render_template('result.html', 
                               prediction_text=f'Status Anda: {obesity_status} (BMI: {bmi:.2f})',
                               recommendations=diet_recommendation)

    except Exception as e:
        return f"Terjadi kesalahan saat memproses data: {e}"


if __name__ == "__main__":
    app.run(debug=True)