# app.py: Aplikasi Flask untuk deteksi obesitas dan rekomendasi diet

import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import joblib
import json

app = Flask(__name__)

# --- Muat Aset Model dan Data Pendukung ---
MODEL_FILENAME = 'best_rf_obesity_model.pkl'
MEAL_SUGGESTION_CSV = 'meal_suggestions.csv'
SELECTED_FEATURES_FILE = 'selected_features.pkl'
OBESITY_MAPPING_FILE = 'obesity_mapping.json'
GENDER_MAPPING_FILE = 'gender_mapping.json'
ACTIVITY_MAP_FILE = 'activity_map.json'

loaded_model = None
meal_suggestion_df = pd.DataFrame()
selected_features = []
obesity_order = ['Underweight', 'Normal', 'Overweight', 'Obese'] # Default jika gagal
gender_mapping_loaded = {}
activity_map_loaded = {}

try:
    loaded_model = joblib.load(MODEL_FILENAME)
    meal_suggestion_df = pd.read_csv(MEAL_SUGGESTION_CSV)
    selected_features = joblib.load(SELECTED_FEATURES_FILE) # Daftar 10 fitur terpilih

    with open(OBESITY_MAPPING_FILE, 'r') as f:
        obesity_mapping_loaded = json.load(f)
    obesity_order = sorted(obesity_mapping_loaded, key=obesity_mapping_loaded.get) # Urutan kategori

    gender_mapping_loaded = joblib.load(GENDER_MAPPING_FILE)
    activity_map_loaded = joblib.load(ACTIVITY_MAP_FILE)

    print("Semua aset (model, data, mappings) berhasil dimuat.")
except FileNotFoundError as e:
    print(f"ERROR: File penting tidak ditemukan: {e}. Pastikan Anda sudah menjalankan train_model.py.")
except Exception as e:
    print(f"ERROR: Terjadi kesalahan saat memuat aset: {e}")


# --- Mapping Kualitatif ke Numerik (SESUAIKAN INI DENGAN DATA ASLI & RISET GIZI ANDA!) ---
# Nilai-nilai ini adalah CONTOH PERKIRAAN dan HARUS divalidasi/disesuaikan.
# Asumsi 10 fitur terpilih (setelah BMI dan penyakit dihapus) adalah:
# 'Weight', 'Height', 'Ages', 'Calories', 'Sodium',
# 'Daily Calorie Target', 'Protein', 'Calorie_Deviation', 'Sugar', 'Carbohydrates'
# (Ini hanya contoh, cek selected_features.pkl Anda yang sebenarnya!)

MAPPING_PORTION_TO_MACROS = {
    'Kecil': {'Calories': 1400, 'Protein': 60, 'Carbohydrates': 180, 'Fat': 45},
    'Sedang': {'Calories': 2000, 'Protein': 100, 'Carbohydrates': 280, 'Fat': 70},
    'Besar': {'Calories': 2600, 'Protein': 140, 'Carbohydrates': 380, 'Fat': 95},
}

MAPPING_SUGAR_FREQUENCY_TO_GRAMS = {
    'Hampir Tidak Pernah': 15,
    'Jarang (1-2x/minggu)': 40,
    'Kadang-kadang (3-4x/minggu)': 75,
    'Sering (5-6x/minggu)': 110,
    'Setiap Hari': 150,
}

# Nilai Default untuk fitur yang mungkin tidak ditanyakan langsung di form
# Ini penting jika fitur ini ada di selected_features tapi tidak ada input langsung
DEFAULT_SODIUM = 2400 # mg
DEFAULT_DAILY_CALORIE_TARGET = 2200 # kcal
DEFAULT_FIBER = 25 # gram

# --- Rute Halaman Utama (Form Input) ---
@app.route('/')
def home():
    if loaded_model is None:
        return "Aplikasi tidak dapat dimuat. Cek log server untuk error aset."
    return render_template('index.html',
                           gender_options=list(gender_mapping_loaded.keys()),
                           activity_options=list(activity_map_loaded.keys()),
                           portion_options=list(MAPPING_PORTION_TO_MACROS.keys()),
                           sugar_freq_options=list(MAPPING_SUGAR_FREQUENCY_TO_GRAMS.keys()))

# --- Rute Prediksi ---
@app.route('/predict', methods=['POST'])
def predict():
    if loaded_model is None:
        return "Model tidak berhasil dimuat. Silakan cek log server."

    try:
        # 1. Ambil data dari form HTML
        age = int(request.form['age'])
        height_cm = float(request.form['height'])
        weight = float(request.form['weight'])
        gender_str = request.form['gender']
        activity_str = request.form['activity_level']
        portion_size_str = request.form['portion_size']
        sugar_freq_str = request.form['sugar_frequency']

        # Ambil input untuk fitur yang ditanyakan langsung (jika masuk 10 besar)
        daily_calorie_target = float(request.form.get('daily_calorie_target', DEFAULT_DAILY_CALORIE_TARGET))
        calories_actual = float(request.form.get('calories_actual', MAPPING_PORTION_TO_MACROS['Sedang']['Calories'])) # Pakai estimasi dari porsi sbg default
        sodium = float(request.form.get('sodium', DEFAULT_SODIUM))
        fiber = float(request.form.get('fiber', DEFAULT_FIBER))


        # 2. Lakukan Mapping Kualitatif ke Numerik & Preprocessing
        gender_encoded = gender_mapping_loaded.get(gender_str)
        activity_encoded = activity_map_loaded.get(activity_str)

        macros_estimation = MAPPING_PORTION_TO_MACROS.get(portion_size_str, MAPPING_PORTION_TO_MACROS['Sedang'])
        calories_val = macros_estimation['Calories']
        protein_val = macros_estimation['Protein']
        carbohydrates_val = macros_estimation['Carbohydrates']
        fat_val = macros_estimation['Fat']

        sugar_val = MAPPING_SUGAR_FREQUENCY_TO_GRAMS.get(sugar_freq_str)

        # Hitung Calorie_Deviation (jika ada di 10 fitur teratas)
        calorie_deviation_val = calories_val - daily_calorie_target # Estimasi kalori vs target


        # Hitung BMI untuk tampilan di hasil (opsional)
        bmi_display = weight / (height_cm / 100)**2 if height_cm > 0 else 0

        # 3. Susun input_data_dict dengan SEMUA fitur yang mungkin dibutuhkan model
        # Termasuk yang dihitung/di-mapping dan yang default
        all_possible_features_data = {
            'Weight': weight,
            'Height': height_cm,
            'Ages': age,
            'Calories': calories_val, # Dari mapping porsi
            'Sodium': sodium, # Dari input form
            'Daily Calorie Target': daily_calorie_target, # Dari input form
            'Protein': protein_val, # Dari mapping porsi
            'Calorie_Deviation': calorie_deviation_val, # Dari perhitungan
            'Sugar': sugar_val, # Dari mapping frekuensi gula
            'Carbohydrates': carbohydrates_val, # Dari mapping porsi
            'Gender': gender_encoded, # Dari mapping
            'Activity Level': activity_encoded, # Dari mapping
            'Fat': fat_val, # Dari mapping porsi
            'Fiber': fiber, # Dari input form
            # Fitur lain yang mungkin ada di selected_features tapi tidak ditanyakan/dihitung
            # Beri nilai default 0 atau rata-rata jika masuk 10 besar Anda
            # Contoh: 'Breakfast Calories': 0, 'Lunch Protein': 0, dll.
            # Anda HARUS menyesuaikan ini sesuai selected_features.pkl Anda
            # Jika ada Diet_... (OHE), sertakan juga
            'Diet_Omnivore': 1, 'Diet_Pescatarian': 0, 'Diet_Vegan': 0, 'Diet_Vegetarian': 0,
        }

        # Filter dan urutkan input_data_dict agar sesuai dengan selected_features
        # Ini langkah KRUSIAL! selected_features harus sama persis dari train_model.py
        final_features_list = [all_possible_features_data.get(feature, 0) for feature in selected_features]
        final_features_array = np.array(final_features_list).reshape(1, -1)


        # 4. Lakukan prediksi
        prediction_encoded = loaded_model.predict(final_features_array)[0]
        obesity_status = obesity_order[prediction_encoded]

        # 5. Berikan rekomendasi diet berdasarkan status
        diet_recommendation = {}
        if meal_suggestion_df is not None and not meal_suggestion_df.empty:
            if obesity_status in ['Obese', 'Overweight']:
                recommendation = meal_suggestion_df.sample(1).to_dict(orient='records')[0]
            elif obesity_status == 'Underweight':
                recommendation = meal_suggestion_df.sample(1).to_dict(orient='records')[0]
            else: # Normal
                recommendation = meal_suggestion_df.sample(1).to_dict(orient='records')[0]

            diet_recommendation = {
                'breakfast': recommendation.get('Breakfast Suggestion', 'Tidak ada saran'),
                'lunch': recommendation.get('Lunch Suggestion', 'Tidak ada saran'),
                'dinner': recommendation.get('Dinner Suggestion', 'Tidak ada saran'),
                'snack': recommendation.get('Snack Suggestion', 'Tidak ada saran')
            }
        else:
            diet_recommendation = {'info': 'Data rekomendasi tidak tersedia.'}

        return render_template('result.html',
                               prediction_text=f'Status Gizi Anda: {obesity_status} (BMI: {bmi_display:.2f})',
                               recommendations=diet_recommendation)

    except KeyError as ke:
        return f"Terjadi kesalahan pada input form: Kolom '{ke}' tidak ditemukan. Pastikan semua nama 'name' di form HTML sesuai."
    except ValueError as ve:
        return f"Terjadi kesalahan pada nilai input: {ve}. Pastikan Anda memasukkan angka yang valid."
    except Exception as e:
        return f"Terjadi kesalahan tidak terduga dalam prediksi: {e}"

if __name__ == "__main__":
    app.run(debug=True)