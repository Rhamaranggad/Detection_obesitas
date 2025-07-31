# app.py: Aplikasi Flask untuk deteksi obesitas dan rekomendasi diet personal

import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
import joblib # Untuk memuat model .pkl dan selected_features.pkl
import json   # Untuk memuat file .json mapping (hanya avg_macros_per_meal.json)

app = Flask(__name__)

# --- Muat Aset Model dan Data Pendukung ---
# --- TAMBAHAN: Direktori untuk Aset Model ---
MODEL_ASSETS_DIR = 'model_assets' # Nama folder yang sama

# --- NAMA FILE ASET (Ubah agar dimuat dari dalam folder) ---
MODEL_FILENAME = os.path.join(MODEL_ASSETS_DIR, 'best_rf_obesity_model.pkl')
MEAL_SUGGESTION_CSV = os.path.join(MODEL_ASSETS_DIR, 'meal_suggestions.csv')
SELECTED_FEATURES_FILE = os.path.join(MODEL_ASSETS_DIR, 'selected_features.pkl')
OBESITY_MAPPING_FILE = os.path.join(MODEL_ASSETS_DIR, 'obesity_mapping.pkl')
GENDER_MAPPING_FILE = os.path.join(MODEL_ASSETS_DIR, 'gender_mapping.pkl')
ACTIVITY_MAP_FILE = os.path.join(MODEL_ASSETS_DIR, 'activity_map.pkl')
AVG_MACROS_FILE = os.path.join(MODEL_ASSETS_DIR, 'avg_macros_per_meal.json')

loaded_model = None
meal_suggestion_df = pd.DataFrame()
selected_features = []
obesity_order = ['Underweight', 'Normal', 'Overweight', 'Obese'] # Default jika gagal
gender_mapping_loaded = {}
activity_map_loaded = {}
avg_macros_loaded = {} # Variabel untuk menampung rata-rata makro

try:
    loaded_model = joblib.load(MODEL_FILENAME)
    print(f"DEBUG: Model '{MODEL_FILENAME}' berhasil dimuat.")

    meal_suggestion_df = pd.read_csv(MEAL_SUGGESTION_CSV)
    print(f"DEBUG: Data rekomendasi '{MEAL_SUGGESTION_CSV}' berhasil dimuat.")

    selected_features = joblib.load(SELECTED_FEATURES_FILE)
    print(f"DEBUG: Fitur terpilih '{SELECTED_FEATURES_FILE}' berhasil dimuat.")

    # --- BAGIAN YANG DIBENAHI: Memuat mapping dengan joblib.load() ---
    # Memuat mapping kategori obesitas (.pkl)
    obesity_mapping_loaded = joblib.load(OBESITY_MAPPING_FILE) # <-- MENGGUNAKAN JOBLIB.LOAD()
    obesity_order = sorted(obesity_mapping_loaded, key=obesity_mapping_loaded.get)
    print(f"DEBUG: Mapping obesitas '{OBESITY_MAPPING_FILE}' berhasil dimuat.")

    # Memuat mapping gender (.pkl)
    gender_mapping_loaded = joblib.load(GENDER_MAPPING_FILE) # <-- MENGGUNAKAN JOBLIB.LOAD()
    print(f"DEBUG: Mapping gender '{GENDER_MAPPING_FILE}' berhasil dimuat.")

    # Memuat mapping activity level (.pkl)
    activity_map_loaded = joblib.load(ACTIVITY_MAP_FILE) # <-- MENGGUNAKAN JOBLIB.LOAD()
    print(f"DEBUG: Mapping activity '{ACTIVITY_MAP_FILE}' berhasil dimuat.")
    # --- AKHIR BAGIAN YANG DIBENAHI ---

    # Memuat rata-rata makronutrien (.json) - Ini tetap json.load()
    with open(AVG_MACROS_FILE, 'r') as f:
        avg_macros_loaded = json.load(f)
    print(f"DEBUG: Rata-rata makronutrien '{AVG_MACROS_FILE}' berhasil dimuat.")

    print("\nSemua aset (model, data, mappings) berhasil dimuat.")

except FileNotFoundError as e:
    print(f"\nERROR: File penting tidak ditemukan: {e}.")
    print("Pastikan Anda sudah menjalankan train_model.py dan file-file aset berada di direktori yang sama dengan app.py.")
    loaded_model = None
    meal_suggestion_df = pd.DataFrame()
    selected_features = []
    obesity_order = ['Underweight', 'Normal', 'Overweight', 'Obese']
    gender_mapping_loaded = {}
    activity_map_loaded = {}
    avg_macros_loaded = {}
except Exception as e:
    print(f"\nERROR: Terjadi kesalahan lain saat memuat aset: {e}")
    loaded_model = None
    meal_suggestion_df = pd.DataFrame()
    selected_features = []
    obesity_order = ['Underweight', 'Normal', 'Overweight', 'Obese']
    gender_mapping_loaded = {}
    activity_map_loaded = {}
    avg_macros_loaded = {}


# --- Fungsi Klasifikasi BMI (Digunakan di Tahap 1) ---
def classify_bmi(bmi):
    if bmi < 18.5: return 'Underweight'
    elif 18.5 <= bmi < 25: return 'Normal'
    elif 25 <= bmi < 30: return 'Overweight'
    else: return 'Obese'

# --- Mapping Kualitatif ke Numerik (Untuk Tahap 2 - Rekomendasi) ---
MAPPING_PORTION_TO_MACROS = {
    'Kecil': {'Calories': 1400, 'Protein': 60, 'Carbohydrates': 180, 'Fat': 45},
    'Sedang': {'Calories': 2000, 'Protein': 100, 'Carbohydrates': 280, 'Fat': 70},
    'Besar': {'Calories': 2600, 'Protein': 140, 'Carbohydrates': 380, 'Fat': 95},
}

MAPPING_SUGAR_FREQUENCY_TO_GRAMS = {
    'Hampir Tidak Pernah': 15, 'Jarang (1-2x/minggu)': 40, 'Kadang-kadang (3-4x/minggu)': 75,
    'Sering (5-6x/minggu)': 110, 'Setiap Hari': 150,
}

DEFAULT_SODIUM = 2400
DEFAULT_DAILY_CALORIE_TARGET = 2200
DEFAULT_FIBER = 25

# --- Rute Halaman Utama (Tahap 1: Input BMI Dasar) ---
@app.route('/')
def home():
    if loaded_model is None:
        return "Aplikasi tidak dapat dimuat. Cek log server untuk error aset."
    return render_template('index.html',
                           gender_options=list(gender_mapping_loaded.keys()),
                           activity_options=list(activity_map_loaded.keys()))

# --- Rute Prediksi BMI (Tahap 1: Proses BMI Dasar) ---
@app.route('/predict_bmi', methods=['POST'])
def predict_bmi():
    try:
        age = int(request.form['age'])
        height_cm = float(request.form['height'])
        weight = float(request.form['weight'])
        gender_str = request.form['gender']
        activity_str = request.form['activity_level']

        # Hitung BMI
        bmi_value = weight / (height_cm / 100)**2 if height_cm > 0 else 0
        obesity_status_bmi = classify_bmi(bmi_value)

        # Hitung Berat Ideal (BMI range)
        height_m = height_cm / 100
        ideal_weight_min_bmi = 18.5 * (height_m ** 2)
        ideal_weight_max_bmi = 24.9 * (height_m ** 2)
        ideal_weight_range = f"{ideal_weight_min_bmi:.0f}-{ideal_weight_max_bmi:.0f}"

        # Hitung Kebutuhan Kalori Harian (Estimasi BMR + PAL)
        if gender_str == 'Female':
            bmr = (10 * weight) + (6.25 * height_cm) - (5 * age) - 161
        else: # Male
            bmr = (10 * weight) + (6.25 * height_cm) - (5 * age) + 5
        
        pal_factors = {
            'Sedentary': 1.2,
            'Lightly Active': 1.375,
            'Moderately Active': 1.55,
            'Very Active': 1.725,
            'Extremely Active': 1.9
        }
        activity_factor = pal_factors.get(activity_str, 1.2) # Default Sedentary

        total_daily_calorie_needs = bmr * activity_factor

        # --- DEBUG PRINTS (TAMBAHKAN INI) ---
        print(f"DEBUG_BMI: age={age}, height={height_cm}, weight={weight}, gender={gender_str}, activity={activity_str}")
        print(f"DEBUG_BMI: bmi_value={bmi_value}, obesity_status_bmi={obesity_status_bmi}")
        print(f"DEBUG_BMI: ideal_weight_range={ideal_weight_range}")
        print(f"DEBUG_BMI: total_daily_calorie_needs={total_daily_calorie_needs}")
        # --- AKHIR DEBUG PRINTS ---

        return render_template('result_bmi.html',
                               age=age, height=height_cm, weight=weight,
                               gender=gender_str, activity=activity_str,
                               bmi_value=f"{bmi_value:.2f}",
                               obesity_status=obesity_status_bmi,
                               ideal_weight_range=ideal_weight_range,
                               total_daily_calorie_needs=f"{total_daily_calorie_needs:.0f}")

    except Exception as e:
        # --- DEBUG PRINT UNTUK ERROR (TAMBAHKAN INI) ---
        import traceback
        print(f"ERROR: Exception caught in predict_bmi: {e}")
        traceback.print_exc() # Ini akan mencetak traceback lengkap ke terminal
        # --- AKHIR DEBUG PRINT ---
        return f"Terjadi kesalahan pada input data diri: {e}. Pastikan semua input terisi dengan benar."

# --- Rute Form Rekomendasi (Tahap 2: Input Nutrisi Sederhana) ---
@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    if loaded_model is None:
        return "Model tidak berhasil dimuat untuk rekomendasi. Silakan cek log server."
    
    age = int(request.form['age'])
    height_cm = float(request.form['height'])
    weight = float(request.form['weight'])
    gender_str = request.form['gender']
    activity_str = request.form['activity_level']

    return render_template('recommendation_form.html',
                           portion_options=list(MAPPING_PORTION_TO_MACROS.keys()),
                           sugar_freq_options=list(MAPPING_SUGAR_FREQUENCY_TO_GRAMS.keys()),
                           age=age, height=height_cm, weight=weight,
                           gender=gender_str, activity=activity_str)

# --- Rute Proses Rekomendasi (Tahap 2: Proses Model ML) ---
@app.route('/show_recommendations', methods=['POST'])
def show_recommendations():
    if loaded_model is None:
        return "Model tidak berhasil dimuat untuk rekomendasi. Silakan cek log server."

    try:
        age = int(request.form['age'])
        height_cm = float(request.form['height'])
        weight = float(request.form['weight'])
        gender_str = request.form['gender']
        activity_str = request.form['activity_level']

        portion_size_str = request.form['portion_size']
        sugar_freq_str = request.form['sugar_frequency']

        daily_calorie_target = float(request.form.get('daily_calorie_target', DEFAULT_DAILY_CALORIE_TARGET))
        calories_actual = float(request.form.get('calories_actual', MAPPING_PORTION_TO_MACROS['Sedang']['Calories']))
        sodium = float(request.form.get('sodium', DEFAULT_SODIUM))
        fiber = float(request.form.get('fiber', DEFAULT_FIBER))

        gender_encoded = gender_mapping_loaded.get(gender_str)
        activity_encoded = activity_map_loaded.get(activity_str)

        macros_estimation = MAPPING_PORTION_TO_MACROS.get(portion_size_str, MAPPING_PORTION_TO_MACROS['Sedang'])
        calories_val = macros_estimation['Calories']
        protein_val = macros_estimation['Protein']
        carbohydrates_val = macros_estimation['Carbohydrates']
        fat_val = macros_estimation['Fat']

        sugar_val = MAPPING_SUGAR_FREQUENCY_TO_GRAMS.get(sugar_freq_str)

        calorie_deviation_val = calories_val - daily_calorie_target

        bmi_display = weight / (height_cm / 100)**2 if height_cm > 0 else 0

        all_possible_features_data = {
            'Weight': weight, 'Height': height_cm, 'Ages': age,
            'Calories': calories_val, 'Sodium': sodium, 'Daily Calorie Target': daily_calorie_target,
            'Protein': protein_val, 'Calorie_Deviation': calorie_deviation_val, 'Sugar': sugar_val,
            'Carbohydrates': carbohydrates_val, 'Gender': gender_encoded, 'Activity Level': activity_encoded,
            'Fat': fat_val, 'Fiber': fiber,
            'Diet_Omnivore': 1, 'Diet_Pescatarian': 0, 'Diet_Vegan': 0, 'Diet_Vegetarian': 0,
        }

        final_features_list = [all_possible_features_data.get(feature, 0) for feature in selected_features]
        final_features_array = np.array(final_features_list).reshape(1, -1)

        prediction_encoded = loaded_model.predict(final_features_array)[0]
        obesity_status_ml = obesity_order[prediction_encoded]

        diet_recommendation = {}
        if meal_suggestion_df is not None and not meal_suggestion_df.empty:
            if obesity_status_ml in ['Obese', 'Overweight']:
                recommendation = meal_suggestion_df.sample(1).to_dict(orient='records')[0]
            elif obesity_status_ml == 'Underweight':
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

        total_daily_calories_estimated = calories_val
        total_daily_protein_estimated = protein_val
        total_daily_carbs_estimated = carbohydrates_val
        total_daily_fat_estimated = fat_val

        return render_template('recommendation_result.html',
                               obesity_status_ml=obesity_status_ml,
                               recommendations=diet_recommendation,
                               avg_macros=avg_macros_loaded,
                               total_daily_calories_estimated=total_daily_calories_estimated,
                               total_daily_protein_estimated=total_daily_protein_estimated,
                               total_daily_carbs_estimated=total_daily_carbs_estimated,
                               total_daily_fat_estimated=total_daily_fat_estimated)

    except Exception as e:
        return f"Terjadai kesalahan tidak terduga dalam prediksi rekomendasi: {e}"

if __name__ == "__main__":
    # Ini hanya akan berjalan jika Anda menjalankan app.py secara langsung (lokal)
    # Saat deployment dengan Docker/Gunicorn, bagian ini diabaikan.
    print("Running Flask app locally (debug mode). For deployment, Gunicorn handles execution.")
    port = int(os.environ.get("PORT", 5000)) # Default lokal tetap 5000
    app.run(host="0.0.0.0", port=port, debug=True)