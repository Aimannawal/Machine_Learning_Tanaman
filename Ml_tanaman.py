import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn
import requests
from fastapi.middleware.cors import CORSMiddleware

df = pd.read_excel('dataset.xlsx') 
df = df.drop(columns=["Tanggal", "Waktu"], errors="ignore")

print("Preview data:\n", df.head())
print("\nTipe data tiap kolom:\n", df.dtypes)
print("\nJumlah data kosong per kolom:\n", df.isnull().sum())

cols_to_clean = ['Temperatur', 'Humidity', 'pH']

for col in cols_to_clean:
    df[col] = df[col].astype(str).str.replace('.', '', regex=False).astype(float)

print("\nData setelah dibersihkan:\n", df[cols_to_clean].head())
print("\nTipe data setelah dibersihkan:\n", df[cols_to_clean].dtypes)

label_counts = df['Label'].value_counts()
print("\nDistribusi Label:")
print(label_counts)

X = df.drop('Label', axis=1)
y = df['Label']

sss = StratifiedShuffleSplit(n_splits=1, test_size=1/3, random_state=42)

for train_idx, test_idx in sss.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print("\n=== SCALING FITUR ===")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Sebelum scaling (sampel 5 data pertama):")
print(X_train.iloc[:5])
print("\nSetelah scaling dengan StandardScaler (sampel 5 data pertama):")
print(pd.DataFrame(X_train_scaled[:5], columns=X.columns))

print("\nJumlah data per kelas:")
print("Total data:", len(df))
print("\nTraining set:")
train_distribution = y_train.value_counts().sort_index()
print(train_distribution)
print("\nTesting set:")
test_distribution = y_test.value_counts().sort_index()
print(test_distribution)

print("\nVerifikasi proporsi data (training : testing):")
for label in sorted(df['Label'].unique()):
    total = label_counts[label]
    train = train_distribution.get(label, 0)
    test = test_distribution.get(label, 0)
    print(f"{label}: {train}/{total} ({train/total:.2f}) : {test}/{total} ({test/total:.2f})")

print("\n=== MODEL TRAINING (NON-SCALED DATA) ===")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training accuracy: {train_accuracy:.4f}")

cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f}")

y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {test_accuracy:.4f}")

print("\n=== PENGECEKAN OVERFITTING / UNDERFITTING ===")
print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Selisih accuracy (training - test): {train_accuracy - test_accuracy:.4f}")

selisih = train_accuracy - test_accuracy

if selisih > 0.05:
    print("⚠️ WARNING: Model kemungkinan overfitting (training jauh lebih bagus dari test)")
elif selisih < -0.05:
    print("⚠️ WARNING: Model kemungkinan underfitting (test lebih bagus dari training)")
else:
    print("✅ Model cukup seimbang (balanced), tidak terindikasi overfitting maupun underfitting")

print("\n=== MODEL TRAINING (SCALED DATA) ===")
model_scaled = RandomForestClassifier(n_estimators=100, random_state=42)
model_scaled.fit(X_train_scaled, y_train)

y_train_pred_scaled = model_scaled.predict(X_train_scaled)
train_accuracy_scaled = accuracy_score(y_train, y_train_pred_scaled)
print(f"Training accuracy (scaled): {train_accuracy_scaled:.4f}")

y_pred_scaled = model_scaled.predict(X_test_scaled)
test_accuracy_scaled = accuracy_score(y_test, y_pred_scaled)
print(f"Test accuracy (scaled): {test_accuracy_scaled:.4f}")
print(f"Selisih accuracy scaled (training - test): {train_accuracy_scaled - test_accuracy_scaled:.4f}")

print("\n=== PERBANDINGAN MODEL (SCALED vs NON-SCALED) ===")
print(f"Test accuracy tanpa scaling: {test_accuracy:.4f}")
print(f"Test accuracy dengan scaling: {test_accuracy_scaled:.4f}")

if test_accuracy_scaled > test_accuracy:
    print("Model dengan scaling menghasilkan performa lebih baik")
    best_model = model_scaled
    X_test_eval = X_test_scaled
    y_pred = y_pred_scaled
else:
    print("Model tanpa scaling menghasilkan performa lebih baik atau setara")
    best_model = model
    X_test_eval = X_test

print("\n=== EVALUASI MODEL ===")
print("\nMetrics keseluruhan:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision (weighted): {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall (weighted): {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1-score (weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}")

print("\nClassification Report Detail:")
class_report = classification_report(y_test, y_pred, output_dict=True)

print("Per label metrics:")
print("-" * 80)
print(f"{'Label':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
print("-" * 80)

for label, metrics in class_report.items():
    if label not in ['accuracy', 'macro avg', 'weighted avg']:
        print(f"{label:<15} {metrics['precision']:.4f}     {metrics['recall']:.4f}     {metrics['f1-score']:.4f}     {metrics['support']}")

print("-" * 80)
print("\nFull Classification Report:")
print(classification_report(y_test, y_pred))


fitur_training = ['N', 'P', 'K', 'pH', 'Temperatur', 'Humidity']
data_baru = pd.DataFrame([[90, 42, 43, 6.50, 20.87, 82.00]], columns=fitur_training)

if best_model == model_scaled:
    data_baru_transformed = scaler.transform(data_baru)
    hasil_prediksi = best_model.predict(data_baru_transformed)
    prob_prediksi = best_model.predict_proba(data_baru_transformed)
else:
    hasil_prediksi = best_model.predict(data_baru)
    prob_prediksi = best_model.predict_proba(data_baru)

print("\nRekomendasi Tanaman untuk data baru:", hasil_prediksi[0])

classes = best_model.classes_

print("\nProbabilitas prediksi untuk setiap kelas:")
for i, kelas in enumerate(classes):
    print(f"{kelas}: {prob_prediksi[0][i]:.4f}")

joblib.dump(best_model, "model.pkl")

if best_model == model_scaled:
    joblib.dump(scaler, "scaler.pkl")

# Load model dan scaler
best_model = joblib.load("model.pkl")
try:
    scaler = joblib.load("scaler.pkl")
    model_scaled = best_model
except:
    scaler = None
    model_scaled = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

THINGSPEAK_CHANNEL_ID = "2944728"
THINGSPEAK_READ_API = "0G5OGUCPEA99RFU5"
THINGSPEAK_URL = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds.json?api_key={THINGSPEAK_READ_API}&results=1"

@app.get("/predict")
def predict():
    response = requests.get(THINGSPEAK_URL)
    if response.status_code != 200:
        return {"error": "Gagal mengambil data dari ThingSpeak"}
    
    feed = response.json()["feeds"][0]

    try:
        data_input = {
            "N": float(feed["field5"]),
            "P": float(feed["field6"]),
            "K": float(feed["field7"]),
            "Temperatur": float(feed["field2"]),
            "Humidity": float(feed["field1"]),
            "pH": float(feed["field4"]),
        }
        fitur_training = ['N', 'P', 'K', 'pH', 'Temperatur', 'Humidity']
        df = pd.DataFrame([[data_input[f] for f in fitur_training]], columns=fitur_training)
    except Exception as e:
        return {"error": f"Format data tidak valid: {e}"}

    if scaler and best_model == model_scaled:
        data_transformed = scaler.transform(df)
        hasil_prediksi = best_model.predict(data_transformed)
        prob_prediksi = best_model.predict_proba(data_transformed)
    else:
        hasil_prediksi = best_model.predict(df)
        prob_prediksi = best_model.predict_proba(df)

    classes = best_model.classes_
    prob_prediksi = prob_prediksi[0]

    top3_idx = prob_prediksi.argsort()[-3:][::-1]
    top3_rekomendasi = [
        {
            "tanaman": classes[i],
            "probabilitas": float(prob_prediksi[i])
        } for i in top3_idx
    ]

    label_prediksi = hasil_prediksi[0]

    return {
        "top_3_rekomendasi_tanaman": top3_rekomendasi,
        "data_input": data_input,
        "label_prediksi": label_prediksi
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
