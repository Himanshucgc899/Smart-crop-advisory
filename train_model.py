import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle


# 1. Load dataset
data = pd.read_csv("C:/Users/himan/OneDrive/Documents/Desktop/SIH2025/Crop_recommendationV2.csv")
data.head()


print(data.shape)
data['label'].unique()
numeric_cols = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
    "soil_moisture",
    "soil_type",
    "sunlight_exposure",
    "wind_speed",
    "co2_concentration",
    "organic_matter",
    "irrigation_frequency",
    "crop_density",
    "pest_pressure",
    "fertilizer_usage",
    "growth_stage",
    "urban_area_proximity",
    "water_source_type",
    "frost_risk",
    "water_usage_efficiency"
    ]

for col in numeric_cols:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop rows with NaN
data = data.dropna()

# -----------------------------
# 3️⃣ Features and target
# -----------------------------
X = data.drop("label", axis=1)
y = data["label"]

# Encode categorical columns
categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# -----------------------------
# 4️⃣ Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# -----------------------------
# 5️⃣ Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# 6️⃣ Train Random Forest Model
# -----------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 7️⃣ Evaluate
# -----------------------------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# -----------------------------
# 8️⃣ Save artifacts
# -----------------------------
with open("crop_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Save individual categorical encoders (e.g., soil_type)
for col, le in label_encoders.items():
    filename = f"{col}_encoder.pkl"
    with open(filename, "wb") as f:
        pickle.dump(le, f)

print("✅ Model, Scaler, and Encoders saved successfully!")