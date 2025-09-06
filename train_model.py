import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle


# 1. Load dataset
data = pd.read_csv("Crop_recommendationV2.csv")
data.head()


print(data.shape)
data['label'].unique()

# 2. Features (X) and target (y)
#features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "soil_type"]
X = data.drop("label", axis=1)
y = data["label"]

# 3. Encode crop labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Encode soil type if exists
if "soil_type" in data.columns:
    le = LabelEncoder()
    data["soil_type"] = le.fit_transform(data["soil_type"])

    # Save encoder with pickle
    with open("soil_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 5. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Train model (Random Forest)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 7. Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 8. Save model, scaler, and label encoder
with open("crop_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Model, Scaler, and Label Encoder saved successfully!")

