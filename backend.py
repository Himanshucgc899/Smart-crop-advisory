from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load saved artifacts
model = pickle.load(open("crop_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
soil_encoder = pickle.load(open("soil_encoder.pkl", "rb"))

# Feature order expected by the model
FEATURES = [
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

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # Build feature vector
        feature_values = []
        for f in FEATURES:
            if f not in data:
                return jsonify({"error": f"Missing feature: {f}"}), 400
            val = data[f]
            # Encode soil_type if feature
            if f == "soil_type":
                val = soil_encoder.transform([val])[0]
            feature_values.append(val)

        features_array = np.array(feature_values).reshape(1, -1)
        features_array = scaler.transform(features_array)

        prediction = model.predict(features_array)
        crop_name = label_encoder.inverse_transform(prediction)[0]

        return jsonify({"recommended_crop": crop_name})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)