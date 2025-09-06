from flask import Flask, request, jsonify
import requests
import json

app = Flask(__name__)

# URL of your ML backend
ML_BACKEND_URL = "http://127.0.0.1:5000/predict"

# Function to call ML backend
def get_crop_prediction(N, P, K, temperature, humidity, ph, rainfall):
    payload = {
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }
    response = requests.post(ML_BACKEND_URL, json=payload)
    crop = response.json().get("recommended_crop", "Unknown")
    return crop

@app.route("/webhook", methods=["POST"])
def webhook():
    # Get JSON request from Dialogflow
    req = request.get_json(force=True)
    
    # Extract parameters sent from Dialogflow intent
    parameters = req.get("queryResult", {}).get("parameters", {})
    
    try:
        N = parameters["N"]
        P = parameters["P"]
        K = parameters["K"]
        temperature = parameters["temperature"]
        humidity = parameters["humidity"]
        ph = parameters["ph"]
        rainfall = parameters["rainfall"]
    except KeyError:
        # Missing parameter
        return jsonify({"fulfillmentText": 
                        "Please provide all 7 values: N, P, K, temperature, humidity, pH, rainfall."})

    # Call ML backend
    recommended_crop = get_crop_prediction(N, P, K, temperature, humidity, ph, rainfall)
    
    # Send response back to Dialogflow
    return jsonify({"fulfillmentText": f"The recommended crop for your input is: {recommended_crop}"})

if __name__ == "__main__":
    app.run(port=5001, debug=True)  # Different port than ML backend
