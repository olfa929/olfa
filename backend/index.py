from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from groq import Groq

# Load model
model = joblib.load("best_model.pkl")

# Initialize Groq
client = Groq(api_key="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")

# Flask app setup
app = Flask(__name__, static_folder='static')
CORS(app)

# ✅ Hardcoded disease labels
diseases = ["Thalasse", "Diabetes", "HeartDi", "Anemia", "Thromboc"]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract 24 input features
        features = [
            data["glucose"],
            data["cholesterol"],
            data["hemoglobin"],
            data["platelets"],
            data["white_blood_cells"],
            data["red_blood_cells"],
            data["hematocrit"],
            data["mean_corpuscular_volume"],
            data["mean_corpuscular_hemoglobin"],
            data["mean_corpuscular_hemoglobin_concentration"],
            data["insulin"],
            data["bmi"],
            data["systolic_blood_pressure"],
            data["diastolic_blood_pressure"],
            data["triglycerides"],
            data["hba1c"],
            data["ldl_cholesterol"],
            data["hdl_cholesterol"],
            data["alt"],
            data["ast"],
            data["heart_rate"],
            data["creatinine"],
            data["troponin"],
            data["c_reactive_protein"]
        ]

        # Predict disease
        prediction = model.predict([features])[0]
        disease_name = diseases[prediction] if prediction < len(diseases) else "Unknown"

        # Generate treatment with Groq
        prompt = f"The patient has {disease_name}. Provide a professional medical treatment plan."
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=300
        )
        treatment = completion.choices[0].message.content

        # ✅ Always return default video
        video_url = "/static/videos/default.mp4"

        return jsonify({
            "disease": disease_name,
            "treatment": treatment,
            "video": video_url
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
