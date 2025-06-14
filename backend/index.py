from flask import Flask, request, jsonify
import joblib
import numpy as np
from groq import Groq

# Load your trained model
model = joblib.load("best_model.pkl")

# Setup Groq client with your API key
client = Groq(api_key="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")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = [
            data["glucose"],
            data["cholesterol"],
            data["hematocrit"],
            data["insulin"],
            data["bmi"],
            data["systolic_blood_pressure"],
            data["diastolic_blood_pressure"],
            data["hba1c"],
            data["ldl_cholesterol"],
            data["hdl_cholesterol"],
            data["heart_rate"],
            data["troponin"]
        ]

        # Predict disease
        prediction = model.predict([features])[0]
        disease_name = str(prediction)  # Or decode using your label_encoder if needed

        # Generate treatment plan using Groq LLaMA-4
        prompt = f"The patient has {disease_name}. Provide a professional medical treatment plan."
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=300,
            top_p=1,
            stream=False
        )
        treatment = completion.choices[0].message.content

        # Video filename placeholder (optional)
        video_filename = f"/static/videos/{disease_name.lower().replace(' ', '_')}.mp4"

        return jsonify({
            "disease": disease_name,
            "treatment": treatment,
            "video": video_filename
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
