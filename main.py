import pandas as pd
import numpy as np
import joblib
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import uvicorn

app = FastAPI(title="🚢 Titanic Survival Predictor API")
app.mount("/static", StaticFiles(directory="static"), name="static")

# -------------------------------
# Predictor Class
# -------------------------------
class TitanicSurvivalPredictor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def preprocess_data(self, df):
        df = df.copy()
        df['Age'].fillna(df['Age'].median(), inplace=True)
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
        df['Sex_encoded'] = self.label_encoder.fit_transform(df['Sex'])
        X = df[['Pclass', 'Sex_encoded', 'Age']].values
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, df['Survived'].values if 'Survived' in df.columns else None

    def train_model(self, csv_file_path):
        df = pd.read_csv(csv_file_path)
        X, y = self.preprocess_data(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.model.fit(X_train, y_train)
        self.save_models()

    def save_models(self):
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, 'models/titanic_model.pkl')
        joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')

    def load_models(self):
        try:
            self.model = joblib.load('models/titanic_model.pkl')
            self.label_encoder = joblib.load('models/label_encoder.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            return True
        except FileNotFoundError:
            return False

    def predict_survival(self, gender, age, pclass):
        gender_encoded = 1 if gender.lower() == 'male' else 0
        input_data = np.array([[pclass, gender_encoded, age]])
        input_scaled = self.scaler.transform(input_data)
        prediction = self.model.predict(input_scaled)[0]
        prob = self.model.predict_proba(input_scaled)[0]
        return {
            "survival": int(prediction),
            "survival_text": "Survived" if prediction == 1 else "Did not survive",
            "probability_survive": float(prob[1]),
            "probability_not_survive": float(prob[0])
        }

# -------------------------------
# Initialize predictor
# -------------------------------
predictor = TitanicSurvivalPredictor()

# -------------------------------
# FastAPI Endpoints
# -------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    html_content = """
    <html>
    <head>
        <title>Titanic Survival Predictor</title>
        <style>
            body { font-family: Arial, sans-serif; background-color: #f0f8ff; text-align: center; padding: 50px; }
            .container { background-color: white; border-radius: 10px; padding: 40px; box-shadow: 0 0 10px rgba(0,0,0,0.1); display: inline-block; }
            h1 { color: #333; }
            form { margin-top: 30px; }
            input, select { padding: 10px; width: 200px; margin: 10px 0; }
            button { padding: 10px 20px; background-color: #28a745; color: white; border: none; cursor: pointer; }
            button:hover { background-color: #218838; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚢 Titanic Survival Predictor</h1>
            <form action="/predict-form" method="get">
                <label for="gender">Gender:</label><br>
                <select name="gender" required>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                </select><br>
                <label for="age">Age:</label><br>
                <input type="number" name="age" step="0.1" min="0" required><br>
                <label for="pclass">Passenger Class (1, 2, or 3):</label><br>
                <input type="number" name="pclass" min="1" max="3" required><br>
                <button type="submit">Predict</button>
            </form>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/predict-form", response_class=HTMLResponse)
def predict_from_form(gender: str, age: float, pclass: int):
    try:
        result = predictor.predict_survival(gender, age, pclass)
        return HTMLResponse(content=f"""
        <html><body style='font-family: Arial; background: #e8f4fc; padding: 40px;'>
        <h2>🧾 Prediction Result</h2>
        <p><b>Gender:</b> {gender}</p>
        <p><b>Age:</b> {age}</p>
        <p><b>Class:</b> {pclass}</p>
        <p><b>Prediction:</b> {result['survival_text']}</p>
        <p><b>Probability Survive:</b> {result['probability_survive']:.2f}</p>
        <p><b>Probability Not Survive:</b> {result['probability_not_survive']:.2f}</p>
        <a href="/">🔙 Go Back</a>
        </body></html>
        """)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict-manual")
def predict_manual(gender: str, age: float, pclass: int):
    try:
        return predictor.predict_survival(gender, age, pclass)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# CLI-based Execution for Local Mode
# -------------------------------
if __name__ == "__main__":
    print("\n🚀 Titanic Survival Predictor CLI")
    print("=" * 40)

    if not predictor.load_models():
        print("❌ Model not found. Please provide Titanic CSV path.")
        path = input("Enter Titanic CSV file path: ").strip()
        if not os.path.exists(path):
            print("❌ File not found.")
            exit()
        predictor.train_model(path)
        print("✅ Model trained and saved!")

    gender = input("Enter gender (male/female): ").strip()
    age = float(input("Enter age: ").strip())
    pclass = int(input("Enter class (1, 2, or 3): ").strip())

    result = predictor.predict_survival(gender, age, pclass)
    print("\n🧾 Prediction Result:")
    print(f"🧍 Gender: {gender}")
    print(f"🎂 Age: {age}")
    print(f"🚢 Class: {pclass}")
    print(f"🔮 Prediction: {result['survival_text']}")
    print(f"✅ Probability Survive: {result['probability_survive']:.2f}")
    print(f"❌ Probability Not Survive: {result['probability_not_survive']:.2f}")

    print("\n🌐 Starting FastAPI server at http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
