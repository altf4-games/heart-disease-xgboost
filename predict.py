import argparse
import pandas as pd
from xgboost import XGBClassifier

def engineer_features(df):
    _df = df.copy()
    _df["cp_restecg"] = _df["cp"] * _df["restecg"]
    _df["thalach_per_age"] = _df["thalach"] / (_df["age"] + 1)
    _df["oldpeak_slope"] = _df["oldpeak"] * _df["slope"]
    _df["chol_ratio"] = _df["chol"] / (_df["trestbps"] + 1)
    _df["ca_thal"] = _df["ca"] * _df["thal"]
    
    # Ensure columns match the exact order expected by XGBoost model
    expected_cols = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang",
        "oldpeak", "slope", "ca", "thal", "stress",
        "cp_restecg", "thalach_per_age", "oldpeak_slope", "chol_ratio", "ca_thal"
    ]
    return _df[expected_cols]

def predict(input_data, model_path="xgboost_model.json"):
    # Load model
    model = XGBClassifier()
    model.load_model(model_path)
    
    # Convert input dict to dataframe
    df = pd.DataFrame([input_data])
    
    # Engineer synthetic features matching training
    X = engineer_features(df)
    
    # Predict probabilities and class
    probs = model.predict_proba(X)[0]
    pred_class = model.predict(X)[0]
    
    print(f"Predicted Class: {pred_class} (severity out of 4, where 0 is no disease)")
    
    return pred_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict heart disease severity using trained XGBoost model.")
    parser.add_argument("--age", type=float, default=63)
    parser.add_argument("--sex", type=float, default=1, help="1=male, 0=female")
    parser.add_argument("--cp", type=float, default=1, help="chest pain type (1-4)")
    parser.add_argument("--trestbps", type=float, default=145, help="resting blood pressure")
    parser.add_argument("--chol", type=float, default=233, help="serum cholestoral in mg/dl")
    parser.add_argument("--fbs", type=float, default=1, help="fasting blood sugar > 120 mg/dl (1=true, 0=false)")
    parser.add_argument("--restecg", type=float, default=2, help="resting electrocardiographic results (0-2)")
    parser.add_argument("--thalach", type=float, default=150, help="maximum heart rate achieved")
    parser.add_argument("--exang", type=float, default=0, help="exercise induced angina (1=yes, 0=no)")
    parser.add_argument("--oldpeak", type=float, default=2.3, help="ST depression induced by exercise relative to rest")
    parser.add_argument("--slope", type=float, default=3, help="the slope of the peak exercise ST segment (1-3)")
    parser.add_argument("--ca", type=float, default=0, help="number of major vessels (0-3) colored by flourosopy")
    parser.add_argument("--thal", type=float, default=6, help="3=normal, 6=fixed defect, 7=reversable defect")
    parser.add_argument("--stress", type=float, default=50, help="fake stress level input (1-100)")
    parser.add_argument("--model", type=str, default="xgboost_model.json", help="Path to trained model file")

    args = parser.parse_args()
    
    # Convert args to dictionary (excluding the model path)
    input_features = vars(args).copy()
    model_path = input_features.pop("model")
    
    print("Input Features:", input_features)
    print("-" * 40)
    
    try:
        predict(input_features, model_path=model_path)
    except Exception as e:
        print(f"Error making prediction. Did you train the model first by running main.py? \nError: {e}")
