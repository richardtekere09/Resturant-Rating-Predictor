# test_model_load.py
import joblib

try:
    model = joblib.load("gridsrfr_model.pkl")
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Failed to load model:")
    print(repr(e))
