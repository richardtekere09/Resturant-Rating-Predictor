# 🍽️ Restaurant Ratings Predictor

A simple and interactive Streamlit web app that predicts restaurant ratings based on key features like price, booking availability, and delivery service.

---

## 🚀 Features

- ✅ Predicts restaurant ratings using a trained `RandomForestRegressor`
- ✅ Uses `StandardScaler` for input normalization
- ✅ Interactive UI with Streamlit widgets
- ✅ Lightweight model loading with `joblib`
- ✅ Easy to deploy locally or on platforms like Streamlit Cloud

---

## 📊 Input Features

| Feature                   | Description                                                |
|--------------------------|------------------------------------------------------------|
| Average Cost for Two     | Estimated cost in local currency (e.g., 1000 for medium)   |
| Table Booking Available  | Yes / No                                                   |
| Online Delivery Available| Yes / No                                                   |
| Price Range              | 1 (Cheapest) to 4 (Most Expensive)                         |

---

## 🧠 Model Info

- **Type**: `RandomForestRegressor` (best estimator from GridSearchCV)
- **Scaler**: `StandardScaler` for numeric feature normalization
- **Exported Files**:
  - `gridsrfr_model.pkl` — saved model
  - `scaler_clean.pkl` — fitted scaler

---

