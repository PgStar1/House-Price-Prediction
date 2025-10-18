# 🏠 House Price Prediction Dashboard

This project is a **machine learning web application** that predicts house prices using user-provided features.  
It combines a trained **Random Forest Regressor** model with an interactive web interface built using **Dash** and **Bootstrap**.

The goal of this project is to demonstrate how a trained machine learning model can be integrated into a real-time dashboard that accepts user input and returns intelligent predictions.

---

## 📚 Project Overview

- **Problem:** Predict the sale price of a house based on its characteristics (e.g. quality, size, and garage capacity).  
- **Dataset:** Based on the [Ames Housing dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques), commonly used for regression tasks.  
- **Model:** `RandomForestRegressor` trained on preprocessed features.  
- **Interface:** Dash + Bootstrap dashboard for interactive prediction.  
- **Purpose:** To serve as an educational or prototype example of ML deployment.  

---

## 🧠 How It Works

1. The dataset (`train.csv`) is preprocessed using the `preprocess()` function.  
2. A `RandomForestRegressor` is trained using selected features.  
3. The model and its feature list are saved with `joblib`:  
   - `models/house_price_model.pkl`  
   - `models/model_features.pkl`  
4. The **Dash app** (`app.py`) lets users input feature values.  
5. When the user clicks **Predict**, the app loads the trained model, fills missing features, and returns the predicted price.  

---

## 💡 Features Used for Prediction

The dashboard uses the following input features:

- `OverallQual` – Overall material and finish quality  
- `GrLivArea` – Above-ground living area (square feet)  
- `GarageCars` – Garage capacity in number of cars  
- `TotalBsmtSF` – Total basement area (square feet)  

The backend automatically fills the remaining 29 features (used during training) with default zeros so that the input shape matches the trained model (33 features total).  

---


## 🚀 How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/House-Price-Prediction.git
cd House-Price-Prediction
