# 🏠 House Price Prediction (MLOps Project)

This project is an end-to-end Machine Learning pipeline using ANN and FastAPI.

## 🚀 Features
- ANN model trained on Boston Housing dataset
- FastAPI-based prediction API
- Feature scaling using StandardScaler
- Logging for monitoring
- Modular code structure

## 📂 Project Structure

--main.py # FastAPI app
--model.py # ANN model
--model.pth # trained model (not included in repo)
--scaler.pkl # scaler (not included)
--features.pkl # feature list


## ⚙️ Setup

```bash
pip install -r requirements.txt
uvicorn main:app --reload
