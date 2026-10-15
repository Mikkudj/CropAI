# 🌱 CropAI — Smart Farming Assistant

An AI-powered crop yield prediction platform designed to help farmers estimate crop yields using historical agricultural data and access weather information and farming recommendations.

## ✨ Features

- **Crop Yield Prediction:** Predict crop yield using a trained machine learning model.
- **Weather Information:** View current weather conditions for the selected state.
- **Weather Forecast:** Access a 7-day weather forecast.
- **Farming Recommendations:** Receive general recommendations based on predicted yield and farm inputs.
- **User Authentication:** Sign up and sign in using Supabase Authentication.
- **Prediction History:** Save and view previous predictions.
- **User Data Protection:** Row Level Security (RLS) policies restrict prediction history access to the relevant user.
- **Model Performance Dashboard:** Compare prediction errors across 62 crops.

## 🛠️ Tech Stack

- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Python, FastAPI
- **Machine Learning:** Pandas, NumPy, Scikit-learn, Random Forest Regressor
- **Database & Authentication:** Supabase
- **Data Visualization:** Chart.js
- **Server:** Uvicorn

## 🤖 Machine Learning

The model is trained on historical agricultural data containing:

- Year
- State
- Crop
- Season
- Cultivated area
- Annual rainfall
- Fertilizer usage
- Pesticide usage

The target variable is crop yield. A preprocessing pipeline handles categorical features, and a Random Forest Regressor generates predictions.

The dataset contains 17,400 training records and 4,350 test records, covering 62 crops across 31 states or territories.

## 📊 Model Evaluation

The initial random test split produced the following results:

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 6.69 |
| Root Mean Squared Error (RMSE) | 81.68 |
| R² Score | 0.9925 |

These are preliminary evaluation results. Performance varies by crop, and unusually large errors occur for some crops. The model should be evaluated further before being used for real agricultural decisions.

## 🚀 Getting Started

### Prerequisites

- Python 3.11
- Git
- Visual Studio Code
- A Supabase project
- The project dataset and trained model files

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd crop-ai-project
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Install the required Python packages using a `requirements.txt` file.

```powershell
pip install -r requirements.txt
```

### 4. Configure Supabase

Configure the Supabase project URL and public client key in the frontend using your project's existing configuration.

Never commit Supabase secret or service-role keys, access tokens, or passwords to GitHub.

### 5. Start the backend

```powershell
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### 6. Open the frontend

Use the VS Code Live Server extension to open:

```text
frontend/index.html
```

The backend API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## 📁 Project Structure

```text
crop-ai-project/
├── app/
│   └── main.py
├── data/
│   ├── train-00000-of-00001.parquet
│   └── test-00000-of-00001.parquet
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── performance.html
├── models/
│   └── crop_yield_model.pkl
├── notebook/
│   └── 02_model.py
├── venv/
├── crop_quality_report.csv
└── README.md
```

## 🔮 Future Improvements

- Improve prediction accuracy for crops with large errors.
- Validate model performance on future years and unseen regions.
- Improve input validation and error handling.
- Deploy the frontend and backend for public access.
- Add more reliable region-specific soil and agricultural data.
- Improve recommendations using verified agricultural guidance.

## ⚠️ Disclaimer

CropAI provides estimates based on historical data. Predictions and recommendations are not a substitute for professional agricultural advice.

## 👨‍💻 Project Purpose

This project demonstrates the application of machine learning, API development, database integration, authentication, and data visualization to an agricultural use case.
