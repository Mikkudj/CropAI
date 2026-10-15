# 🌱 CropAI — Smart Farming Assistant

CropAI is an AI-powered web application that predicts crop yield using historical agricultural data and provides weather information and general farming recommendations.

## Features

- Crop yield prediction using a trained machine learning model
- Current weather information
- 7-day weather forecast
- General farming recommendations
- Supabase authentication
- User-specific prediction history
- Model performance dashboard covering 62 crops

## Tech Stack

- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Python, FastAPI, Uvicorn
- **Machine Learning:** Scikit-learn, Random Forest Regressor
- **Data Processing:** Pandas, NumPy
- **Database and Authentication:** Supabase
- **Charts:** Chart.js

## Machine Learning

The model uses these input features:

- Year
- State
- Crop
- Season
- Cultivated area
- Annual rainfall
- Fertilizer usage
- Pesticide usage

The dataset contains 17,400 training records and 4,350 test records, covering 62 crops.

### Initial Evaluation Results

| Metric | Result |
|---|---:|
| MAE | 6.69 |
| RMSE | 81.68 |
| R² Score | 0.9925 |

These results are from an initial random test split. Performance varies by crop, and some crops have large prediction errors. Further validation is needed before real-world agricultural use.

## Project Structure

```text
CropAI/
├── app/
│   ├── main.py
│   └── predict.py
├── data/
│   ├── train-00000-of-00001.parquet
│   └── test-00000-of-00001.parquet
├── frontend/
│   ├── index.html
│   ├── performance.html
│   └── style.css
├── models/
│   └── crop_yield_model.pkl
├── notebook/
│   ├── 01_data_exploration.py
│   └── 02_model.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup Instructions

### Prerequisites

- Python 3.11
- Git
- Git LFS
- Visual Studio Code
- A Supabase project

### 1. Clone the repository

```powershell
git clone https://github.com/Mikkudj/CropAI.git
cd CropAI
git lfs pull
```

### 2. Create a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure Supabase

Copy `.env.example` to `.env` in the project root.

Add your own Supabase project URL and publishable key:

```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_PUBLISHABLE_KEY=your_supabase_publishable_key
```

Configure the required Supabase database tables and Row Level Security policies.

**Security:** Never commit `.env`, secret keys, service-role keys, or access tokens to GitHub.

### 5. Start the backend

From the project root:

```powershell
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The API documentation will be available at:

http://127.0.0.1:8000/docs

### 6. Open the frontend

Open `frontend/index.html` using the VS Code Live Server extension.

## Future Improvements

- Improve accuracy for crops with large prediction errors.
- Evaluate performance on unseen years and regions.
- Improve input validation and error handling.
- Deploy the frontend and backend.
- Enhance recommendations with verified agricultural guidance.

## Disclaimer

CropAI provides estimates based on historical data. Predictions and recommendations should not replace professional agricultural advice.

## Project Purpose

This project demonstrates machine learning, API development, authentication, database integration, and data visualization applied to agriculture.