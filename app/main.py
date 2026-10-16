import csv
from pathlib import Path
from fastapi import HTTPException
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
import joblib
import pandas as pd
import requests
import os
import time
from threading import Lock
# Load configuration from the project-root .env file.
load_dotenv()
os.getenv
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_PUBLISHABLE_KEY = os.getenv("SUPABASE_PUBLISHABLE_KEY", "")


app = FastAPI(title="CropAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://rococo-monstera-5b4848.netlify.app",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Weather and geocoding cache
_api_cache = {}
_api_cache_lock = Lock()

WEATHER_CACHE_SECONDS = 1800  # 30 minutes
GEOCODING_CACHE_SECONDS = 86400  # 24 hours



def get_cached_json(url, params, cache_key, ttl, allow_unavailable=False):
    now = time.time()

    with _api_cache_lock:
        cached = _api_cache.get(cache_key)

    if cached and now - cached["time"] < ttl:
        return cached["data"]

    try:
        response = requests.get(url, params=params, timeout=15)

        if response.status_code == 429:
            print("Open-Meteo 429 response:", response.text[:1000])

        response.raise_for_status()
        data = response.json()

        with _api_cache_lock:
            _api_cache[cache_key] = {
                "time": now,
                "data": data
            }

        return data

    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None

        if status == 429 and cached:
            print("Using stale cached data for:", cache_key)
            return cached["data"]

        if status == 429 and allow_unavailable:
            return None

        if status == 429:
            raise HTTPException(
                status_code=503,
                detail="Weather service is temporarily rate-limited. Please try again later."
            ) from exc

        raise HTTPException(
            status_code=502,
            detail="Weather service returned an error."
        ) from exc

    except requests.exceptions.RequestException as exc:
        if cached:
            print("Using stale cached data for:", cache_key)
            return cached["data"]

        if allow_unavailable:
            return None

        raise HTTPException(
            status_code=503,
            detail="Weather service is temporarily unavailable."
        ) from exc


# ======================================
# LOAD MODEL
# ======================================

model = joblib.load("models/crop_yield_model.pkl")

security = HTTPBearer(auto_error=False)


# ======================================
# AUTHENTICATION
# ======================================

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Ask Supabase Auth to validate the supplied access token.
    The HS256 shared secret is never stored in this application.
    """

    if not SUPABASE_URL or not SUPABASE_PUBLISHABLE_KEY:
        raise HTTPException(
            status_code=500,
            detail="Supabase backend configuration is missing."
        )

    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please sign in."
        )

    token = credentials.credentials

    try:
        response = requests.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "apikey": SUPABASE_PUBLISHABLE_KEY,
                "Authorization": f"Bearer {token}"
            },
            timeout=10
        )

        if response.status_code in (401, 403):
            raise HTTPException(
                status_code=401,
                detail="Your session is invalid or expired. Please sign in again."
            )

        response.raise_for_status()
        user = response.json()

        if not user.get("id"):
            raise HTTPException(
                status_code=401,
                detail="Could not verify your account."
            )

        return user

    except HTTPException:
        raise

    except requests.RequestException:
        raise HTTPException(
            status_code=503,
            detail="Unable to verify your session with Supabase. Try again."
        )


# ======================================
# HOME
# ======================================

@app.get("/")
def home():
    return {
        "message": "Crop Yield Prediction API is running!"
    }


# ======================================
# CROP YIELD PREDICTION
# AUTHENTICATED USERS ONLY
# ======================================

@app.post("/predict")
def predict(
    year: int,
    state: str,
    crop: str,
    season: str,
    area: float,
    rainfall: float,
    fertilizer: float,
    pesticide: float,
    current_user: dict = Depends(get_current_user)
):
    # Validate text inputs
    if not state.strip():
        raise HTTPException(status_code=400, detail="State cannot be empty.")

    if not crop.strip():
        raise HTTPException(status_code=400, detail="Crop cannot be empty.")

    if not season.strip():
        raise HTTPException(status_code=400, detail="Season cannot be empty.")

    # Validate year
    if year < 2000 or year > 2026:
        raise HTTPException(
            status_code=400,
            detail="Year must be between 2000 and 2026."
        )

    # Validate numeric inputs
    if area <= 0:
        raise HTTPException(
            status_code=400,
            detail="Area must be greater than zero."
        )

    if rainfall < 0:
        raise HTTPException(
            status_code=400,
            detail="Rainfall cannot be negative."
        )

    if fertilizer < 0:
        raise HTTPException(
            status_code=400,
            detail="Fertilizer cannot be negative."
        )

    if pesticide < 0:
        raise HTTPException(
            status_code=400,
            detail="Pesticide cannot be negative."
        )

    # Prepare model input
    input_data = pd.DataFrame({
        "Year": [year],
        "State": [state.strip()],
        "Crop": [crop.strip()],
        "Season": [season.strip()],
        "Area": [area],
        "Annual_Rainfall": [rainfall],
        "Fertilizer": [fertilizer],
        "Pesticide": [pesticide]
    })

    # Predict yield
    prediction = model.predict(input_data)
    predicted_yield = round(float(prediction[0]), 2)

    recommendations = []

    if predicted_yield < 1:
        recommendations.append(
            "Predicted yield is low. Review soil health and irrigation needs."
        )
    elif predicted_yield < 5:
        recommendations.append(
            "Predicted yield is moderate. Monitor crop growth and nutrients."
        )
    else:
        recommendations.append(
            "Predicted yield is relatively high. Continue monitoring crop health."
        )

    if rainfall < 500:
        recommendations.append(
            "Annual rainfall is low. Check soil moisture and irrigation needs."
        )
    elif rainfall > 2000:
        recommendations.append(
            "Annual rainfall is high. Check drainage and waterlogging."
        )

    if fertilizer < 10000:
        recommendations.append(
            "Review fertilizer needs using soil testing and crop-specific guidance."
        )

    if pesticide < 50:
        recommendations.append(
            "Monitor for pests and use appropriate control measures when needed."
        )

    return {
        "state": state.strip(),
        "crop": crop.strip(),
        "season": season.strip(),
        "predicted_yield": predicted_yield,
        "recommendation": " ".join(recommendations)
    }


# ======================================
# LOCATION HELPER
# ======================================

STATE_CITIES = {
    "andhra pradesh": "Vijayawada",
    "arunachal pradesh": "Itanagar",
    "assam": "Guwahati",
    "bihar": "Patna",
    "chhattisgarh": "Raipur",
    "goa": "Panaji",
    "gujarat": "Ahmedabad",
    "haryana": "Chandigarh",
    "himachal pradesh": "Shimla",
    "jharkhand": "Ranchi",
    "karnataka": "Bengaluru",
    "kerala": "Thiruvananthapuram",
    "madhya pradesh": "Bhopal",
    "maharashtra": "Mumbai",
    "manipur": "Imphal",
    "meghalaya": "Shillong",
    "mizoram": "Aizawl",
    "nagaland": "Kohima",
    "odisha": "Bhubaneswar",
    "punjab": "Ludhiana",
    "rajasthan": "Jaipur",
    "sikkim": "Gangtok",
    "tamil nadu": "Chennai",
    "telangana": "Hyderabad",
    "tripura": "Agartala",
    "uttar pradesh": "Lucknow",
    "uttarakhand": "Dehradun",
    "west bengal": "Kolkata",
    "delhi": "Delhi",
    "jammu and kashmir": "Srinagar",
    "ladakh": "Leh"
}


def find_indian_location(state: str):
    search_name = STATE_CITIES.get(
        state.strip().lower(),
        state.strip()
    )

    geocoding_data = get_cached_json(
        "https://geocoding-api.open-meteo.com/v1/search",
        {
            "name": search_name,
            "count": 10,
            "language": "en",
            "format": "json",
            "countryCode": "IN"
        },
        ("geocoding", search_name.strip().lower()),
        GEOCODING_CACHE_SECONDS
    )

    results = geocoding_data.get("results", [])         

    return next(
        (
            item for item in results
            if item.get("country_code", "").upper() == "IN"
        ),
        None
    )


# ======================================
# CURRENT WEATHER
# ======================================


@app.get("/weather")
def get_weather(state: str, crop: str = "Rice"):
    location = find_indian_location(state)

    if not location:
        return {"error": f"Location not found in India: {state}"}

    latitude = location["latitude"]
    longitude = location["longitude"]
    api_key = os.getenv("WEATHER_API_KEY")

    if not api_key:
        return {
            "location": location["name"],
            "latitude": latitude,
            "longitude": longitude,
            "crop": crop,
            "temperature": None,
            "humidity": None,
            "precipitation": None,
            "wind_speed": None,
            "weather_advice": ["Weather API key is not configured."],
            "weather_status": "unavailable"
        }

    try:
        response = requests.get(
            "https://api.weatherapi.com/v1/current.json",
            params={
                "key": api_key,
                "q": f"{latitude},{longitude}"
            },
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            raise ValueError(data["error"].get("message", "Weather API error"))

        current = data["current"]
        temperature = current.get("temp_c")
        humidity = current.get("humidity")
        precipitation = current.get("precip_mm")
        wind_speed = current.get("wind_kph")

        advice = []

        if humidity is not None and humidity >= 80:
            advice.append("High humidity: monitor crops for fungal diseases.")

        if precipitation is not None and precipitation == 0:
            advice.append(
                "No current precipitation: check soil moisture before irrigating."
            )

        if precipitation is not None and precipitation > 10:
            advice.append(
                "Heavy current precipitation: check drainage and waterlogging."
            )

        if wind_speed is not None and wind_speed >= 30:
            advice.append(
                "Strong winds: check vulnerable crops and provide support where appropriate."
            )

        crop_advice = {
            "rice": "Monitor field water levels and maintain drainage according to the rice growth stage.",
            "wheat": "Check soil moisture, especially during flowering and grain filling.",
            "maize": "Monitor soil moisture and inspect plants for pests and leaf damage.",
            "sugarcane": "Monitor soil moisture and inspect stalks and leaves for pests.",
            "banana": "Ensure good drainage and support plants if strong winds threaten them.",
            "tomato": "Avoid prolonged leaf wetness and monitor for fungal disease symptoms.",
            "potato": "Avoid waterlogged soil and monitor foliage for disease symptoms."
        }

        advice.append(
            f"{crop}: " + crop_advice.get(
                crop.strip().lower(),
                "Monitor soil moisture, crop health, and local pest or disease symptoms."
            )
        )

        return {
            "location": location["name"],
            "latitude": latitude,
            "longitude": longitude,
            "crop": crop,
            "temperature": temperature,
            "humidity": humidity,
            "precipitation": precipitation,
            "wind_speed": wind_speed,
            "weather_advice": advice,
            "weather_status": "live"
        }

    except (requests.RequestException, ValueError, KeyError) as exc:
        print(f"WeatherAPI current weather error: {exc}")
        return {
            "location": location["name"],
            "latitude": latitude,
            "longitude": longitude,
            "crop": crop,
            "temperature": None,
            "humidity": None,
            "precipitation": None,
            "wind_speed": None,
            "weather_advice": [
                "Live weather is temporarily unavailable.",
                "Check a trusted local weather source before making irrigation or crop-protection decisions."
            ],
            "weather_status": "unavailable"
        }


# ======================================
# CROP STATISTICS
# ======================================

@app.get("/statistics")
def crop_statistics():

    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "train-00000-of-00001.parquet"
    )

    df = pd.read_parquet(dataset_path)

    crop_yields = (
        df.groupby("Crop")["Yield"]
        .median()
        .sort_values(ascending=False)
        .head(10)
    )

    season_counts = df["Season"].value_counts().to_dict()

    return {
        "total_records": int(len(df)),
        "total_crops": int(df["Crop"].nunique()),
        "total_states": int(df["State"].nunique()),
        "top_crops_by_median_yield": [
            {
                "crop": str(crop),
                "median_yield": round(float(value), 2)
            }
            for crop, value in crop_yields.items()
        ],
        "records_by_season": [
            {
                "season": str(season),
                "records": int(count)
            }
            for season, count in season_counts.items()
        ]
    }


# ======================================
# SEVEN-DAY FORECAST
# ======================================


@app.get("/forecast")
def get_forecast(state: str):

    location = find_indian_location(state)

    if not location:
        return {"error": f"Location not found in India: {state}"}

    latitude = location["latitude"]
    longitude = location["longitude"]
    api_key = os.getenv("WEATHER_API_KEY")

    if not api_key:
        return {
            "location": location["name"],
            "state": state,
            "forecast": [],
            "forecast_status": "unavailable",
            "message": "Weather API key is not configured."
        }

    try:
        response = requests.get(
            "https://api.weatherapi.com/v1/forecast.json",
            params={
                "key": api_key,
                "q": f"{latitude},{longitude}",
                "days": 3,
                "aqi": "no",
                "alerts": "no"
            },
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            raise ValueError(data["error"].get("message", "Weather API error"))

        forecasts = []

        for item in data.get("forecast", {}).get("forecastday", []):
            day = item.get("day", {})
            condition = day.get("condition", {})

            forecasts.append({
                "date": item.get("date"),
                "weather_code": condition.get("code"),
                "max_temperature": day.get("maxtemp_c"),
                "min_temperature": day.get("mintemp_c"),
                "rain_probability": day.get("daily_chance_of_rain"),
                "precipitation_mm": day.get("totalprecip_mm")
            })

        return {
            "location": location["name"],
            "state": state,
            "forecast": forecasts,
            "forecast_status": "live"
        }

    except (requests.RequestException, ValueError, KeyError) as exc:
        print(f"WeatherAPI forecast error: {exc}")
        return {
            "location": location["name"],
            "state": state,
            "forecast": [],
            "forecast_status": "unavailable",
            "message": "Live forecast is temporarily unavailable. Please check again later."
        }

@app.get("/model-performance")
def get_model_performance():
    report_path = Path(__file__).resolve().parent.parent / "crop_quality_report.csv"

    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Model evaluation report not found."
        )

    with report_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    return {"crops": rows, "total_crops": len(rows)}
