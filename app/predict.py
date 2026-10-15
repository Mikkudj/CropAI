import pandas as pd
import joblib

# Load the trained model
model = joblib.load(
    "models/crop_yield_model.pkl"
)

print("\n===== CROP YIELD PREDICTION =====")

# Take input from user
state = input("Enter state: ").strip().title()
crop = input("Enter crop: ").strip().title()
season = input("Enter season: ").strip().title()

year = int(input("Enter year: "))
area = float(input("Enter area: "))
rainfall = float(input("Enter annual rainfall: "))
fertilizer = float(input("Enter fertilizer amount: "))
pesticide = float(input("Enter pesticide amount: "))

# Create input DataFrame
input_data = pd.DataFrame({
    "Year": [year],
    "State": [state],
    "Crop": [crop],
    "Season": [season],
    "Area": [area],
    "Annual_Rainfall": [rainfall],
    "Fertilizer": [fertilizer],
    "Pesticide": [pesticide]
})

# Make prediction
prediction = model.predict(input_data)

print("\n===== RESULT =====")
print("State:", state)
print("Crop:", crop)
print("Season:", season)
print("Predicted Yield:", round(prediction[0], 2))