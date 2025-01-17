import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("WEATHER_API_KEY")

CITY = "Tallinn"
BASE_URL = "https://api.openweathermap.org/data/2.5/onecall"

LAT = 59.437
LON = 24.753

# Fetch weather data
params = {
    "lat": LAT,
    "lon": LON,
    "exclude": "minutely,hourly",
    "appid": API_KEY,
    "units": "metric"  # Metric for Celsius
}

response = requests.get(BASE_URL, params=params)
data = response.json()
print(data)

# Inspect the daily weather data
daily_weather = data["daily"]

weather_data = []
for day in daily_weather:
    date = datetime.fromtimestamp(day["dt"]).strftime("%Y-%m-%d")
    temp = day["temp"]["day"]  # Day temperature
    rain = day.get("rain", 0)  # Rain (if available)
    weather_data.append({"Date": date, "Temperature (°C)": temp, "Rain (mm)": rain})

# Create a DataFrame
df = pd.DataFrame(weather_data)

# Display the DataFrame
print(df.head())
