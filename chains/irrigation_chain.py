import requests, json, re,os
from geopy.geocoders import Nominatim
from dotenv import load_dotenv
from groq import Groq


load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
print("Groq client initialized!\n")


def get_latlon_from_city(city: str):
    try:
        geolocator = Nominatim(user_agent="krishimitra")
        loc = geolocator.geocode(city)
        if loc:
            print(f"City found: {city} ({loc.latitude}, {loc.longitude})")
            return loc.latitude, loc.longitude
    except Exception as e:
        print(f"[WARN] Location fetch failed: {e}")
    print("Using default Delhi coordinates.")
    return 28.6, 77.2


def fetch_weather(lat: float, lon: float):
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&daily=temperature_2m_max,"
        f"temperature_2m_min,precipitation_sum,relative_humidity_2m_max&forecast_days=7"
    )
    res = requests.get(url).json()
    daily = res["daily"]
    data = []

    print("\n7-Day Forecast:")
    print("-----------------------------------------------------------")

    for i in range(7):
        day_info = {
            "day": daily["time"][i],
            "rainfall": daily["precipitation_sum"][i],
            "temperature": (daily["temperature_2m_max"][i] + daily["temperature_2m_min"][i]) / 2,
            "humidity": daily["relative_humidity_2m_max"][i]
        }
        data.append(day_info)
        print(
            f"{day_info['day']} | Rain: {day_info['rainfall']} mm | "
            f"Temp: {day_info['temperature']:.1f}°C | Humidity: {day_info['humidity']}%"
        )

    print("-----------------------------------------------------------")

    total_rain = sum(d["rainfall"] for d in data)
    if total_rain == 0:
        print("☀️ Dry week detected — koi barish forecast nahi hai.\n")
    else:
        print(f"🌧️ Total weekly rainfall forecast: {total_rain:.1f} mm\n")

    return data




def predict_weather_trend(forecast):
    print("[DEBUG] Summarizing weather trend via Groq...")

    summary_text = " ".join(
        [f"{f['day']} rain {f['rainfall']}mm temp {f['temperature']}°C hum {f['humidity']}%" for f in forecast]
    )

    system = (
        "You are KrishiMitra AI — an Indian agriculture assistant. "
        "Summarize the upcoming 7-day weather pattern for farmers in 1-2 simple Hinglish lines. "
        "Mention if it will be dry, rainy, ya mixed weather."
    )
    user = f"Weather data: {summary_text}"

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0.4,
            max_tokens=150
        )
        text = response.choices[0].message.content.strip()
        print("[DEBUG] Weather trend summary received:\n", text)
        return text
    except Exception as e:
        print("[ERROR] Groq trend generation failed:", e)
        return "Lagbhag dry aur thoda mixed mausam rehne wala hai."


def generate_irrigation_advice(crop: str, soil: str, trend: str, city: str):
    print("[DEBUG] Generating irrigation advice via Groq...")

    system = (
    "You are KrishiMitra AI — an Indian agriculture and irrigation expert. "
    "Based on the given crop, soil type, and weather trend, generate 4–5 line Hinglish irrigation advice. "
    "Each answer should explain (1) kitna paani dena hai, (2) kab aur kitni baar dena hai, "
    "(3) mausam ke hisaab se kya badlav karein, aur (4) soil moisture bachane ke tips. "
    "Use simple, friendly Hinglish — jaise aap kisi kisan se baat kar rahe ho. "
    "Avoid brand names or technical chemical terms. "
    "Keep the tone positive, natural, and practical with easy daily-life suggestions."
)

    user = f"""
📍 City: {city}
🌾 Crop: {crop}
🌱 Soil Type: {soil}
🌦️ Weather Trend: {trend}

👉 Based on these inputs, give 4–5 line Hinglish irrigation advice for the farmer.

Return response strictly in JSON format:
{{
  "advice": "<4–5 line Hinglish irrigation tips>"
}}
"""


    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0.6,
            max_tokens=300
        )
        text = response.choices[0].message.content.strip()
        print("[DEBUG] Groq irrigation response:\n", text)

        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            data = json.loads(match.group(0).replace("'", '"'))
            if "advice" in data:
                data["advice"] = data["advice"].replace(". ", ".\n")
                return data["advice"]
        return text.replace(". ", ".\n")

    except Exception as e:
        print("[ERROR] Groq irrigation generation failed:", e)
        return "Subah jaldi irrigation karein aur heavy rain ke din paani band rakhein."


def analyze_irrigation(data: dict):
    city = data.get("city", "Delhi")
    crop = data.get("crop", "Wheat")
    soil = data.get("soil_type", "Loamy")

    lat, lon = get_latlon_from_city(city)
    forecast = fetch_weather(lat, lon)
    trend = predict_weather_trend(forecast)
    advice = generate_irrigation_advice(crop, soil, trend, city)

    return {
        "weather_trend": trend,
        "irrigation_advice": advice
    }


if __name__ == "__main__":
    user_input = {"city": "Kolkata", "crop": "Wheat", "soil_type": "Black"}
    result = analyze_irrigation(user_input)
    print(f"\nCity: {result['city']} ({result['lat']}, {result['lon']})")
    print("\nWeather Trend →", result["weather_trend"])
    print("\nIrrigation Advice →\n", result["irrigation_advice"])
