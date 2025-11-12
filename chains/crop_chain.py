import requests, json, re, os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def get_location_coords(location):
    print(f"[DEBUG] Fetching coordinates for: {location}")
    try:
        r = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1")
        data = r.json()
        if "results" in data and len(data["results"]) > 0:
            lat = data["results"][0]["latitude"]
            lon = data["results"][0]["longitude"]
            print(f"[DEBUG] Found coordinates: ({lat}, {lon})")
            return lat, lon
    except Exception as e:
        print("[ERROR] Location fetch failed:", e)
    print("[DEBUG] Using default coordinates (Delhi)")
    return 28.61, 77.23

def fetch_weather(lat, lon):
    print(f"[DEBUG] Fetching live weather for ({lat}, {lon})")
    try:
        r = requests.get(
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&"
            f"current=temperature_2m,relative_humidity_2m,soil_moisture_0_to_10cm"
        )
        current = r.json().get("current", {})
        result = {
            "temperature": current.get("temperature_2m", 30),
            "humidity": current.get("relative_humidity_2m", 60),
            "moisture": round(current.get("soil_moisture_0_to_10cm", 0.25) * 100, 1)
        }
        print(f"[DEBUG] Weather data: {result}")
        return result
    except Exception as e:
        print("[ERROR] Weather fetch failed:", e)
        raise RuntimeError("Weather fetch failed — stopping.")

def fetch_soil(lat, lon):
    print(f"[DEBUG] Fetching soil data for ({lat}, {lon})")
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&"
        f"current=soil_temperature_0cm,soil_moisture_0_to_10cm"
    )
    print("[DEBUG] Request URL:", url)
    r = requests.get(url, timeout=10)
    print("[DEBUG] HTTP Status:", r.status_code)

    if r.status_code != 200:
        raise RuntimeError(f"HTTP Error {r.status_code}: {r.text[:200]}")

    data = r.json().get("current")
    if not data:
        raise ValueError("No soil data in response.")

    result = {
        "soil_temperature": data["soil_temperature_0cm"],
        "soil_moisture": round(data["soil_moisture_0_to_10cm"] * 100, 2)
    }
    print(f"[DEBUG] Soil data: {result}")
    return result

def recommend_crop(data: dict):
    """
    Input: {"location": "Jaipur", "season": "Kharif"}
    Output: {"crops": ["Rice", "Maize", "Pulses"], "explanation": "..."}
    """
    print("[DEBUG] Starting crop recommendation via Groq AI...\n")
    location = data.get("location", "Delhi")
    season = data.get("season", "Kharif")

    lat, lon = get_location_coords(location)
    weather = fetch_weather(lat, lon)
    soil = fetch_soil(lat, lon)

    system = (
    "You are KrishiMitra AI — a professional Indian agriculture expert and crop advisor. "
    "Your goal is to recommend the 3 most suitable crops for the given region and season "
    "based on live soil and weather data. "
    "Always include at least one major (staple) crop and one minor (rotation/cash) crop. "
    "Explain in Hinglish (mix of Hindi + English) within 2–3 lines — "
    "clearly describe why these crops suit the soil, climate, and season, "
    "and give one short tip on crop rotation or income stability. "
    "Avoid generic or repetitive suggestions and base reasoning only on provided parameters."
)

    user = f"""
📍 Location: {location}
🗓️ Season: {season}
🧭 Coordinates: {lat}, {lon}
🌡️ Temperature: {weather['temperature']} °C
💧 Humidity: {weather['humidity']} %
🌱 Soil Moisture (0–10 cm): {soil['soil_moisture']} %
🌡️ Soil Temperature (0 cm): {soil['soil_temperature']} °C

👉 Based on this data, suggest 3 best-fit crops (both major and minor).

Return response strictly in JSON:
{{
  "crops": [
    {{
      "name": "<crop1>",
      "type": "<Major or Minor>",
      "reason": "<1-line soil + weather suitability>",
      "rotation_tip": "<short rotation or income tip>"
    }},
    {{
      "name": "<crop2>",
      "type": "<Major or Minor>",
      "reason": "<1-line soil + weather suitability>",
      "rotation_tip": "<short rotation or income tip>"
    }},
    {{
      "name": "<crop3>",
      "type": "<Major or Minor>",
      "reason": "<1-line soil + weather suitability>",
      "rotation_tip": "<short rotation or income tip>"
    }}
  ],
  "summary": "<2–3 line Hinglish explanation combining all insights>"
}}
"""


    print("[DEBUG] Sending prompt to Groq...")
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0.6
        )
        text = response.choices[0].message.content
        print("[DEBUG] Groq response received:\n", text)
    except Exception as e:
        print("[ERROR] Groq API call failed:", e)
        text = ""

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            print("[DEBUG] Extracted JSON from response")
            return json.loads(match.group(0))
        except Exception as e:
            print("[ERROR] JSON parse failed:", e)
            raise RuntimeError("Groq AI returned invalid JSON.") from e

    raise RuntimeError("No valid JSON response from Groq.")

if __name__ == "__main__":
    print("Running crop_chain.py debug mode...\n")
    result = recommend_crop({"location": "Jaipur", "season": "Kharif"})
    print("\nFinal Output:")
    print(json.dumps(result, indent=2))
