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
        print("⚠️ [ERROR] Location fetch failed:", e)
    print("⚠️ [DEBUG] Using default coordinates (Delhi)")
    return 28.61, 77.23


def fetch_weather(lat, lon):
    print(f"[DEBUG] Fetching live weather for ({lat}, {lon})")
    try:
        r = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,relative_humidity_2m,soil_moisture_0_to_10cm"
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
        return {"temperature": 30, "humidity": 60, "moisture": 25}


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
    if not data or "soil_moisture_0_to_10cm" not in data:
        raise ValueError("Essential soil data missing in Open-Meteo response.")

    result = {
        "soil_temperature": data["soil_temperature_0cm"],
        "soil_moisture": round(data["soil_moisture_0_to_10cm"] * 100, 2)  # convert to %
    }
    print(f"[DEBUG] Soil data: {result}")
    return result


def analyze_soil(data: dict):
    print("[DEBUG] Starting soil analysis via Groq AI...\n")
    crop = data.get("crop", "General")
    location = data.get("location", "Delhi")

    lat, lon = get_location_coords(location)
    weather = fetch_weather(lat, lon)
    soil = fetch_soil(lat, lon)

    system = (
    "You are KrishiMitra AI — Bharat ka ek expert agriculture advisor. "
    "Tumhara kaam hai farmer ko unke sheher, crop aur soil condition ke hisaab se "
    "best fertilizer recommend karna. "
    "Jawab Hinglish me do (simple Hindi + English mix), taaki har kisan samajh sake. "
    "Har answer thoda detailed ho — around 4 to 5 lines. "
    "Har recommendation me yeh teen cheezein honi chahiye:\n"
    "1️⃣ Kyon yeh fertilizer us crop aur soil ke liye best hai (scientific reason)\n"
    "2️⃣ Kitni matra aur kaise lagani chahiye (dose + method)\n"
    "3️⃣ Kya precautions aur soil-care tips follow karni chahiye\n"
    "Avoid chemical brand names aur overly technical language."
)

    user = f"""
Location: {location}
Crop: {crop}
Latitude: {lat}, Longitude: {lon}
Air Temperature: {weather['temperature']} °C
Humidity: {weather['humidity']} %
Soil Moisture (0–10 cm): {soil['soil_moisture']} %
Soil Temperature (0 cm): {soil['soil_temperature']} °C

Return JSON strictly in this format:
{{
  "fertilizer": "<best fertilizer name>",
  "dose_hint": "<recommended quantity and method>",
  "explanation": "<2-3 line detailed Hinglish explanation including why, how, and soil-care tips>"
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
            temperature=0.5
        )
        text = response.choices[0].message.content
        print("[DEBUG] Groq response received:\n", text)
    except Exception as e:
        print("Groq API Error:", e)
        text = ""

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            print("[DEBUG] Extracted JSON from response")
            return json.loads(match.group(0))
        except Exception as e:
            print("[ERROR] JSON parse failed:", e)

    raise RuntimeError("Groq AI returned invalid response.")

if __name__ == "__main__":
    print("Running soil_chain.py debug mode...\n")
    result = analyze_soil({"crop": "Wheat", "location": "Jaipur"})
    print("\nFinal Output:")
    print(json.dumps(result, indent=2))