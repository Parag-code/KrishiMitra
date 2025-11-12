import os, json, torch, re
from PIL import Image
from dotenv import load_dotenv
from groq import Groq
from transformers import AutoImageProcessor, AutoModelForImageClassification

print("Loading plant disease detection model...")
VISION_MODEL = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
vision_processor = AutoImageProcessor.from_pretrained(VISION_MODEL)
vision_model = AutoModelForImageClassification.from_pretrained(VISION_MODEL)
print("Vision model ready!")


load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
print("Groq client initialized!")


def detect_disease(image_path: str):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"[DEBUG] Processing image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    inputs = vision_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = vision_model(**inputs)
        probs = outputs.logits.softmax(dim=-1)
        idx = probs.argmax().item()

    disease_label = vision_model.config.id2label.get(idx, "Unknown Disease")
    confidence = round(float(probs[0][idx].item()), 2)
    print(f"Detected disease: {disease_label} (confidence: {confidence})")
    return disease_label, confidence


def generate_remedy_groq(disease_name: str, crop_hint: str = "General"):
    print(f"[DEBUG] Generating remedy for {disease_name} ({crop_hint})...")

    system = (
    "You are KrishiMitra AI — an Indian agriculture expert specialized in plant health. "
    "Generate a 3-line Hinglish remedy for the detected crop disease. "
    "Cover biological control, irrigation, and soil improvement. "
    "Avoid chemical brand names and use simple natural tone."
)

    user = f"""
Crop: {crop_hint}
Disease: {disease_name}

Return response strictly in JSON:
{{
  "remedy": [
    "Line 1: <pesticide or organic spray suggestion>",
    "Line 2: <irrigation or environmental tip>",
    "Line 3: <soil or compost improvement advice>"
  ],
  "summary": "<1-line simplified Hinglish explanation>",
  "severity": "<Low | Medium | High>",
  "natural_treatment": "<Neem, garlic spray, etc.>"
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
        text = response.choices[0].message.content
        print("[DEBUG] Groq response:\n", text)

        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group(0))
        return {"remedy": text.strip(), "summary": "Remedy suggestion generated."}

    except Exception as e:
        print("[ERROR] Groq API failed:", e)
        raise RuntimeError("Groq remedy generation failed.")


def analyze_leaf(image_path: str):
    disease, confidence = detect_disease(image_path)
    crop_hint = disease.split()[0] if " " in disease else "General"
    remedy_data = generate_remedy_groq(disease, crop_hint)

    result = {
        "disease": disease,
        "remedy": remedy_data.get("remedy", ""),
        "summary": remedy_data.get("summary", ""),
    }

    print("[FINAL RESULT]")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result

if __name__ == "__main__":
    sample = "samples/patato_leaf.jpg"
    if not os.path.exists(sample):
        print("Please add a sample leaf image under /samples/")
    else:
        print("Running KrishiMitra Vision AI...\n")
        analyze_leaf(sample)
