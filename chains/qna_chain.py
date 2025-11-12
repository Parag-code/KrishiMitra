import os
from dotenv import load_dotenv
from groq import Groq


load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
print("Groq client initialized successfully!\n")


def krishimitra_answer(query: str):
    """
    Takes any farmer query and gives a Hinglish helpful answer.
    Powered by Groq API (Llama-3.3-70B).
    """
    system = (
    "You are KrishiMitra — Bharat ka digital kheti dost aur Indian agriculture expert. "
    "Tumhara kaam hai Indian farmers ke sawalon ka 4–6 line ka detailed, friendly aur practical Hinglish me jawab dena. "
    "Har answer me simple explanation ke saath khaad ki matra, beej daalne ka samay, paani dene ka tarika, "
    "aur ek useful kheti tip zarur ho. "
    "Agar sawaal fertilizer se related ho to approximate quantity bhi batao jaise "
    "‘DAP 50 kg per acre’, ‘Urea 100 kg per hectare’, ya ‘Neem spray 30 ml per litre paani’. "
    "Tone hamesha desi aur garamjoshi bhara rakho — jaise ek anubhav wala kisan apne bhai se baat kar raha ho. "
    "Avoid brand names, scientific shabd aur overly technical baatein. "
    "Har jawab me Bharat ki mitti, mausam aur kheti ke anubhav ka touch rakho "
    "taaki farmer ko lage ki KrishiMitra uske gaon ka asli madadgaar hai."
)

    user = f'Farmer asked: "{query}"'

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0.0,
            max_tokens=400,
            top_p=0.0
        )
        answer = response.choices[0].message.content.strip()
        return answer

    except Exception as e:
        print("[ERROR] Groq API failed:", e)
        return "Sorry, kuch technical dikkat ho gayi. Kripya fir se try karein."


if __name__ == "__main__":
    print("KrishiMitra AI Chatbot Ready (Groq Version)!\n")
    print("Type your farming question (or 'exit' to quit)\n")

    while True:
        query = input("Farmer: ").strip()
        if query.lower() in ["exit", "quit", "stop"]:
            print("Dhanyavaad! KrishiMitra aapke saath hamesha hai.")
            break

        response = krishimitra_answer(query)
        print(f"KrishiMitra: {response}\n")
