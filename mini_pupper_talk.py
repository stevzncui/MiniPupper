import google.generativeai as genai
from gtts import gTTS
import os
import time
import random
import socket

# put your api key in here
genai.configure(api_key="")

# put a model you have access to in here
model = genai.GenerativeModel("")

# List of prompts
prompts = [
    "say a joke",
    "write me a poem on space",
    "can you make an animal noise",
    "give me one random English word",
    "say a wise quote",
    "tell me a dad joke",
    "give me a fun fact about the sky"
]

# Check for internet connection
def is_online():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=2)
        return True
    except:
        return False

# Speak using gTTS or fallback to espeak
def speak(text):
    print("Speaking:", text)
    if is_online():
        try:
            tts = gTTS(text, lang="en")
            tts.save("say.mp3")
            os.system("mpg123 say.mp3")
        except:
            print("gTTS failed. Using espeak fallback.")
            os.system(f'espeak "{text}"')
    else:
        print("No internet. Using offline voice.")
        os.system(f'espeak "{text}"')

# Generate text with Gemini and speak it
def generate_and_speak():
    prompt = random.choice(prompts)
    print("Prompt:", prompt)
    try:
        response = model.generate_content(prompt)
        message = response.text.strip().replace("\n", " ")
        print("Gemini response:", message)
        speak(message)
    except Exception as e:
        print("Error:", e)
        speak("There was an error trying to think.")

# Main loop
while True:
    generate_and_speak()
    time.sleep(20)

