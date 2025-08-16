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

# list of prompts
prompts = [
    "say a joke",
    "write me a poem on space",
    "can you make an animal noise",
    "give me one random English word",
    "say a wise quote",
    "tell me a dad joke",
    "give me a fun fact about the sky"
]

# check for internet connection
def is_online():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=2)
        return True
    except:
        return False

# speak using gTTS 
def speak(text):
    print("Speaking:", text)
    tts = gTTS(text, lang="en")
    tts.save("say.mp3")
    os.system("mpg123 say.mp3")

# generate text with gemini and says it
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
        speak("There was an error trying to think")

# main loop
if is_online():
    while True:
        generate_and_speak()
        time.sleep(20)
else:
    print("No internet connection")
