import os
import torch
import sounddevice as sd
import time
import speech_recognition as sr
import openai
from dotenv import load_dotenv

# загрузка переменных из файла .env
load_dotenv("settings.env")

api_key = os.getenv('API_KEY')
openai.api_key = api_key

r = sr.Recognizer()
language = 'ru'
model_id = 'ru_v3'
sample_rate = int(os.getenv('SAMPLE_RATE')) # 48000
speaker = os.getenv('VOICE') # marusya, aidar, baya, kseniya, xenia, random
put_accent = True
put_yo = False

device = torch.device('cpu') # cpu или gpu


model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                        model='silero_tts',
                        language=language,
                        speaker=model_id)
model.to(device)

# задаем параметры записи аудио
mic = sr.Microphone(device_index=0)

def speak(txt: str):
    audio = model.apply_tts(text=txt+"..",
    speaker=speaker,
    sample_rate=sample_rate,
    put_accent=put_accent,
    put_yo=put_yo)

    sd.play(audio, sample_rate * 1.05)
    time.sleep((len(audio) / sample_rate) + 0.5)
    sd.stop()

speak("Привет, я Искуственный интеллект - ваш персональный помошник. Как я могу вам помочь?")

while True:
    with mic as source:
        try:
            r.adjust_for_ambient_noise(source)
            speak("Говорите сейчас...")
            audio = r.listen(source)
        except KeyboardInterrupt:
            speak("До встречи!")
            exit()
    # распознаем речь с помощью Google Speech Recognition
    try:
        text = r.recognize_google(audio, language="ru-RU")
    except sr.UnknownValueError:
        speak("Извините, но я не могу вас понять. Можете перефразировать ваш вопрос?")
    finally:
        print("Получены входные данные: %s" % text)

        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": text}
        ]
        )

        answer = response.choices[0].message["content"]
        print("Получены выходные данные")
        speak(answer)