# adrianwork
Robotics lab work
import sounddevice as sd
import numpy as np
import pyttsx3
from vosk import Model, KaldiRecognizer
from transformers import MarianMTModel, MarianTokenizer
import json
import warnings

# with this we suppress some specific warnings about the driver
warnings.filterwarnings("ignore", message="Can't initialize NVML")
warnings.filterwarnings("ignore", message="CUDA initialization: The NVIDIA driver on your system is too old")

# This part set up the Vosk model (ensure you have the model files downloaded)
vosk_model_path = 'vosk-model-small-ru-0.22'  # specify the path to your vosk model
model = Model(vosk_model_path)

#  HERE YOU Initialize the translation model
model_name = "Helsinki-NLP/opus-mt-ru-en"
translation_model = MarianMTModel.from_pretrained(model_name)
translation_tokenizer = MarianTokenizer.from_pretrained(model_name)

def recognize_speech_from_mic(samplerate=16000, duration=5):
    """Record audio from the microphone and transcribe it using Vosk."""
    print("Listening...")
    try:
        audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
        sd.wait()
        print("Recording complete.")
        
        #  here we play back the recorded audio for verification and check if it is correct
        sd.play(audio, samplerate)
        sd.wait()

        audio = np.squeeze(audio)
        audio_data = audio.tobytes()

        recognizer = KaldiRecognizer(model, samplerate)
        
        #  this is for processing audio data in chunks to handle longer audio
        recognizer.AcceptWaveform(audio_data)
        result = json.loads(recognizer.FinalResult())  # To convert JSON data string to  Python dictionary
        text = result.get('text', '')

        response = {
            "success": True,
            "error": None,
            "transcription": text
        }

    except Exception as e:
        response = {
            "success": False,
            "error": f"An error occurred during recording: {e}",
            "transcription": None
        }

    return response

def translate_text(text, src_lang="ru", tgt_lang="en"):
    """Translate text from source language to target language."""
    try:
        inputs = translation_tokenizer(text, return_tensors="pt", truncation=True)
        translated = translation_model.generate(**inputs)
        translation = translation_tokenizer.batch_decode(translated, skip_special_tokens=True)
        return translation[0]
    except Exception as e:
        return f"An error occurred during translation: {e}"

def speak_text(text):     # this speaks the translated text to have an audio response
    """Convert text to speech."""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def main():
    print("Say something!")
    speech_response = recognize_speech_from_mic()

    if speech_response["success"]:
        if speech_response["transcription"]:
            print(f"You said: {speech_response['transcription']}")
            translated_text = translate_text(speech_response["transcription"])
            print(f"Translated text: {translated_text}")
            speak_text(translated_text)  
        else:
            print("I didn't catch that. Please try again.")
    else:
        print(f"Error: {speech_response['error']}")

if __name__ == "__main__":
    main()
