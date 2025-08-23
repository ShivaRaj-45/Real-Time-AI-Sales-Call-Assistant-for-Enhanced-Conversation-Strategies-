import speech_recognition as sr

# Initialize recognizer
recognizer = sr.Recognizer()

# Use microphone as source
with sr.Microphone() as source:
    print("Speak now: ")
    # Listen to audio from the microphone
    audio = recognizer.listen(source)

# Try to recognize the speech using Google Speech Recognition
try:
    text = recognizer.recognize_google(audio)
    print("You said:", text)
except sr.UnknownValueError:
    print("Sorry, could not understand audio.")
except sr.RequestError as e:
    print("Could not request results; check your internet connection.")