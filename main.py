# File: main.py

import pyaudio
import time
import json
import threading
import numpy as np
from collections import deque
import datetime
from faster_whisper import WhisperModel
import torch
from deepmultilingualpunctuation import PunctuationModel
from crm_connector import get_customer_profile

# --- Configuration ---
CONFIG = {
    "CHUNK": 1024,
    "FORMAT": pyaudio.paInt16,
    "CHANNELS": 1,
    "RATE": 16000,
    "WHISPER_MODEL": "small",
    "SILENCE_THRESHOLD": 350,
    "SILENCE_DURATION": 3.5,
    "COMMUNICATION_FILE": "communication.jsonl"
}

AGENT_CONFIG = {"CONTEXT_WINDOW": 3}

# --- Global Models ---
punctuation_model = PunctuationModel()

# --- Whisper Model Setup ---
def setup_whisper_model():
    device, compute_type = ("cuda", "float16") if torch.cuda.is_available() else ("cpu", "int8")
    whisper_model = WhisperModel(CONFIG["WHISPER_MODEL"], device=device, compute_type=compute_type)
    return whisper_model

# --- Sentence Analysis ---
def analyze_sentence(sentence, context_sentences):
    """Dummy CRIS reasoning for demo purposes"""
    intent = "General Inquiry"
    sentiment = "Neutral"
    reasoning = "Analyzed based on sentence content."
    if any(word in sentence.lower() for word in ["good","great","love"]):
        sentiment = "Positive"
    if any(word in sentence.lower() for word in ["bad","hate","problem"]):
        sentiment = "Negative"
    return {"Intent": intent, "Sentiment": sentiment, "Reasoning": reasoning}

# --- Milestone 3: CRM-based Recommendations ---
def recommend_products(customer_profile, context_sentences):
    recommended = []
    if "Interests" in customer_profile:
        recommended.extend(customer_profile["Interests"])
    keywords_to_products = {
        "upgrade":"Premium Package",
        "budget":"Budget-Friendly Package",
        "feature":"Feature Add-on"
    }
    for sentence in context_sentences:
        for k, p in keywords_to_products.items():
            if k in sentence.lower() and p not in recommended:
                recommended.append(p)
    return recommended[:5]

def generate_prompts(context_sentences):
    prompts=[]
    last_sentence = context_sentences[-1] if context_sentences else ""
    if "price" in last_sentence.lower():
        prompts.append("Highlight long-term ROI and savings.")
    if "feature" in last_sentence.lower():
        prompts.append("Offer a demo for this feature.")
    if "concern" in last_sentence.lower():
        prompts.append("Ask for more details on their concern to clarify.")
    return prompts[:3]

def analyze_and_write_to_file(phrase, context_window, customer_id="C001"):
    """Analyze phrase, include recommendations/prompts, write to communication file"""
    if len(phrase.split()) < 2:
        return

    analysis = analyze_sentence(phrase, list(context_window))
    customer_profile = get_customer_profile(customer_id)

    recommendations = recommend_products(customer_profile, list(context_window)) if customer_profile else []
    prompts = generate_prompts(list(context_window))

    result_row = {
        "Time": datetime.datetime.now().strftime("%H:%M:%S"),
        "Sentence": phrase,
        **analysis,
        "Recommendations": recommendations,
        "Prompts": prompts
    }

    try:
        with open(CONFIG["COMMUNICATION_FILE"], "a") as f:
            f.write(json.dumps(result_row) + "\n")
    except Exception as e:
        print(f"[ERROR writing to communication file: {e}]")

    context_window.append(phrase)

# --- Live Transcription & Analysis ---
def live_transcribe_and_analyze(model):
    p = pyaudio.PyAudio()
    stream = p.open(
        format=CONFIG["FORMAT"],
        channels=CONFIG["CHANNELS"],
        rate=CONFIG["RATE"],
        input=True,
        frames_per_buffer=CONFIG["CHUNK"]
    )

    print("\nListening for speech. Press Ctrl+C to stop.\n")
    audio_buffer = []
    context_window = deque(maxlen=AGENT_CONFIG["CONTEXT_WINDOW"])
    is_speaking = False
    silence_start_time = None

    try:
        while True:
            data = stream.read(CONFIG["CHUNK"], exception_on_overflow=False)
            rms = np.sqrt(np.mean(np.frombuffer(data, dtype=np.int16).astype(float)**2))

            if rms > CONFIG["SILENCE_THRESHOLD"]:
                is_speaking = True
                silence_start_time = None
                audio_buffer.append(data)
            elif is_speaking:
                if silence_start_time is None:
                    silence_start_time = time.time()
                if time.time() - silence_start_time > CONFIG["SILENCE_DURATION"]:
                    # Process audio
                    audio_np = np.frombuffer(b''.join(audio_buffer), dtype=np.int16).astype(np.float32)/32768.0
                    segments, info = model.transcribe(audio_np, beam_size=5)
                    phrase = "".join(s.text for s in segments).strip()
                    if phrase and info.language == 'en':
                        punctuated = punctuation_model.restore_punctuation(phrase)
                        if punctuated:
                            analyze_and_write_to_file(punctuated, context_window)
                    # Reset buffer
                    audio_buffer = []
                    is_speaking = False
                    silence_start_time = None

    except KeyboardInterrupt:
        print("\nBackend stopped by user.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Audio stream closed. Program finished.")

# --- Main ---
if __name__ == "__main__":
    whisper_model = setup_whisper_model()
    # Clear previous communication file
    open(CONFIG["COMMUNICATION_FILE"], 'w').close()
    live_transcribe_and_analyze(whisper_model)
