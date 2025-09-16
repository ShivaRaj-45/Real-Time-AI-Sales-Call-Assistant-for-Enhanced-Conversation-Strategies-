# File: main.py

import pyaudio
import time
import json
import threading
import numpy as np
from collections import deque
import re
from faster_whisper import WhisperModel
import torch
from groq import Groq
from deepmultilingualpunctuation import PunctuationModel
import datetime

# --- Configuration ---
CONFIG = {
    "CHUNK": 1024, "FORMAT": pyaudio.paInt16, "CHANNELS": 1, "RATE": 16000,
    "WHISPER_MODEL": "small", "SILENCE_THRESHOLD": 350, "SILENCE_DURATION": 3.5,
    "GROQ_API_KEY": "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",
    "COMMUNICATION_FILE": "communication.jsonl"
}
AGENT_CONFIG = {"LLM_MODEL": "llama-3.1-8b-instant", "CONTEXT_WINDOW": 3}

# --- Global Models & Clients ---
punctuation_model = None
try:
    groq_client = Groq(api_key=CONFIG["GROQ_API_KEY"])
except Exception as e:
    groq_client = None
    print(f"FATAL: Could not initialize Groq client. Check API key. Error: {e}")

def setup_models():
    """Loads all AI models using the 'small' Whisper model on GPU if available."""
    global punctuation_model
    try:
        print("Loading punctuation restoration model...")
        punctuation_model = PunctuationModel()
        print(" Punctuation model loaded successfully.")
        chosen_model = CONFIG["WHISPER_MODEL"]
        if torch.cuda.is_available():
            device, compute_type = "cuda", "float16"
            gpu_name = torch.cuda.get_device_name(0)
            print(f" NVIDIA GPU detected: {gpu_name}.")
            print(f"   -> Using model: '{chosen_model}' with '{compute_type}' precision for maximum speed.")
        else:
            device, compute_type = "cpu", "int8"
            print(f"INFO: No NVIDIA GPU detected. Using CPU with model '{chosen_model}'.")
        whisper_model = WhisperModel(chosen_model, device=device, compute_type=compute_type)
        print(" Faster-Whisper model loaded successfully.")
        return whisper_model
    except Exception as e:
        print(f"FATAL: Error loading models: {e}")
        return None

def analyze_sentence(sentence, context_sentences):
    """Analyzes a sentence using the CRIS framework via Groq API."""
    if not groq_client: return {"Intent": "Error", "Sentiment": "Neutral", "Reasoning": "Groq client not initialized."}
    context = " ".join(context_sentences)
    # --- KEY CHANGE: More specific prompt for consistent reasoning ---
    prompt = f"""
    Analyze the 'Current Sentence' based on the 'Previous Context'.
    Previous Context: "{context}"
    Current Sentence: "{sentence}"

    Your desired output is a single, raw JSON object with three keys: "intent", "sentiment", and "reasoning".
    - "intent": Classify the user's primary goal.
    - "sentiment": Label the emotion as 'Positive', 'Negative', or 'Neutral'.
    - "reasoning": Provide a concise, objective, third-person explanation for your classification based only on the evidence in the Current Sentence.
    """
    try:
        response = groq_client.chat.completions.create(model=AGENT_CONFIG["LLM_MODEL"], messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"}, temperature=0.0)
        result = json.loads(response.choices[0].message.content)
        return {
            "Intent": result.get("intent", "Unknown"),
            "Sentiment": result.get("sentiment", "Neutral"),
            "Reasoning": result.get("reasoning", "N/A"),
        }
    except Exception as e:
        print(f"[Error in CRIS Agent: {e}]")
        return {"Intent": "Error", "Sentiment": "Neutral", "Reasoning": str(e)}

def analyze_and_write_to_file(phrase, context_window):
    """Analyzes an entire phrase and writes the JSON result to the communication file."""
    if len(phrase.split()) < 2: return
    analysis = analyze_sentence(phrase, list(context_window))
    result_row = {"Time": datetime.datetime.now().strftime("%H:%M:%S"), "Sentence": phrase, **analysis}
    try:
        with open(CONFIG["COMMUNICATION_FILE"], "a") as f: f.write(json.dumps(result_row) + "\n")
    except Exception as e:
        print(f"\n[ERROR writing to communication file: {e}]")
    context_window.append(phrase)

def live_transcribe_and_analyze(model):
    """Manages the real-time audio capture, transcription, and analysis loop."""
    p = pyaudio.PyAudio()
    stream = p.open(format=CONFIG["FORMAT"], channels=CONFIG["CHANNELS"], rate=CONFIG["RATE"], input=True, frames_per_buffer=CONFIG["CHUNK"])
    print("\n" + "="*50 + "\n Backend is running. Listening for speech.\n   Press Ctrl+C to stop.\n" + "="*50 + "\n")
    audio_buffer, context_window, is_speaking, silence_start_time = [], deque(maxlen=AGENT_CONFIG["CONTEXT_WINDOW"]), False, None
    try:
        while True:
            data = stream.read(CONFIG["CHUNK"], exception_on_overflow=False)
            rms = np.sqrt(np.mean(np.frombuffer(data, dtype=np.int16).astype(float)**2))
            if rms > CONFIG["SILENCE_THRESHOLD"]:
                is_speaking, silence_start_time = True, None
                audio_buffer.append(data)
            elif is_speaking:
                if silence_start_time is None: silence_start_time = time.time()
                if time.time() - silence_start_time > CONFIG["SILENCE_DURATION"]:
                    audio_np = np.frombuffer(b''.join(audio_buffer), dtype=np.int16).astype(np.float32) / 32768.0
                    segments, info = model.transcribe(audio_np, beam_size=5)
                    phrase = "".join(s.text for s in segments).strip()
                    if phrase and info.language == 'en':
                        punctuated_phrase = punctuation_model.restore_punctuation(phrase)
                        if punctuated_phrase:
                            analyze_and_write_to_file(punctuated_phrase, context_window)
                    audio_buffer, is_speaking, silence_start_time = [], False, None
    except KeyboardInterrupt:
        print("\n\n" + "="*50 + "\nBackend stopped by user.")
    finally:
        stream.stop_stream(); stream.close(); p.terminate()
        print("Audio stream closed. Program finished.")

if __name__ == "__main__":
    whisper_model = setup_models()
    if whisper_model:
        open(CONFIG["COMMUNICATION_FILE"], 'w').close()
        live_transcribe_and_analyze(whisper_model)
