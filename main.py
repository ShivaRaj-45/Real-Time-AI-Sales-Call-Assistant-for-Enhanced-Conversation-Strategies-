# File: backend.py

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

# --- Configuration ---
CONFIG = {
    "CHUNK": 1024,
    "FORMAT": pyaudio.paInt16,
    "CHANNELS": 1,
    "RATE": 16000,
    # ⭐ KEY CHANGE 1: Upgraded model for maximum accuracy
    "WHISPER_MODEL": "medium",
    "SILENCE_THRESHOLD": 350,
    "SILENCE_DURATION": 2, # Reduced pause time for faster response
    "GROQ_API_KEY": "YOUR GROQ API KEY", # IMPORTANT: PASTE YOUR KEY HERE
    "COMMUNICATION_FILE": "communication.jsonl"
}

AGENT_CONFIG = {
    "LLM_MODEL": "llama-3.1-8b-instant",
    "CONTEXT_WINDOW": 3,
}

# --- Global Models & Clients ---
punctuation_model = None
try:
    groq_client = Groq(api_key=CONFIG["GROQ_API_KEY"])
except Exception as e:
    groq_client = None
    print(f"FATAL: Could not initialize Groq client. Check API key. Error: {e}")

def setup_models():
    """Loads all necessary AI models."""
    global punctuation_model
    try:
        print("Loading punctuation restoration model...")
        punctuation_model = PunctuationModel()
        print("✅ Punctuation model loaded successfully.")

        print(f"Loading faster-whisper model ('{CONFIG['WHISPER_MODEL']}')...")

        if torch.cuda.is_available():
            device = "cuda"
            # ⭐ KEY CHANGE: Using int8 for maximum memory savings
            compute_type = "int8"
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ NVIDIA GPU detected: {gpu_name}. Using CUDA with int8 precision for lower VRAM usage.")
        else:
            device = "cpu"
            compute_type = "int8"
            print("INFO: No NVIDIA GPU detected. Using CPU, which will be slower.")
        
        whisper_model = WhisperModel(CONFIG["WHISPER_MODEL"], device=device, compute_type=compute_type)
        print("✅ Faster-Whisper model loaded successfully.")
        return whisper_model
        
    except Exception as e:
        print(f"FATAL: Error loading models: {e}")
        return None
    

def CRIS_analyzer_agent(sentence, context_sentences):
    """Analyzes a sentence using the CRIS framework via Groq API."""
    if not groq_client:
        return {"Intent": "Error", "Sarcasm": False, "Sentiment": "Neutral", "Reasoning": "Groq client not initialized."}

    context = " ".join(context_sentences)
    prompt = f"""
    You are a master analyst AI for sales calls, using the CRIS framework. Your task is to analyze the 'Current Sentence'.
    **Primary Directive: Your analysis MUST be based on the 'Current Sentence'. Use the 'Previous Context' ONLY to understand nuance (like sarcasm), NOT to influence the sentiment of a factually neutral sentence.**
    Previous Context: "{context}"
    Current Sentence: "{sentence}"
    Perform the following reasoning steps:
    1.  **Intent Classification:** What is the user's primary intent? (e.g., 'Information Seeking', 'Expressing Frustration', 'Objecting').
    2.  **Sarcasm Check:** Is sarcasm present?
    3.  **Sentiment Analysis:** Based ONLY on the Current Sentence and your sarcasm check, what is the final sentiment?
    Your desired output is a single, raw JSON object with four keys: "intent", "sarcasm_detected" (boolean), "sentiment" ('Positive', 'Negative', or 'Neutral'), and "reasoning" (brief explanation).
    """
    try:
        response = groq_client.chat.completions.create(
            model=AGENT_CONFIG["LLM_MODEL"],
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        result = json.loads(response.choices[0].message.content)
        return {
            "Intent": result.get("intent", "Unknown"),
            "Sarcasm": result.get("sarcasm_detected", False),
            "Sentiment": result.get("sentiment", "Neutral"),
            "Reasoning": result.get("reasoning", "N/A"),
        }
    except Exception as e:
        print(f"[Error in CRIS Agent: {e}]")
        return {"Intent": "Error", "Sarcasm": False, "Sentiment": "Neutral", "Reasoning": str(e)}

def analyze_and_write_to_file(sentence, context_window):
    """Analyzes a sentence and writes the JSON result to the communication file."""
    if len(sentence.split()) < 2: return

    print(f"\n[Analyzing]> {sentence}")
    analysis = CRIS_analyzer_agent(sentence, list(context_window))
    
    result_row = {
        "Time": time.strftime("%H:%M:%S"),
        "Sentence": sentence,
        **analysis
    }
    
    print(f"[Result]> {result_row}")

    try:
        with open(CONFIG["COMMUNICATION_FILE"], "a") as f:
            f.write(json.dumps(result_row) + "\n")
    except Exception as e:
        print(f"\n[ERROR writing to communication file: {e}]")
    
    context_window.append(sentence)

def live_transcribe_and_analyze(model):
    """Manages the real-time audio capture, transcription, and analysis loop."""
    p = pyaudio.PyAudio()
    stream = p.open(format=CONFIG["FORMAT"], channels=CONFIG["CHANNELS"],
                    rate=CONFIG["RATE"], input=True,
                    frames_per_buffer=CONFIG["CHUNK"])
    
    print("\n" + "="*50)
    print("✅ Backend is running. Listening for speech.")
    print("   Press Ctrl+C in this terminal to stop.")
    print("="*50 + "\n")
    
    audio_buffer = []
    context_window = deque(maxlen=AGENT_CONFIG["CONTEXT_WINDOW"])
    silence_start_time = None
    is_speaking = False

    try:
        while True:
            data = stream.read(CONFIG["CHUNK"], exception_on_overflow=False)
            audio_chunk = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_chunk.astype(float)**2))

            if rms > CONFIG["SILENCE_THRESHOLD"]:
                if not is_speaking: is_speaking = True
                silence_start_time = None
                audio_buffer.append(data)
            elif is_speaking:
                if silence_start_time is None: silence_start_time = time.time()
                
                if time.time() - silence_start_time > CONFIG["SILENCE_DURATION"]:
                    full_audio_data = b''.join(audio_buffer)
                    audio_np = np.frombuffer(full_audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    segments, info = model.transcribe(audio_np, beam_size=5)
                    transcribed_phrase = "".join(segment.text for segment in segments).strip()
                    
                    if transcribed_phrase and info.language == 'en':
                        punctuated_text = punctuation_model.restore_punctuation(transcribed_phrase)
                        sentences = re.split(r'(?<=[.?!])\s+', punctuated_text)
                        
                        for sentence in sentences:
                            if sentence:
                                threading.Thread(
                                    target=analyze_and_write_to_file,
                                    args=(sentence.strip(), context_window)
                                ).start()
                    
                    audio_buffer, is_speaking, silence_start_time = [], False, None

    except KeyboardInterrupt:
        print("\n\n" + "="*50)
        print("Backend stopped by user.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Audio stream closed. Program finished.")

if __name__ == "__main__":
    if CONFIG["GROQ_API_KEY"] == "YOUR_GROQ_API_KEY_HERE":
        print("FATAL: Please replace 'YOUR_GROQ_API_KEY_HERE' in backend.py with your actual Groq API key.")
    else:
        whisper_model = setup_models()
        if whisper_model:
            open(CONFIG["COMMUNICATION_FILE"], 'w').close()

            live_transcribe_and_analyze(whisper_model)
