import pyaudio
import wave
import os
import time
from faster_whisper import WhisperModel
import torch
import numpy as np

CONFIG = {
    "CHUNK": 1024,
    "FORMAT": pyaudio.paInt16,
    "CHANNELS": 1,
    "RATE": 16000,
    "OUTPUT_DIR": "mock_sales_calls",
    "WHISPER_MODEL": "medium",
    "RECORD_SECONDS": 10, 
    "ENERGY_THRESHOLD": 300 
}

def setup_environment():
    """Sets up the project environment."""
    print("Setting up the environment...")
    if not os.path.exists(CONFIG["OUTPUT_DIR"]):
        os.makedirs(CONFIG["OUTPUT_DIR"])
        print(f" Created directory: {CONFIG['OUTPUT_DIR']}")
    else:
        print(f" Directory '{CONFIG['OUTPUT_DIR']}' already exists.")
    print("Environment setup complete!")


def live_transcribe(model):
    """
    Handles the entire live transcription process.
    - Continuously listens to the microphone in chunks.
    - Uses a simple energy-based silence detection.
    - Transcribes the audio captured so far and prints only new text.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=CONFIG["FORMAT"],
                    channels=CONFIG["CHANNELS"],
                    rate=CONFIG["RATE"],
                    input=True,
                    frames_per_buffer=CONFIG["CHUNK"])
    
    print("\n Live transcription started... Speak into your microphone.")
    print(" Press Ctrl+C in this terminal to stop the transcription.")
    
    audio_buffer = []
    last_transcribed_text = ""

    try:
        while True:
            frames = []
            for _ in range(0, int(CONFIG["RATE"] / CONFIG["CHUNK"] * CONFIG["RECORD_SECONDS"])):
                data = stream.read(CONFIG["CHUNK"])
                frames.append(data)

            audio_chunk = np.frombuffer(b''.join(frames), dtype=np.int16)

            rms = np.sqrt(np.mean(audio_chunk.astype(float)**2))
            if rms < CONFIG["ENERGY_THRESHOLD"]:
                continue

            if len(audio_buffer) > 0:
                audio_buffer = np.concatenate([audio_buffer, audio_chunk])
            else:
                audio_buffer = audio_chunk
            
            audio_for_whisper = audio_buffer.astype(np.float32) / 32768.0

            segments, _ = model.transcribe(audio_for_whisper, beam_size=5)
            
            full_transcript = "".join(segment.text for segment in segments).strip()

            if len(full_transcript) > len(last_transcribed_text):
                new_text = full_transcript[len(last_transcribed_text):]
                print(new_text, end='', flush=True)
                last_transcribed_text = full_transcript

    except KeyboardInterrupt:
        print("\n\n Live transcription stopped by user.")
    finally:
        print("\n   Closing audio stream...")
        stream.stop_stream()
        stream.close()
        p.terminate()

        if len(audio_buffer) > 0:
            print("   Performing final transcription on the full conversation...")
    
            final_audio = audio_buffer.astype(np.float32) / 32768.0
            segments, info = model.transcribe(final_audio, beam_size=5)
            final_text = "".join(segment.text for segment in segments).strip()

            print("\n" + "="*50)
            print("           FULL CONVERSATION TRANSCRIPT           ")
            print("="*50)
            print(final_text)
            print("="*50 + "\n")


def main():
    """Main function to load the model and start the live transcription."""
    setup_environment()
    
    model_size = CONFIG["WHISPER_MODEL"]

    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
        device_index = 0
        print(" NVIDIA GPU detected. The script will use GPU 0 for high-speed transcription.")
    else:
        device = "cpu"
        compute_type = "int8"
        device_index = 0
        print(" No NVIDIA GPU detected. Using CPU. This will be significantly slower.")

    print(f"\nLoading faster-whisper model ('{model_size}')... This may take a moment.")
    try:
        model = WhisperModel(
            model_size, 
            device=device, 
            compute_type=compute_type, 
            device_index=device_index
        )
        print(" faster-whisper model loaded successfully.")
    except Exception as e:
        print(f" Error loading faster-whisper model: {e}")
        return

    
    live_transcribe(model)
    print(" Program finished.")


if __name__ == "__main__":
    main()
