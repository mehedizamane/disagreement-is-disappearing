import torch
from pyannote.audio import Pipeline, Audio
import whisper
import pandas as pd
import librosa
import numpy as np
import os
import glob
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore")

# Reduce logging levels
logging.getLogger("pyannote.audio").setLevel(logging.ERROR)
logging.getLogger("speechbrain").setLevel(logging.ERROR)
logging.getLogger("libmpg123").setLevel(logging.ERROR)

# Suppress mpg123 warnings
os.environ["MPG123_NO_WARN"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"

# Ensure CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# Initialize PyAnnote pipeline and Whisper model
auth_token = "your_token_here"
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=auth_token
).to(device)

# Load Whisper model (base is small and fast)
model = whisper.load_model("base").to(device)

def diarize_and_transcribe_full_audio(audio_file):
    """Diarize and transcribe the entire audio optimized for memory and speed."""
    print(f"📢 Loading entire audio into memory: {audio_file}")
    
    # 1. Load the FULL audio into RAM once (Speed fix: prevents re-decoding file 100s of times)
    # We use 16000Hz because both Whisper and PyAnnote require this sample rate.
    full_audio_array, sr = librosa.load(audio_file, sr=16000)
    
    # 2. Perform Diarization
    # Convert numpy array to torch tensor for the pipeline
    audio_tensor = torch.from_numpy(full_audio_array).unsqueeze(0)
    print(f"🧠 Running diarization...")
    diarization = pipeline({"waveform": audio_tensor, "sample_rate": sr})

    # 3. Transcribe each speaker segment
    text_output = []
    print(f"✍️ Starting transcription of segments...")
    
    # Sort turns chronologically
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # Calculate indices for the array slice based on timestamps
        start_idx = int(turn.start * sr)
        end_idx = int(turn.end * sr)
        
        # Slice the audio array in memory (instant operation)
        segment = full_audio_array[start_idx:end_idx]
        
        if len(segment) == 0:
            continue

        try:
            # Transcribe the NumPy slice directly (No temp files created!)
            result = model.transcribe(segment.astype(np.float32))
            text_output.append(f"{speaker}: {result['text']}")
        except Exception as e:
            print(f"❌ Error transcribing segment at {turn.start}s: {e}")

    return '\n'.join(text_output)

# Paths
input_dir = "/scratch/sz841/disagreement_in_podcast/Datasets/The Glenn Beck Program/2021/podcast_episodes/"
output_dir = "/scratch/sz841/Diarize/glenn/diarize/"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# --- Filter to only 2021 files ---
audio_files = glob.glob(os.path.join(input_dir, "2021-*.mp3"))

for audio_file in audio_files:
    output_path = os.path.join(
        output_dir,
        f"{os.path.basename(audio_file)[:-4]}.txt"
    )
    
    # Skip if file already exists
    if os.path.exists(output_path):
        print(f"🔍 Skipping (already done): {audio_file}")
        continue

    print(f"🎙️ Processing: {audio_file}")
    try:
        transcription = diarize_and_transcribe_full_audio(audio_file)
        
        if not transcription.strip():
            print(f"⚠️ Empty transcription for {audio_file}, skipping.")
            continue
        
        # Save results to scratch
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(transcription)
            f.flush()
            os.fsync(f.fileno())
        
        print(f"✅ Saved: {output_path}")
    except Exception as e:
        print(f"❌ Failed {audio_file}: {e}")

print("🎉 All 2021 audio files have been processed.")
