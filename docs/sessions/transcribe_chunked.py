from openai import OpenAI
from dotenv import load_dotenv
import os
import subprocess
import math

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key from .env
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

client = OpenAI(api_key=api_key)

# Configuration
audio_file_path = "audio_compressed.mp3"
chunk_duration_minutes = 10  # Split into 10-minute chunks
output_file = "transcript.txt"

print(f"Processing: {audio_file_path}")

# Get audio duration using ffprobe
try:
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', audio_file_path],
        capture_output=True,
        text=True,
        check=True
    )
    total_duration = float(result.stdout.strip())
    total_minutes = total_duration / 60
    print(f"Total duration: {total_minutes:.1f} minutes")
except Exception as e:
    print(f"Error getting audio duration: {e}")
    exit(1)

# Calculate number of chunks
chunk_duration = chunk_duration_minutes * 60  # in seconds
num_chunks = math.ceil(total_duration / chunk_duration)
print(f"Splitting into {num_chunks} chunks of ~{chunk_duration_minutes} minutes each\n")

# Create temp directory for chunks
os.makedirs("temp_chunks", exist_ok=True)

all_transcripts = []

for i in range(num_chunks):
    start_time = i * chunk_duration
    chunk_file = f"temp_chunks/chunk_{i:03d}.mp3"

    print(f"[{i+1}/{num_chunks}] Processing chunk {i+1}...")

    # Extract chunk using ffmpeg
    print(f"  - Extracting audio chunk...")
    subprocess.run(
        ['ffmpeg', '-i', audio_file_path, '-ss', str(start_time),
         '-t', str(chunk_duration), '-c', 'copy', chunk_file, '-y'],
        capture_output=True,
        check=True
    )

    # Transcribe chunk
    print(f"  - Transcribing chunk...")
    try:
        with open(chunk_file, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        all_transcripts.append(transcript)
        print(f"  [OK] Chunk {i+1} completed ({len(transcript)} characters)")
    except Exception as e:
        print(f"  [ERROR] Error transcribing chunk {i+1}: {e}")
        all_transcripts.append(f"[Error transcribing chunk {i+1}]")

    # Clean up chunk file
    os.remove(chunk_file)

# Combine all transcripts
print(f"\nCombining {len(all_transcripts)} transcripts...")
final_transcript = "\n\n".join(all_transcripts)

# Save the complete transcript
with open(output_file, "w", encoding="utf-8") as f:
    f.write(final_transcript)

# Clean up temp directory
os.rmdir("temp_chunks")

print(f"\n[DONE] Complete transcript saved to {output_file}")
print(f"Total length: {len(final_transcript):,} characters")
