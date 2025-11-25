from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key from .env
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

client = OpenAI(api_key=api_key)

# Path to your audio file (using compressed version)
audio_file_path = "audio_compressed.mp3"

print("Starting transcription... this may take a few minutes depending on audio length")
print(f"Transcribing: {audio_file_path}")

try:
    with open(audio_file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )

    # Save the transcript
    output_file = "transcript.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(transcript)

    print(f"\n✓ Done! Transcript saved to {output_file}")
    print(f"Transcript length: {len(transcript)} characters")

except FileNotFoundError:
    print(f"Error: Audio file '{audio_file_path}' not found")
except Exception as e:
    print(f"Error during transcription: {str(e)}")
