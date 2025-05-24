# test_11labs.py
# A simple script to test the 11 Labs API with your voice ID

import os
import requests
import tempfile
import time

# Your 11 Labs credentials - now loaded from environment or secrets
def get_api_key():
    """Get 11 Labs API key from environment variable"""
    return os.environ.get("ELEVENLABS_API_KEY", "your-api-key-here")

API_KEY = get_api_key()
VOICE_ID = "Sqahs9NqWlhYumWOfZRh"

def test_11labs_api(text="This is a test of my voice clone using the 11 Labs API."):
    """
    Test the 11 Labs API with your voice ID

    Args:
        text: Text to convert to speech

    Returns:
        Path to the generated audio file if successful, None otherwise
    """
    print(f"Testing 11 Labs API with voice ID: {VOICE_ID}")
    print(f"API Key: {API_KEY[:5]}...{API_KEY[-5:]}")

    # API endpoint
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"

    # Request headers
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": API_KEY
    }

    # Request body
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }

    try:
        print("Sending request to 11 Labs API...")
        # Make the request
        response = requests.post(url, json=data, headers=headers)

        # Check if request was successful
        if response.status_code == 200:
            # Save the audio file
            output_path = os.path.join(tempfile.gettempdir(), f"test_11labs_{int(time.time())}.mp3")
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"Success! Audio saved to {output_path}")
            return output_path
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"Exception: {e}")
        return None

def list_available_voices():
    """
    List all available voices in your 11 Labs account
    """
    print("Listing available voices in your 11 Labs account...")

    # API endpoint
    url = "https://api.elevenlabs.io/v1/voices"

    # Request headers
    headers = {
        "Accept": "application/json",
        "xi-api-key": API_KEY
    }

    try:
        # Make the request
        response = requests.get(url, headers=headers)

        # Check if request was successful
        if response.status_code == 200:
            voices = response.json().get("voices", [])
            print(f"Found {len(voices)} voices:")
            for voice in voices:
                print(f"- {voice.get('name')}: {voice.get('voice_id')}")
            return voices
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"Exception: {e}")
        return None

if __name__ == "__main__":
    print("=== 11 Labs API Test ===")

    # List available voices
    print("\n=== Available Voices ===")
    voices = list_available_voices()

    # Check if your voice ID exists in the available voices
    if voices:
        voice_exists = any(voice.get('voice_id') == VOICE_ID for voice in voices)
        if voice_exists:
            print(f"\nYour voice ID ({VOICE_ID}) exists in your account.")
        else:
            print(f"\nWARNING: Your voice ID ({VOICE_ID}) was NOT found in your account!")

    # Test generating audio with your voice
    print("\n=== Testing Audio Generation ===")
    output_path = test_11labs_api()

    if output_path:
        print("\nTest completed successfully!")
        print(f"Audio file saved to: {output_path}")
        print("Please play this file to verify your voice clone is working correctly.")
    else:
        print("\nTest failed. Please check the error messages above.")
