# elevenlabs_tts.py
# Integration with 11 Labs API for high-quality text-to-speech

import os
import requests
import json
import tempfile
from gtts import gTTS

# Default API key - will be loaded from secrets or environment
DEFAULT_API_KEY = None

def get_api_key():
    """Get 11 Labs API key from Streamlit secrets, environment, or return None"""
    try:
        import streamlit as st
        return st.secrets["elevenlabs"]["api_key"]
    except (ImportError, KeyError, FileNotFoundError):
        # Fall back to environment variable
        import os
        return os.environ.get("ELEVENLABS_API_KEY")
    except Exception:
        return None

# Available voices in 11 Labs - using voices available in your account
AVAILABLE_VOICES = {
    # Your voice clone
    "Your Voice Clone": "Sqahs9NqWlhYumWOfZRh",  # Indian | Classical English Story | Male | Middle Aged | Srikant Rajan

    # Standard voices available in your account
    "Sarah": "EXAVITQu4vr4xnSDxMaL",
    "Aria": "9BWtsMINqrJLrRacOk9x",
    "Laura": "FGY2WhTYpPnrIDTdsKH5",
    "Charlie": "IKne3meq5aSn9XLyUdCD",
    "George": "JBFqnCBsd6RMkjVDRZzb",
    "Callum": "N2lVS1w4EtoT3dr4eOWO",
    "River": "SAz9YHcvj6GT2YYXdXww",
    "Liam": "TX3LPaxmHKxFdv7VOQHJ",
    "Charlotte": "XB0fDUnXU5powFXDhCwa",
    "Alice": "Xb7hH8MSUJpSbSDYk0k2",
    "Matilda": "XrExE9yKIg1WjnnlVkGX"
}

def text_to_speech_11labs(text, output_path, api_key=None, voice_id=None):
    """
    Convert text to speech using 11 Labs API

    Args:
        text: Text to convert to speech
        output_path: Path to save the audio file
        api_key: 11 Labs API key (optional, will use default if not provided)
        voice_id: Voice ID to use (optional, will use default if not provided)

    Returns:
        True if successful, False otherwise
    """
    # Use provided API key or get from secrets/environment
    api_key = api_key or get_api_key()

    # Use provided voice ID or default to your voice clone
    voice_id = voice_id or AVAILABLE_VOICES.get("Your Voice Clone")

    # API endpoint
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    # Request headers
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }

    # Request body with improved settings for voice clones
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",  # Better model for voice clones
        "voice_settings": {
            "stability": 0.75,  # Higher stability for more consistent output
            "similarity_boost": 0.75  # Higher similarity to match your voice better
        }
    }

    try:
        # Make the request
        response = requests.post(url, json=data, headers=headers)

        # Check if request was successful
        if response.status_code == 200:
            # Save the audio file
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"Audio saved to {output_path}")
            return True
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"Error generating audio with 11 Labs: {e}")
        return False

def get_available_voices():
    """
    Get a list of available voices

    Returns:
        Dictionary of voice names and IDs
    """
    return AVAILABLE_VOICES

def text_to_speech_with_fallback(text, output_path, use_11labs=True, api_key=None, voice_id=None):
    """
    Convert text to speech using 11 Labs API with fallback to gTTS

    Args:
        text: Text to convert to speech
        output_path: Path to save the audio file
        use_11labs: Whether to use 11 Labs API (True) or gTTS (False)
        api_key: 11 Labs API key (optional, will use default if not provided)
        voice_id: Voice ID to use (optional, will use default if not provided)

    Returns:
        True if successful, False otherwise
    """
    # Use your voice clone
    your_api_key = get_api_key()
    your_voice_id = AVAILABLE_VOICES.get("Your Voice Clone")

    print(f"Using 11 Labs with voice ID: {your_voice_id} (Your Voice Clone)")

    # Always try 11 Labs first with your voice clone
    success = text_to_speech_11labs(text, output_path, your_api_key, your_voice_id)
    if success:
        print("Successfully generated audio with your 11 Labs voice clone")
        return True

    print("Falling back to gTTS...")

    # Fallback to gTTS
    try:
        tts = gTTS(text)
        tts.save(output_path)
        print(f"Audio saved to {output_path} using gTTS")
        return True
    except Exception as e:
        print(f"Error generating audio with gTTS: {e}")
        return False

# Test the function
if __name__ == "__main__":
    test_text = "Hello, this is a test of the 11 Labs text-to-speech API."
    test_output = os.path.join(tempfile.gettempdir(), "test_11labs.mp3")
    success = text_to_speech_with_fallback(test_text, test_output)
    print(f"Test {'successful' if success else 'failed'}")
