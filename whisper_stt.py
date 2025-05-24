# whisper_stt.py
# Integration with OpenAI's Whisper model for high-quality speech-to-text

import os
import tempfile
import whisper
import librosa
import soundfile as sf
import time

# Default model size - can be 'tiny', 'base', 'small', 'medium', or 'large'
DEFAULT_MODEL_SIZE = "base"

# Cache for loaded models to avoid reloading
_model_cache = {}

def load_whisper_model(model_size=DEFAULT_MODEL_SIZE):
    """
    Load a Whisper model with the specified size

    Args:
        model_size: Size of the model to load ('tiny', 'base', 'small', 'medium', 'large')

    Returns:
        Loaded Whisper model
    """
    if model_size in _model_cache:
        return _model_cache[model_size]

    try:
        print(f"Loading Whisper {model_size} model...")
        start_time = time.time()
        model = whisper.load_model(model_size)
        load_time = time.time() - start_time
        print(f"Whisper {model_size} model loaded in {load_time:.2f} seconds")

        _model_cache[model_size] = model
        return model
    except Exception as e:
        error_message = f"Error loading Whisper model: {e}"
        print(error_message)
        raise Exception(error_message)

def transcribe_with_whisper(audio_path, model_size=DEFAULT_MODEL_SIZE, language="en"):
    """
    Transcribe audio using Whisper

    Args:
        audio_path: Path to the audio file
        model_size: Size of the model to use ('tiny', 'base', 'small', 'medium', 'large')
        language: Language code (e.g., 'en' for English)

    Returns:
        Transcribed text
    """
    try:
        # Load audio file and convert to proper format if needed
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)  # Using _ for unused sample rate

        # Save as WAV in the right format for Whisper
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            sf.write(tmp_wav.name, audio, 16000)
            temp_audio_path = tmp_wav.name

        # Load the model
        model = load_whisper_model(model_size)

        # Transcribe the audio
        print(f"Transcribing audio with Whisper {model_size} model...")
        print(f"Audio file path: {temp_audio_path}")
        print(f"File exists: {os.path.exists(temp_audio_path)}")
        print(f"File size: {os.path.getsize(temp_audio_path) if os.path.exists(temp_audio_path) else 'N/A'} bytes")

        start_time = time.time()
        try:
            # Try with fp16=False to avoid CUDA errors
            result = model.transcribe(temp_audio_path, language=language, fp16=False)
            transcribe_time = time.time() - start_time
            print(f"Transcription completed in {transcribe_time:.2f} seconds")
        except Exception as e:
            print(f"Error during transcription: {e}")
            # Try alternative approach with direct audio loading
            print("Attempting alternative transcription method...")
            try:
                # Load audio directly using whisper's functions
                audio_array = whisper.load_audio(temp_audio_path)
                audio_array = whisper.pad_or_trim(audio_array)

                # Log audio array properties
                print(f"Audio array shape: {audio_array.shape}")
                print(f"Audio array min/max: {audio_array.min()}/{audio_array.max()}")

                # Process with whisper
                mel = whisper.log_mel_spectrogram(audio_array).to(model.device)
                options = whisper.DecodingOptions(language=language, fp16=False)
                result = whisper.decode(model, mel, options)

                # Convert result to match standard transcribe output format
                result = {"text": result.text}

                transcribe_time = time.time() - start_time
                print(f"Alternative transcription completed in {transcribe_time:.2f} seconds")
            except Exception as nested_e:
                print(f"Alternative transcription also failed: {nested_e}")
                raise

        # Clean up temp file
        os.remove(temp_audio_path)

        # Return the transcribed text
        return result["text"].strip()

    except Exception as e:
        error_message = f"Error transcribing with Whisper: {e}"
        print(error_message)
        # Return the error message instead of empty string to help with debugging
        return f"TRANSCRIPTION_ERROR: {error_message}"

def transcribe_audio(audio_path, model_size=DEFAULT_MODEL_SIZE, language="en"):
    """
    Main function to transcribe audio using Whisper

    Args:
        audio_path: Path to the audio file
        model_size: Size of the model to use ('tiny', 'base', 'small', 'medium', 'large')
        language: Language code (e.g., 'en' for English)

    Returns:
        Transcribed text
    """
    return transcribe_with_whisper(audio_path, model_size, language)

# Test the function
if __name__ == "__main__":
    test_audio = "test.wav"
    if os.path.exists(test_audio):
        text = transcribe_audio(test_audio)
        print(f"Transcribed text: {text}")
    else:
        print(f"Test audio file {test_audio} not found.")
