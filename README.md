# Interactive Chat QnA with Direct Pinecone API

This project demonstrates a real-time voice chat application using Google's Gemini 1.5 Flash model, Pinecone vector database, and 11 Labs text-to-speech, with speech recognition capabilities.

## Files

- `voice_chat.py` - The original application (your initial code)
- `simple_voice_chat.py` - A simplified version using Bokeh for speech recognition
- `browser_voice_chat.py` - **RECOMMENDED** - Uses HTML5 Web Speech API with form submission
- `simple_html_voice_chat.py` - Uses HTML5 Web Speech API with component communication
- `basic_voice_chat.py` - A basic version with only text input and speech output
- `test_voice_chat.py` - A test suite to verify all components are working
- `requirements.txt` - Dependencies needed to run the application

## Which File Should I Use?

If you're having trouble with the speech recognition button not appearing:

1. **Try `browser_voice_chat.py` first** - This uses a different approach with HTML and JavaScript
2. If that doesn't work, try `simple_html_voice_chat.py` - Another HTML-based approach
3. If you just want to test the text-to-speech, use `basic_voice_chat.py`

## Setup

1. Install the required dependencies:

   **Option A: Full Installation (Recommended)**
   ```bash
   pip install -r requirements.txt
   ```

   **Option B: Minimal Installation (Faster)**
   ```bash
   pip install -r requirements-minimal.txt
   ```

   **Option C: Manual Installation (Core packages only)**
   ```bash
   pip install streamlit google-generativeai openai-whisper langchain langchain-community langchain-huggingface sentence-transformers pinecone-client librosa soundfile gTTS PyPDF2 requests streamlit-mic-recorder
   ```

2. API Key Setup (choose one option):

   **Option 1: Streamlit Secrets (Recommended for Security)**

   Copy the template and add your API keys:
   ```bash
   # Copy the template
   cp .streamlit/secrets.toml.template .streamlit/secrets.toml

   # Edit with your actual API keys
   # The file should look like this:
   ```
   ```toml
   [pinecone]
   api_key = "pcsk_your-actual-pinecone-key"
   environment = "us-east-1"
   index_name = "qna"

   [google]
   api_key = "AIza_your-actual-google-key"

   [elevenlabs]
   api_key = "sk_your-actual-elevenlabs-key"
   ```

   **Option 2: Environment Variables**
   ```bash
   # On Windows
   set PINECONE_API_KEY=your_pinecone_api_key_here
   set GOOGLE_API_KEY=your_google_api_key_here
   set ELEVENLABS_API_KEY=your_elevenlabs_api_key_here

   # On macOS/Linux
   export PINECONE_API_KEY=your_pinecone_api_key_here
   export GOOGLE_API_KEY=your_google_api_key_here
   export ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
   ```

   **Option 3: Manual Entry**
   - The app will prompt you to enter API keys in the interface

   **Get your API keys from:**
   - Pinecone: [Pinecone Console](https://www.pinecone.io/)
   - Google Gemini: [Google AI Studio](https://makersuite.google.com/)
   - 11 Labs: [11 Labs](https://elevenlabs.io/)

3. Run the main application:

```bash
# Run the main PDF chat application
streamlit run pinecone_direct_api.py

# Alternative applications (if available)
streamlit run updated_live.py
streamlit run pinecone_chat_interface.py
```

## Features

- **Speech Recognition**: Uses the Web Speech API through your browser
- **AI Responses**: Powered by Google's Gemini 1.5 Flash model
- **Text-to-Speech**: Uses pyttsx3 to speak the AI's responses
- **Chat History**: The simplified version maintains a conversation history
- **Voice Selection**: Choose from available system voices for the AI's responses

## How to Use

1. **Start the application** using one of the commands in the Setup section
2. **Look for the large blue "🎤 Speak" button** - this is your microphone button
3. **Click the button** and your browser will ask for microphone permission
4. **Speak clearly** after granting permission
5. **Wait for the response** - Gemini will process your speech and respond both in text and audio
6. You can also type your question in the text input field if you prefer

## Requirements

- **Python 3.10.16** (tested and recommended version)
- A modern web browser that supports the Web Speech API (Chrome recommended)
- API keys for:
  - Pinecone (vector database)
  - Google Gemini (AI responses)
  - 11 Labs (text-to-speech)

## Troubleshooting

If you encounter issues:

1. **Speech recognition not working**: Make sure you're using a supported browser (Chrome works best) and have granted microphone permissions.

2. **Gemini API errors**: Verify your API key is correct and has the necessary permissions.

3. **Text-to-speech issues**: pyttsx3 uses your system's speech engines. Make sure you have a speech engine installed:
   - Windows: Uses SAPI5 (built-in)
   - macOS: Uses NSSpeechSynthesizer (built-in)
   - Linux: Requires espeak (`sudo apt-get install espeak`)

4. **Installation problems**: Try creating a fresh virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Installation Troubleshooting

### Common Package Installation Issues:

**1. Audio Processing Issues:**
```bash
# If you get errors with librosa or soundfile:
pip install librosa soundfile --upgrade

# On Windows, you might need:
pip install pywin32
```

**2. Whisper Installation Issues:**
```bash
# If OpenAI Whisper fails to install:
pip install --upgrade pip setuptools wheel
pip install openai-whisper

# Alternative: Use faster-whisper
pip install faster-whisper
```

**3. LangChain Version Conflicts:**
```bash
# If you get LangChain import errors:
pip install langchain langchain-community langchain-huggingface --upgrade
```

**4. Streamlit Microphone Issues:**
```bash
# If microphone recording doesn't work:
pip install streamlit-mic-recorder --upgrade
```

**5. Vector Database Issues:**
```bash
# For Pinecone connection issues:
pip install pinecone-client --upgrade

# If using ChromaDB instead:
pip install chromadb --upgrade
```

## Testing

The `test_voice_chat.py` file includes a comprehensive test suite to verify:

1. Environment setup and package installation
2. Individual component tests (Gemini API, speech recognition, text-to-speech)
3. Integration tests with sample inputs
4. Manual testing interface

Run the test suite to diagnose any issues before using the main application.
