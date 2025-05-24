# Requirements Information

## Environment Details

**Tested Environment:**
- **Python Version**: 3.10.16
- **Operating System**: Windows
- **Package Manager**: pip

## Package Versions (Tested & Working)

### Core Application
- `streamlit==1.45.1` - Web interface framework
- `requests==2.32.3` - HTTP requests for API calls

### AI & Machine Learning
- `google-generativeai==0.8.5` - Google Gemini AI integration
- `openai-whisper==20240930` - Speech-to-text conversion
- `sentence-transformers==4.1.0` - Text embeddings
- `transformers==4.51.3` - HuggingFace transformers
- `huggingface-hub==0.31.4` - HuggingFace model hub

### LangChain Ecosystem
- `langchain==0.3.25` - Core LangChain framework
- `langchain-community==0.3.24` - Community integrations
- `langchain-core==0.3.60` - Core LangChain components
- `langchain-huggingface==0.2.0` - HuggingFace integration
- `langchain-pinecone==0.2.6` - Pinecone integration
- `langchain-text-splitters==0.3.8` - Text splitting utilities

### Vector Database
- `pinecone==6.0.2` - New Pinecone SDK
- `pinecone-client==6.0.0` - Pinecone client library
- `chromadb==1.0.9` - Alternative vector database

### Audio Processing
- `librosa==0.11.0` - Audio analysis and processing
- `soundfile==0.13.1` - Audio file I/O
- `gTTS==2.5.4` - Google Text-to-Speech
- `pyttsx3==2.98` - Cross-platform text-to-speech
- `streamlit_mic_recorder==0.0.8` - Streamlit microphone component
- `vosk==0.3.45` - Offline speech recognition

### Document Processing
- `PyPDF2==3.0.1` - PDF reading and processing
- `pypdf==5.5.0` - Alternative PDF library

### Core Dependencies
- `numpy==2.2.6` - Numerical computing
- `pandas==2.2.3` - Data manipulation
- `torch==2.7.0` - PyTorch deep learning framework
- `torchaudio==2.7.0` - PyTorch audio processing

### System Dependencies (Windows)
- `pywin32==310` - Windows-specific functionality
- `pypiwin32==223` - Windows Python extensions

### Utility Libraries
- `pydantic==2.11.4` - Data validation
- `PyYAML==6.0.2` - YAML processing
- `packaging==24.2` - Package version handling
- `typing_extensions==4.13.2` - Extended typing support

## Installation Files

1. **`requirements.txt`** - Complete installation with all dependencies
2. **`requirements-minimal.txt`** - Essential packages only for core functionality
3. **`install.py`** - Automated installation script

## Key Notes

### Pinecone SDK Update
- **Important**: The Pinecone SDK has been updated from `pinecone-client` to `pinecone`
- Both packages are included for compatibility
- Uses new initialization pattern: `Pinecone()` constructor

### Python Version Compatibility
- **Recommended**: Python 3.10.16 (tested environment)
- **Minimum**: Python 3.10+ (for best compatibility)
- **Note**: Some packages may work with Python 3.8+, but not tested

### Platform-Specific Notes
- **Windows**: Includes `pywin32` for text-to-speech support
- **macOS/Linux**: May require additional audio system dependencies
- **Audio Processing**: Requires system audio libraries (usually pre-installed)

## Installation Commands

```bash
# Full installation (recommended)
pip install -r requirements.txt

# Minimal installation (faster)
pip install -r requirements-minimal.txt

# Automated installation with setup
python install.py
```

## Troubleshooting

### Common Issues
1. **Audio Processing**: Install system audio libraries if needed
2. **Whisper**: May require `ffmpeg` for audio conversion
3. **Windows**: Ensure Windows SDK is installed for `pywin32`
4. **Memory**: Large models (Whisper, Transformers) require sufficient RAM

### Version Conflicts
If you encounter version conflicts, create a fresh virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```
