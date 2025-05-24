# updated_live.py - Interactive QnA Interface with Multiple Voice Selection
import streamlit as st
import tempfile
import os
import time
# These imports are used in file processing
import numpy as np  # Used for audio array processing
import soundfile as sf  # Used for audio file operations
from main import load_and_split_pdf, setup_vector_store
# Import speech-to-text functionality from Whisper
try:
    from whisper_stt import transcribe_audio
    print("Using OpenAI Whisper for speech recognition (much more accurate)")
except ImportError:
    from main import transcribe_audio
    print("Using Vosk for speech recognition (fallback)")

# Import 11 Labs text-to-speech functionality
try:
    from elevenlabs_tts import text_to_speech_11labs, text_to_speech_with_fallback, get_available_voices

    def text_to_speech(text, output_path, voice_id=None):
        # Use the selected voice ID if provided, otherwise use default
        if voice_id:
            return text_to_speech_11labs(text, output_path, voice_id=voice_id)
        else:
            return text_to_speech_with_fallback(text, output_path)

    # Get available voices for selection
    AVAILABLE_VOICES = get_available_voices()
    print(f"Using 11 Labs text-to-speech module with {len(AVAILABLE_VOICES)} available voices")
except ImportError:
    from main import text_to_speech
    AVAILABLE_VOICES = {"Default": "default"}
    print("Using default text-to-speech module")

# === CUSTOM PROMPT TEMPLATE ===
def get_prompt_template(question, context):
    return f"""You are a friendly and helpful AI assistant with expertise in a wide range of topics.
Use the knowledge base context below to answer the question. If the context doesn't contain enough information,
use your own knowledge to provide a complete answer. Always be polite, respectful, and helpful.

When answering:
1. First prioritize information from the provided context
2. If the context doesn't have enough information, use your general knowledge
3. If you're unsure, acknowledge the limitations and provide the best answer you can
4. Always maintain a helpful and conversational tone

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

# === MODIFIED GEMINI INTEGRATION ===
class CustomGeminiLLM:
    def __init__(self, model):
        self.model = model

    def ask(self, question, context):
        prompt = get_prompt_template(question, context)
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return "I'm sorry, I couldn't generate a response."

# === CUSTOM QA CHAIN ===
def setup_custom_qa_chain(vector_store, model):
    try:
        gemini = CustomGeminiLLM(model)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        def run_qa(question):
            docs = retriever.invoke(question)
            context = "\n\n".join([doc.page_content for doc in docs])
            return gemini.ask(question, context)

        return run_qa
    except Exception as e:
        print(f"Error setting up custom QA chain: {e}")
        return None

# === STREAMLIT UI ===
st.set_page_config(
    page_title="🤖 Interactive QnA with Gemini",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for settings and PDF upload
with st.sidebar:
    st.title("📋 Document & Settings")

    # PDF Upload section
    st.header("📄 Upload Document")
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    # Divider
    st.divider()

    # Settings section
    st.header("⚙️ Settings")
    # Speech-to-text settings
    st.subheader("Speech-to-Text Settings")

    st.success("""
    Using OpenAI Whisper for speech recognition.
    This is a state-of-the-art speech recognition model with much higher accuracy than Vosk.
    """)

    # Whisper model size selection
    model_sizes = ["tiny", "base", "small", "medium", "large"]

    if 'whisper_model_size' not in st.session_state:
        st.session_state.whisper_model_size = "base"

    st.session_state.whisper_model_size = st.selectbox(
        "Whisper Model Size",
        options=model_sizes,
        index=model_sizes.index(st.session_state.whisper_model_size),
        help="Larger models are more accurate but slower and use more memory"
    )

    st.info(f"""
    Selected model: {st.session_state.whisper_model_size}

    - tiny: Fastest, lowest accuracy, ~75MB
    - base: Good balance of speed and accuracy, ~150MB
    - small: Better accuracy, slower, ~500MB
    - medium: High accuracy, much slower, ~1.5GB
    - large: Highest accuracy, slowest, ~3GB
    """)

    # Divider
    st.divider()

    # Text-to-speech settings
    st.subheader("Text-to-Speech Settings")

    # Voice selection
    if 'selected_voice' not in st.session_state:
        st.session_state.selected_voice = "Your Voice Clone"

    voice_options = list(AVAILABLE_VOICES.keys())
    st.session_state.selected_voice = st.selectbox(
        "Select Voice:",
        options=voice_options,
        index=voice_options.index(st.session_state.selected_voice) if st.session_state.selected_voice in voice_options else 0,
        help="Choose from available 11 Labs voices"
    )

    # Show information about the selected voice
    if st.session_state.selected_voice == "Your Voice Clone":
        st.success(f"Using your personal voice clone from 11 Labs!")
        st.info("Your voice clone: Indian | Classical English Story | Male | Middle Aged | Srikant Rajan")
    elif st.session_state.selected_voice in AVAILABLE_VOICES:
        st.success(f"Using 11 Labs voice: {st.session_state.selected_voice}")
        # Add descriptions for standard voices
        voice_descriptions = {
            "Sarah": "American English | Female | Professional",
            "Aria": "American English | Female | Warm",
            "Laura": "British English | Female | Friendly",
            "Charlie": "American English | Male | Casual",
            "George": "British English | Male | Authoritative",
            "Callum": "Scottish English | Male | Young Adult",
            "River": "American English | Non-binary | Calm",
            "Liam": "American English | Male | Energetic",
            "Charlotte": "British English | Female | Mature",
            "Alice": "American English | Female | Young",
            "Matilda": "Australian English | Female | Cheerful"
        }
        if st.session_state.selected_voice in voice_descriptions:
            st.info(voice_descriptions[st.session_state.selected_voice])
    else:
        st.info("Using default text-to-speech")

    # Divider
    st.divider()

# Initialize session state for conversation history
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'audio_buffer' not in st.session_state:
    st.session_state.audio_buffer = []
if 'qa_ready' not in st.session_state:
    st.session_state.qa_ready = False

# Main content area
st.markdown('<div class="header">', unsafe_allow_html=True)
st.title("🤖 Interactive Voice & Text QnA")
st.markdown("</div>", unsafe_allow_html=True)

# Welcome message in a card
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("""
### 👋 Welcome to the Interactive QnA System!

This system allows you to ask questions about your PDF document using:
- 🎤 **Voice input** with OpenAI Whisper for accurate speech recognition
- ⌨️ **Text input** for direct typing
- 🔊 **Voice output** using multiple 11 Labs voices

Upload a PDF document in the sidebar to get started!
""")
st.markdown('</div>', unsafe_allow_html=True)

# Process PDF when uploaded
if pdf_file and not st.session_state.qa_ready:
    with st.spinner("Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            tmp_pdf.write(pdf_file.read())
            pdf_path = tmp_pdf.name

        # Import Gemini model
        import google.generativeai as genai
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        if not google_api_key:
            st.error("❌ Google API key not found. Please set GOOGLE_API_KEY environment variable.")
            st.stop()
        genai.configure(api_key=google_api_key)

        model = genai.GenerativeModel("models/gemini-1.5-flash-8b")

        # Process PDF and setup QA chain
        pdf_chunks = load_and_split_pdf(pdf_path)
        vector_store = setup_vector_store(pdf_chunks)
        qa_chain = setup_custom_qa_chain(vector_store, model)

        st.session_state.qa_chain = qa_chain
        st.session_state.qa_ready = True
        st.success("✅ PDF processed successfully! You can now ask questions.")
        st.rerun()  # Refresh the page to show the chat interface

# === AUDIO CAPTURE ===
if st.session_state.qa_ready:
    st.subheader("💬 Ask Your Question")

    # Option tabs for input method
    tab1, tab2 = st.tabs(["🎤 Voice Input", "⌨️ Text Input"])

    with tab1:
        st.markdown("""
        ### Option 1: Upload an audio file
        Record your question using your device's voice recorder app, then upload the file below.
        """)

        # File uploader for audio files
        audio_file = st.file_uploader("Upload your audio question", type=["wav", "mp3", "m4a", "ogg"])

        if audio_file is not None:
            # Display the audio file
            st.audio(audio_file, format=f"audio/{audio_file.name.split('.')[-1]}")

            # Button to process uploaded audio
            if st.button("🔊 Process Uploaded Audio"):
                with st.spinner("Transcribing your question..."):
                    # Save uploaded file to temp location
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(audio_file.getvalue())
                        audio_path = tmp_file.name

                    # Get selected Whisper model size
                    model_size = st.session_state.get('whisper_model_size', 'base')

                    # Transcribe audio using Whisper
                    st.info(f"Transcribing audio with OpenAI Whisper ({model_size} model)...")

                    # Check if we're using Whisper
                    try:
                        # The import is just to check if whisper is available
                        import whisper  # noqa: F401
                        question = transcribe_audio(audio_path, model_size=model_size)
                    except ImportError:
                        # Fallback to Vosk
                        st.warning("Whisper not available, falling back to Vosk...")
                        question = transcribe_audio(audio_path)

                    # Check if we got a valid transcription or an error message
                    if question and not question.startswith("TRANSCRIPTION_ERROR:"):
                        st.success(f"Transcribed: {question}")

                        # Process the question and get answer
                        with st.spinner("Thinking..."):
                            answer = st.session_state.qa_chain(question)

                            # Get current time for timestamp
                            current_time = time.strftime("%I:%M %p")

                            # Add to conversation history with timestamps
                            st.session_state.conversation.append({
                                "role": "user",
                                "content": question,
                                "timestamp": current_time
                            })
                            st.session_state.conversation.append({
                                "role": "assistant",
                                "content": answer,
                                "timestamp": current_time
                            })

                            # Get the selected voice ID
                            selected_voice = st.session_state.get('selected_voice', 'Your Voice Clone')
                            voice_id = AVAILABLE_VOICES.get(selected_voice)

                            # Generate speech response with the selected voice
                            answer_audio_path = os.path.join(tempfile.gettempdir(), f"answer_{int(time.time())}.mp3")
                            text_to_speech(answer, answer_audio_path, voice_id=voice_id)

                            # Store the audio path for later playback
                            if 'audio_paths' not in st.session_state:
                                st.session_state.audio_paths = {}
                            answer_idx = len(st.session_state.conversation) // 2
                            st.session_state.audio_paths[answer_idx] = answer_audio_path

                            # Play audio immediately
                            st.subheader("🔊 Audio Response:")
                            if os.path.exists(answer_audio_path):
                                try:
                                    st.audio(answer_audio_path, format='audio/mp3')
                                except Exception as e:
                                    st.error(f"Error playing audio: {e}")
                                    st.markdown(f"[Download audio file]({answer_audio_path})")
                            else:
                                st.warning("Audio file could not be generated.")
                    else:
                        # Display detailed error information
                        st.error("❌ Could not transcribe audio. Please try again with a clearer recording.")

                        # If we have a specific error message, display it
                        if question and question.startswith("TRANSCRIPTION_ERROR:"):
                            error_details = question.replace("TRANSCRIPTION_ERROR:", "").strip()
                            st.error(f"Error details: {error_details}")

                            # Provide troubleshooting suggestions based on the error
                            st.info("""
                            ### Troubleshooting suggestions:

                            1. **Check audio format**: Make sure your audio is in a common format (WAV, MP3, etc.)
                            2. **Check audio quality**: Ensure there's clear speech with minimal background noise
                            3. **Try a different model size**: Smaller models are faster but less accurate
                            4. **Check system resources**: Larger models require more memory
                            5. **Check file permissions**: Make sure the system can read the temporary files
                            """)
                        else:
                            st.info("No specific error details available. Try recording with less background noise.")

        st.markdown("""
        ### Option 2: Record directly (if your browser supports it)
        """)

        # Initialize session state for audio recording
        if 'audio_recorder_key' not in st.session_state:
            st.session_state.audio_recorder_key = 0

        # Simple audio recorder
        try:
            from streamlit_mic_recorder import mic_recorder

            result = mic_recorder(
                key=f"recorder_{st.session_state.audio_recorder_key}",
                start_prompt="Click to record",
                stop_prompt="Click to stop recording",
                use_container_width=True
            )

            if result and isinstance(result, dict) and 'bytes' in result:
                # Extract the audio bytes from the result dictionary
                audio_bytes = result['bytes']

                # Display the audio
                st.audio(audio_bytes)

                # Button to process recorded audio
                if st.button("🎙️ Process Recorded Audio"):
                    with st.spinner("Transcribing your question..."):
                        # Save recorded audio to temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            tmp_file.write(audio_bytes)
                            audio_path = tmp_file.name

                        # Get selected Whisper model size
                        model_size = st.session_state.get('whisper_model_size', 'base')

                        # Transcribe audio using Whisper
                        st.info(f"Transcribing audio with OpenAI Whisper ({model_size} model)...")

                        # Check if we're using Whisper
                        try:
                            # The import is just to check if whisper is available
                            import whisper  # noqa: F401
                            question = transcribe_audio(audio_path, model_size=model_size)
                        except ImportError:
                            # Fallback to Vosk
                            st.warning("Whisper not available, falling back to Vosk...")
                            question = transcribe_audio(audio_path)

                        # Check if we got a valid transcription or an error message
                        if question and not question.startswith("TRANSCRIPTION_ERROR:"):
                            st.success(f"Transcribed: {question}")

                            # Process the question and get answer
                            with st.spinner("Thinking..."):
                                answer = st.session_state.qa_chain(question)

                                # Get current time for timestamp
                                current_time = time.strftime("%I:%M %p")

                                # Add to conversation history with timestamps
                                st.session_state.conversation.append({
                                    "role": "user",
                                    "content": question,
                                    "timestamp": current_time
                                })
                                st.session_state.conversation.append({
                                    "role": "assistant",
                                    "content": answer,
                                    "timestamp": current_time
                                })

                                # Get the selected voice ID
                                selected_voice = st.session_state.get('selected_voice', 'Your Voice Clone')
                                voice_id = AVAILABLE_VOICES.get(selected_voice)

                                # Generate speech response with the selected voice
                                answer_audio_path = os.path.join(tempfile.gettempdir(), f"answer_{int(time.time())}.mp3")
                                text_to_speech(answer, answer_audio_path, voice_id=voice_id)

                                # Store the audio path for later playback
                                if 'audio_paths' not in st.session_state:
                                    st.session_state.audio_paths = {}
                                answer_idx = len(st.session_state.conversation) // 2
                                st.session_state.audio_paths[answer_idx] = answer_audio_path

                                # Play audio immediately
                                st.subheader("🔊 Audio Response:")
                                if os.path.exists(answer_audio_path):
                                    try:
                                        st.audio(answer_audio_path, format='audio/mp3')
                                    except Exception as e:
                                        st.error(f"Error playing audio: {e}")
                                        st.markdown(f"[Download audio file]({answer_audio_path})")
                                else:
                                    st.warning("Audio file could not be generated.")

                                # Increment the recorder key to reset it for next recording
                                st.session_state.audio_recorder_key += 1
                        else:
                            # Display detailed error information
                            st.error("❌ Could not transcribe audio. Please try again with a clearer recording.")

                            # If we have a specific error message, display it
                            if question and question.startswith("TRANSCRIPTION_ERROR:"):
                                error_details = question.replace("TRANSCRIPTION_ERROR:", "").strip()
                                st.error(f"Error details: {error_details}")

                                # Provide troubleshooting suggestions based on the error
                                st.info("""
                                ### Troubleshooting suggestions:

                                1. **Check audio format**: Make sure your audio is in a common format (WAV, MP3, etc.)
                                2. **Check audio quality**: Ensure there's clear speech with minimal background noise
                                3. **Try a different model size**: Smaller models are faster but less accurate
                                4. **Check system resources**: Larger models require more memory
                                5. **Check file permissions**: Make sure the system can read the temporary files
                                """)
                            else:
                                st.info("No specific error details available. Try recording with less background noise.")
        except ImportError:
            st.warning("The streamlit_mic_recorder package is not installed. You can only use the file upload option.")
            st.markdown("""
            To enable direct recording, install the package with:
            ```
            pip install streamlit-mic-recorder
            ```
            """)
        except Exception as e:
            st.error(f"Error with audio recorder: {str(e)}")
            st.info("Please use the file upload option instead.")

    with tab2:
        # Text input for questions
        text_question = st.text_area("Type your question here:", height=100)

        if st.button("📤 Submit Question"):
            if text_question:
                with st.spinner("Thinking..."):
                    # Process the question and get answer
                    answer = st.session_state.qa_chain(text_question)

                    # Get current time for timestamp
                    current_time = time.strftime("%I:%M %p")

                    # Add to conversation history with timestamps
                    st.session_state.conversation.append({
                        "role": "user",
                        "content": text_question,
                        "timestamp": current_time
                    })
                    st.session_state.conversation.append({
                        "role": "assistant",
                        "content": answer,
                        "timestamp": current_time
                    })

                    # Get the selected voice ID
                    selected_voice = st.session_state.get('selected_voice', 'Your Voice Clone')
                    voice_id = AVAILABLE_VOICES.get(selected_voice)

                    # Generate speech response with the selected voice
                    answer_audio_path = os.path.join(tempfile.gettempdir(), f"answer_{int(time.time())}.mp3")
                    text_to_speech(answer, answer_audio_path, voice_id=voice_id)

                    # Store the audio path for later playback
                    if 'audio_paths' not in st.session_state:
                        st.session_state.audio_paths = {}
                    answer_idx = len(st.session_state.conversation) // 2
                    st.session_state.audio_paths[answer_idx] = answer_audio_path

                    # Play audio immediately
                    st.subheader("🔊 Audio Response:")
                    if os.path.exists(answer_audio_path):
                        try:
                            st.audio(answer_audio_path, format='audio/mp3')
                        except Exception as e:
                            st.error(f"Error playing audio: {e}")
                            st.markdown(f"[Download audio file]({answer_audio_path})")
                    else:
                        st.warning("Audio file could not be generated.")
            else:
                st.warning("⚠️ Please enter a question.")

    # Display conversation history
    if st.session_state.conversation:
        st.subheader("💬 Conversation History")

        for message in st.session_state.conversation:
            timestamp = message.get("timestamp", "now")

            if message["role"] == "user":
                st.markdown(f"**🧑 You ({timestamp}):** {message['content']}")
            else:
                st.markdown(f"**🤖 Assistant ({timestamp}):** {message['content']}")

                # Play audio for assistant messages
                if message["role"] == "assistant":
                    # Generate a unique but consistent ID for this message
                    message_idx = st.session_state.conversation.index(message)
                    if message_idx % 2 == 1:  # Assistant messages are at odd indices
                        answer_idx = message_idx // 2
                        # Use the same audio file that was generated when the answer was created
                        if 'audio_paths' not in st.session_state:
                            st.session_state.audio_paths = {}

                        if answer_idx in st.session_state.audio_paths:
                            audio_path = st.session_state.audio_paths[answer_idx]
                            if os.path.exists(audio_path):
                                try:
                                    st.audio(audio_path, format='audio/mp3')
                                except Exception as e:
                                    st.error(f"Error playing audio: {e}")
                                    st.markdown(f"[Download audio file]({audio_path})")
                            else:
                                st.warning(f"Audio file not found: {audio_path}")

        # Option to clear conversation
        if st.button("🗑️ Clear Conversation"):
            st.session_state.conversation = []
            if 'audio_paths' in st.session_state:
                st.session_state.audio_paths = {}
            st.rerun()
else:
    st.info("👆 Please upload a PDF document to start asking questions.")
