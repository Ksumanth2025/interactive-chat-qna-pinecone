# pinecone_chat_interface.py - Modern Chat Interface with Pinecone Knowledge Base
import streamlit as st
import tempfile
import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Import Pinecone for vector storage
from langchain_pinecone import PineconeVectorStore
import pinecone

# Import required libraries for audio processing
import whisper
print("Using OpenAI Whisper for speech recognition (much more accurate)")

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

# === PINECONE SETUP AND FUNCTIONS ===
def initialize_pinecone(api_key):
    """Initialize Pinecone with the provided API key"""
    try:
        # Try different environment settings
        environments_to_try = ["gcp-starter", "us-west1-gcp", "us-east1-gcp", "eu-west1-gcp", "asia-southeast1-gcp"]

        for env in environments_to_try:
            try:
                # Initialize Pinecone with the older API
                pinecone.init(api_key=api_key, environment=env)
                # If we get here, initialization was successful
                print(f"Pinecone initialized successfully with environment: {env}")
                st.session_state.pinecone_environment = env
                return True
            except Exception as specific_env_error:
                print(f"Failed with environment {env}: {specific_env_error}")
                continue

        # If we get here, all environment attempts failed
        raise Exception("All environment attempts failed. Please check your API key and Pinecone account settings.")

    except Exception as e:
        error_message = f"Error initializing Pinecone: {e}"
        print(error_message)
        st.error(error_message)

        # Show additional troubleshooting information
        st.error("""
        Troubleshooting tips:
        1. Verify your API key is correct and not expired
        2. Check if your Pinecone account is active
        3. Try creating an index manually in the Pinecone console first
        4. Check your network connection
        """)
        return False

def load_and_split_pdf(pdf_path, chunk_size=1000, chunk_overlap=200):
    """Load and split a PDF into chunks for processing"""
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        return chunks
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return []

def setup_pinecone_vector_store(pdf_chunks, index_name="pdf-qa-index"):
    """Set up Pinecone vector store with the provided PDF chunks"""
    try:
        # Check if index exists, if not create it
        try:
            existing_indexes = pinecone.list_indexes()
            print(f"Existing indexes: {existing_indexes}")

            if index_name not in existing_indexes:
                # Create a new index
                print(f"Creating new index: {index_name}")
                pinecone.create_index(
                    name=index_name,
                    dimension=768,  # Dimension for 'sentence-transformers/all-mpnet-base-v2'
                    metric="cosine"
                )
                print(f"Created new Pinecone index: {index_name}")
                # Wait for index to be ready
                import time
                time.sleep(20)  # Increased wait time to ensure index is ready
            else:
                print(f"Using existing index: {index_name}")

        except Exception as index_error:
            print(f"Error with index operations: {index_error}")
            st.warning(f"There was an issue with the Pinecone index: {index_error}")
            st.info("Attempting to proceed with existing index if available...")

        # Initialize the embedding model
        print("Initializing embedding model...")
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

        # Create the vector store
        print("Creating vector store...")
        vector_store = PineconeVectorStore.from_documents(
            documents=pdf_chunks,
            embedding=embedding_model,
            index_name=index_name
        )

        print(f"Vector store created with {len(pdf_chunks)} chunks")
        return vector_store
    except Exception as e:
        error_message = f"Error setting up Pinecone vector store: {e}"
        print(error_message)
        st.error(error_message)

        # Show additional troubleshooting information
        st.error("""
        Troubleshooting tips for vector store issues:
        1. Check if your index was created successfully
        2. Verify your embedding model is working
        3. Make sure your document chunks are valid
        4. Try with a smaller number of chunks first
        """)
        return None

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

# === WHISPER TRANSCRIPTION FUNCTION ===
def transcribe_with_whisper(audio_path, model_size="base"):
    """
    Transcribe audio using Whisper with robust error handling

    Args:
        audio_path: Path to the audio file
        model_size: Size of the Whisper model to use

    Returns:
        Transcribed text
    """
    try:
        # Force CPU to avoid CUDA errors
        device = "cpu"

        # Load the model
        model = whisper.load_model(model_size, device=device)

        # Try the standard transcription first
        try:
            result = model.transcribe(audio_path, fp16=False)
            return result["text"].strip()
        except Exception as e:
            print(f"Standard transcription failed: {e}")

            # Try the lower-level API approach
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(device)
            _, probs = model.detect_language(mel)
            detected_lang = max(probs, key=probs.get)
            options = whisper.DecodingOptions(language=detected_lang, fp16=False)
            result = whisper.decode(model, mel, options)
            return result.text.strip()
    except Exception as e:
        print(f"Whisper transcription error: {e}")
        return ""

# === CUSTOM CSS FOR CHAT INTERFACE ===
def apply_custom_css():
    st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #f9f9f9;
    }

    /* Chat container */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
        padding: 20px 0;
    }

    /* Message bubbles */
    .user-bubble {
        background-color: #e1f5fe;
        border-radius: 18px 18px 0 18px;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 80%;
        align-self: flex-end;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }

    .assistant-bubble {
        background-color: #f5f5f5;
        border-radius: 18px 18px 18px 0;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 80%;
        align-self: flex-start;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }

    /* Chat input area */
    .chat-input {
        display: flex;
        gap: 10px;
        padding: 10px;
        background-color: white;
        border-radius: 24px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        margin: 20px 0;
    }

    /* Input field */
    .stTextInput input {
        border-radius: 20px !important;
        border: 1px solid #e0e0e0 !important;
        padding: 10px 15px !important;
    }

    /* Buttons */
    .stButton button {
        border-radius: 50% !important;
        width: 40px !important;
        height: 40px !important;
        padding: 8px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        transition: all 0.3s ease !important;
    }

    /* Send button */
    .send-btn button {
        background-color: #4CAF50 !important;
        color: white !important;
    }

    /* Record button */
    .record-btn button {
        background-color: #f44336 !important;
        color: white !important;
    }

    /* Upload button */
    .upload-btn {
        background-color: #2196F3 !important;
        border-radius: 50% !important;
        width: 40px !important;
        height: 40px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
    }

    /* Play again button */
    .play-again-btn button {
        background-color: #9C27B0 !important;
        color: white !important;
        font-size: 12px !important;
        width: auto !important;
        border-radius: 20px !important;
        padding: 4px 12px !important;
    }

    /* Button hover effects */
    .stButton button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    }

    /* Audio player */
    .stAudio {
        margin-top: 5px;
    }

    /* Hide audio player but keep it functional */
    .audio-container audio {
        display: none !important;
    }
    .audio-container div {
        display: none !important;
    }

    /* Timestamp */
    .timestamp {
        font-size: 0.7em;
        color: #888;
        margin-top: 4px;
    }

    /* File uploader */
    .uploadedFile {
        border-radius: 12px !important;
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: #f0f2f6;
    }
    </style>

    <script>
    // Function to auto-play all audio elements
    function autoPlayAudio() {
        // Find all audio elements
        const audioElements = document.querySelectorAll('audio');

        // Play each audio element
        audioElements.forEach(audio => {
            // Only play if not already playing
            if (audio.paused) {
                // Add event listener for when audio can play
                audio.addEventListener('canplaythrough', function() {
                    // Try to play the audio
                    const playPromise = audio.play();

                    // Handle play promise (required for Chrome)
                    if (playPromise !== undefined) {
                        playPromise.catch(error => {
                            console.log("Auto-play prevented by browser:", error);
                        });
                    }
                }, { once: true });

                // Force load the audio
                audio.load();
            }
        });
    }

    // Run the function when the page loads
    document.addEventListener('DOMContentLoaded', autoPlayAudio);

    // Also run it periodically to catch new audio elements
    setInterval(autoPlayAudio, 1000);
    </script>
    """, unsafe_allow_html=True)

# === STREAMLIT UI ===
def main():
    st.set_page_config(
        page_title="💬 Interactive Chat QnA with Pinecone",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Apply custom CSS
    apply_custom_css()

    # Sidebar for settings and PDF upload
    with st.sidebar:
        st.title("📋 Document & Settings")

        # Pinecone API Key
        st.header("🔑 Pinecone API Key")

        # Show information about Pinecone API keys
        st.info("""
        Your Pinecone API key should start with 'pcsk_' or 'pc_'.
        You can find it in your Pinecone dashboard under API Keys.
        """)

        # API key input with validation
        pinecone_api_key = st.text_input(
            "Enter your Pinecone API Key",
            value="",
            type="password",
            help="Your Pinecone API key (starts with 'pcsk_' or 'pc_')"
        )

        # Basic validation
        if pinecone_api_key and not (pinecone_api_key.startswith("pcsk_") or pinecone_api_key.startswith("pc_")):
            st.warning("⚠️ This doesn't look like a valid Pinecone API key. It should start with 'pcsk_' or 'pc_'.")

        # Initialize button with more information
        if st.button("Initialize Pinecone"):
            if pinecone_api_key:
                with st.spinner("Initializing Pinecone... This may take a few moments."):
                    if initialize_pinecone(pinecone_api_key):
                        st.success("✅ Pinecone initialized successfully!")
                        st.session_state.pinecone_initialized = True

                        # Show environment information if available
                        if 'pinecone_environment' in st.session_state:
                            st.success(f"Connected to Pinecone environment: {st.session_state.pinecone_environment}")
                    else:
                        st.error("❌ Failed to initialize Pinecone. See error details above.")
                        st.session_state.pinecone_initialized = False
            else:
                st.warning("⚠️ Please enter your Pinecone API key.")

        # PDF Upload section
        st.header("📄 Upload Document")
        pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

        # Divider
        st.divider()

        # Settings section
        st.header("⚙️ Settings")

        # Speech-to-text settings
        st.subheader("Speech-to-Text Settings")

        # Whisper model size selection
        model_sizes = ["tiny", "base", "small", "medium", "large"]

        if 'whisper_model_size' not in st.session_state:
            st.session_state.whisper_model_size = "base"

        st.session_state.whisper_model_size = st.selectbox(
            "Select Whisper model size:",
            model_sizes,
            index=model_sizes.index(st.session_state.whisper_model_size)
        )

        # Text-to-speech settings
        st.subheader("Text-to-Speech Settings")

        # Voice selection
        if 'selected_voice' not in st.session_state:
            st.session_state.selected_voice = "Your Voice Clone"

        voice_options = list(AVAILABLE_VOICES.keys())
        st.session_state.selected_voice = st.selectbox(
            "Select voice:",
            voice_options,
            index=voice_options.index(st.session_state.selected_voice) if st.session_state.selected_voice in voice_options else 0
        )

        # Show information about the selected voice
        if st.session_state.selected_voice == "Your Voice Clone":
            st.success(f"Using your personal voice clone from 11 Labs!")
            st.info("Your voice clone: Indian | Classical English Story | Male | Middle Aged")
        elif st.session_state.selected_voice in AVAILABLE_VOICES:
            st.success(f"Using 11 Labs voice: {st.session_state.selected_voice}")
        else:
            st.info("Using default text-to-speech")

    # Initialize session state
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'qa_ready' not in st.session_state:
        st.session_state.qa_ready = False
    if 'audio_paths' not in st.session_state:
        st.session_state.audio_paths = {}
    if 'pinecone_initialized' not in st.session_state:
        st.session_state.pinecone_initialized = False

    # Main content area
    st.title("💬 Interactive Chat with Pinecone Knowledge Base")

    # Check if Pinecone is initialized
    if not st.session_state.pinecone_initialized:
        st.warning("⚠️ Please initialize Pinecone with your API key in the sidebar first.")
        return

    # Process PDF when uploaded
    if pdf_file and not st.session_state.qa_ready:
        with st.spinner("Processing PDF and storing in Pinecone..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                tmp_pdf.write(pdf_file.read())
                pdf_path = tmp_pdf.name

            # Import Gemini model
            import google.generativeai as genai
            google_api_key = os.environ.get("GOOGLE_API_KEY")
            if not google_api_key:
                st.error("❌ Google API key not found. Please set GOOGLE_API_KEY environment variable.")
                return
            genai.configure(api_key=google_api_key)

            model = genai.GenerativeModel("models/gemini-1.5-flash-8b")

            # Process PDF and setup QA chain with Pinecone
            pdf_chunks = load_and_split_pdf(pdf_path)
            if not pdf_chunks:
                st.error("❌ Failed to process PDF. Please try another file.")
                return

            st.info(f"PDF processed into {len(pdf_chunks)} chunks. Storing in Pinecone...")

            # Setup Pinecone vector store
            vector_store = setup_pinecone_vector_store(pdf_chunks)
            if not vector_store:
                st.error("❌ Failed to store document in Pinecone. Please check your API key and try again.")
                return

            # Setup QA chain
            qa_chain = setup_custom_qa_chain(vector_store, model)

            st.session_state.qa_chain = qa_chain
            st.session_state.qa_ready = True
            st.success("✅ PDF processed and stored in Pinecone successfully! You can now ask questions.")
            st.rerun()  # Refresh the page to show the chat interface

    # Display conversation history
    if st.session_state.conversation:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)

        for message in st.session_state.conversation:
            timestamp = message.get("timestamp", "now")

            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-bubble">
                    {message['content']}
                    <div class="timestamp">{timestamp}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="assistant-bubble">
                    {message['content']}
                    <div class="timestamp">{timestamp}</div>
                </div>
                """, unsafe_allow_html=True)

                # Automatically play audio for assistant messages without showing the player
                message_idx = st.session_state.conversation.index(message)
                if message_idx % 2 == 1:  # Assistant messages are at odd indices
                    answer_idx = message_idx // 2
                    if answer_idx in st.session_state.audio_paths:
                        audio_path = st.session_state.audio_paths[answer_idx]
                        if os.path.exists(audio_path):
                            # Play the audio automatically using Streamlit's audio component (hidden by CSS)
                            st.markdown('<div class="audio-container">', unsafe_allow_html=True)
                            st.audio(audio_path, format="audio/mp3", start_time=0)
                            st.markdown('</div>', unsafe_allow_html=True)

                            # Also add a small play button if user wants to hear it again
                            with st.container():
                                st.markdown('<div class="play-again-btn">', unsafe_allow_html=True)
                                st.button(f"🔊 Play again", key=f"play_{message_idx}")
                                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Chat input area (only show if PDF is processed)
    if st.session_state.qa_ready:
        st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)

        # Create columns for the chat input components
        col1, col2, col3, col4 = st.columns([0.7, 0.1, 0.1, 0.1])

        with col1:
            question = st.text_input("Type your message...", key="text_question", label_visibility="collapsed")

        with col2:
            # Audio file upload button with custom styling
            st.markdown('<div class="upload-btn">', unsafe_allow_html=True)
            audio_file = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a", "ogg"], key="audio_upload", label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            # Record audio button with custom styling
            st.markdown('<div class="record-btn">', unsafe_allow_html=True)
            try:
                from streamlit_mic_recorder import mic_recorder

                if 'audio_recorder_key' not in st.session_state:
                    st.session_state.audio_recorder_key = 0

                # Track if we've already processed the current recording
                if 'last_processed_recording' not in st.session_state:
                    st.session_state.last_processed_recording = None

                # Use mic_recorder with unique key
                audio_result = mic_recorder(
                    key=f"recorder_{st.session_state.audio_recorder_key}",
                    start_prompt="🎤",
                    stop_prompt="⏹️",
                    use_container_width=True
                )

                # Check if this is a new recording that hasn't been processed yet
                if (audio_result and isinstance(audio_result, dict) and 'bytes' in audio_result and
                    audio_result.get('bytes') != st.session_state.last_processed_recording):
                    # Store this recording to avoid processing it multiple times
                    st.session_state.last_processed_recording = audio_result.get('bytes')
                else:
                    # If we've already processed this recording, set to None to avoid reprocessing
                    audio_result = None

            except ImportError:
                st.button("🎤", disabled=True, help="Install streamlit-mic-recorder to enable recording")
                audio_result = None
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            # Send button with custom styling
            st.markdown('<div class="send-btn">', unsafe_allow_html=True)
            send_pressed = st.button("📤", help="Send message")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Process text input
        if send_pressed and question:
            process_text_question(question)

        # Process audio file upload
        if audio_file is not None:
            process_audio_file(audio_file)

        # Process recorded audio
        if audio_result and isinstance(audio_result, dict) and 'bytes' in audio_result:
            process_recorded_audio(audio_result['bytes'])

        # Option to clear conversation
        if st.session_state.conversation:
            if st.button("🗑️ Clear Conversation"):
                st.session_state.conversation = []
                if 'audio_paths' in st.session_state:
                    st.session_state.audio_paths = {}
                st.rerun()
    else:
        st.info("👆 Please upload a PDF document in the sidebar to start chatting.")

# === PROCESSING FUNCTIONS ===
def process_text_question(question):
    with st.spinner("Thinking..."):
        # Process the question and get answer
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

        # Generate speech response
        answer_audio_path = os.path.join(tempfile.gettempdir(), f"answer_{int(time.time())}.mp3")

        # Get the selected voice ID
        selected_voice = st.session_state.get('selected_voice', 'Your Voice Clone')
        voice_id = AVAILABLE_VOICES.get(selected_voice)

        # Generate speech with the selected voice
        text_to_speech(answer, answer_audio_path, voice_id=voice_id)

        # Store the audio path for later playback
        answer_idx = len(st.session_state.conversation) // 2 - 1
        st.session_state.audio_paths[answer_idx] = answer_audio_path

        # Auto-play the audio without showing the player
        # Use Streamlit's audio component with CSS to hide it
        if os.path.exists(answer_audio_path):
            # Play the audio using Streamlit's audio component (it will be hidden by CSS)
            st.markdown('<div class="audio-container">', unsafe_allow_html=True)
            st.audio(answer_audio_path, format="audio/mp3", start_time=0)
            st.markdown('</div>', unsafe_allow_html=True)

        st.rerun()

def process_audio_file(audio_file):
    with st.spinner("Transcribing your question..."):
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(audio_file.getvalue())
            audio_path = tmp_file.name

        # Display the audio for debugging (temporarily enabled to help diagnose issues)
        st.audio(audio_path, format=f"audio/{audio_file.name.split('.')[-1]}")
        st.info("👆 This is the audio being processed. If you can hear it clearly, but transcription fails, there might be an issue with the transcription service.")

        # Get selected Whisper model size
        model_size = st.session_state.get('whisper_model_size', 'base')

        # Use our improved transcription function
        question = transcribe_with_whisper(audio_path, model_size)

        # If we got a question, process it
        if question and question.strip():
            st.success(f"Transcribed: {question}")
            process_text_question(question)
        else:
            st.error("❌ Could not transcribe audio. Please try again with a clearer recording.")
            st.info("Tips: Make sure the audio file contains clear speech and is in a supported format.")

def process_recorded_audio(audio_bytes):
    with st.spinner("Transcribing your question..."):
        # Save recorded audio to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            audio_path = tmp_file.name

        # Display the audio for debugging (temporarily enabled to help diagnose issues)
        st.audio(audio_path, format="audio/wav")
        st.info("👆 This is the audio being processed. If you can hear it clearly, but transcription fails, there might be an issue with the transcription service.")

        # Get selected Whisper model size
        model_size = st.session_state.get('whisper_model_size', 'base')

        # Use our improved transcription function
        question = transcribe_with_whisper(audio_path, model_size)

        # If we got a question, process it
        if question and question.strip():
            st.success(f"Transcribed: {question}")
            process_text_question(question)
        else:
            st.error("❌ Could not transcribe audio. Please try again with a clearer recording.")
            st.info("Tips: Speak clearly, reduce background noise, and try a different microphone if possible.")

        # Increment the recorder key to reset it for next recording
        st.session_state.audio_recorder_key += 1

if __name__ == "__main__":
    main()