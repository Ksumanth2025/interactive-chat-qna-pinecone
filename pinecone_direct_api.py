# pinecone_direct_api.py - Chat Interface with Direct Pinecone API Integration
import streamlit as st
import tempfile
import os
import time
import requests
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Import required libraries for audio processing
import whisper

# Import 11 Labs text-to-speech functionality
try:
    from elevenlabs_tts import text_to_speech_11labs, text_to_speech_with_fallback, get_available_voices

    def text_to_speech(text, output_path, voice_id=None):
        # Get 11 Labs API key from secrets
        try:
            elevenlabs_api_key = st.secrets["elevenlabs"]["api_key"]
        except (KeyError, FileNotFoundError):
            elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY")

        # Use the selected voice ID if provided, otherwise use default
        if voice_id:
            return text_to_speech_11labs(text, output_path, api_key=elevenlabs_api_key, voice_id=voice_id)
        else:
            return text_to_speech_with_fallback(text, output_path)

    # Get available voices for selection
    AVAILABLE_VOICES = get_available_voices()
    print(f"Using 11 Labs text-to-speech module with {len(AVAILABLE_VOICES)} available voices")
except ImportError:
    from main import text_to_speech
    AVAILABLE_VOICES = {"Default": "default"}
    print("Using default text-to-speech module")

# === DIRECT PINECONE API FUNCTIONS ===
# Note: This uses direct REST API calls to Pinecone
# For the new Pinecone Python SDK (renamed from 'pinecone-client' to 'pinecone'),
# you would use: from pinecone import Pinecone; pc = Pinecone(api_key="your-key"); index = pc.Index("qna")
class PineconeDirectAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.pinecone.io"
        self.headers = {
            "Api-Key": api_key,
            "Content-Type": "application/json"
        }
        self.index_name = "qna"  # Updated to match your index name
        self.dimension = 768  # For 'sentence-transformers/all-mpnet-base-v2'
        self.namespace = "default"
        # Your specific index host URL
        self.index_host = "https://qna-ydx1c5v.svc.aped-4627-b74a.pinecone.io"

    def check_connection(self):
        """Test the connection to Pinecone API"""
        try:
            response = requests.get(
                f"{self.base_url}/actions/whoami",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Connection test failed: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Response: {e.response.text}")
            raise e

    def list_indexes(self):
        """List all indexes in the Pinecone account"""
        try:
            response = requests.get(
                f"{self.base_url}/databases",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"List indexes failed: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Response: {e.response.text}")
            raise e

    def create_index(self):
        """Check if index exists (your 'qna' index should already exist)"""
        try:
            # Check if the index already exists
            try:
                indexes = self.list_indexes()
                for index in indexes:
                    if index.get('name') == self.index_name:
                        print(f"Index {self.index_name} already exists and is ready to use")
                        return {"message": f"Index {self.index_name} already exists"}
            except Exception as e:
                print(f"Error checking existing indexes: {e}")
                # If we can't list indexes, assume the index exists since you provided the details

            # Since you already have the 'qna' index, we don't need to create it
            print(f"Using existing index: {self.index_name}")
            return {"message": f"Using existing index: {self.index_name}"}

        except Exception as e:
            print(f"Index check failed: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Response: {e.response.text}")
            # Don't raise an exception, just proceed assuming the index exists
            return {"message": f"Proceeding with existing index: {self.index_name}"}

    def upsert_vectors(self, vectors, metadatas):
        """Upsert vectors to the index"""
        try:
            # Prepare the vectors in the format Pinecone expects
            vector_data = []
            for i, (vector, metadata) in enumerate(zip(vectors, metadatas)):
                # Convert numpy arrays to lists if needed
                if hasattr(vector, 'tolist'):
                    vector_values = vector.tolist()
                else:
                    vector_values = vector

                # Ensure metadata is serializable
                clean_metadata = {}
                for k, v in metadata.items():
                    if k == 'text':
                        # Ensure text is included
                        clean_metadata[k] = str(v)
                    elif isinstance(v, (str, int, float, bool)) or v is None:
                        # Only include simple types
                        clean_metadata[k] = v
                    else:
                        # Convert complex types to strings
                        clean_metadata[k] = str(v)

                vector_data.append({
                    "id": metadata.get('id', f"vec_{i}"),
                    "values": vector_values,
                    "metadata": clean_metadata
                })

            # Print the first vector for debugging
            if vector_data:
                print(f"First vector ID: {vector_data[0]['id']}")
                print(f"First vector metadata keys: {list(vector_data[0]['metadata'].keys())}")
                print(f"First vector dimension: {len(vector_data[0]['values'])}")

            payload = {
                "vectors": vector_data,
                "namespace": self.namespace
            }

            # Try the upsert using your specific index host
            try:
                response = requests.post(
                    f"{self.index_host}/vectors/upsert",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                print(f"Index host upsert failed: {e}")
                if hasattr(e, 'response') and e.response:
                    print(f"Response: {e.response.text}")

                # Try standard API endpoint as fallback
                print("Trying standard API endpoint...")
                response = requests.post(
                    f"{self.base_url}/databases/{self.index_name}/vectors/upsert",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            print(f"Upsert vectors failed: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Response: {e.response.text}")
            raise e

    def delete_all_vectors(self):
        """Delete all vectors from the index namespace"""
        try:
            payload = {
                "deleteAll": True,
                "namespace": self.namespace
            }

            # Try the delete using your specific index host
            try:
                response = requests.post(
                    f"{self.index_host}/vectors/delete",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                print(f"Successfully cleared all vectors from namespace '{self.namespace}'")
                return result
            except Exception as e:
                print(f"Index host delete failed: {e}")
                if hasattr(e, 'response') and e.response:
                    print(f"Response: {e.response.text}")

                # Try standard API endpoint as fallback
                print("Trying standard API endpoint for delete...")
                response = requests.post(
                    f"{self.base_url}/databases/{self.index_name}/vectors/delete",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                print(f"Successfully cleared all vectors from namespace '{self.namespace}' (fallback)")
                return result
        except Exception as e:
            print(f"Delete all vectors failed: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Response: {e.response.text}")
            raise e

    def query_vectors(self, query_vector, top_k=3):
        """Query vectors from the index"""
        try:
            # Convert numpy arrays to lists if needed
            if hasattr(query_vector, 'tolist'):
                vector_values = query_vector.tolist()
            else:
                vector_values = query_vector

            payload = {
                "vector": vector_values,
                "topK": top_k,
                "includeValues": False,
                "includeMetadata": True,
                "namespace": self.namespace
            }

            # Try the query using your specific index host
            try:
                response = requests.post(
                    f"{self.index_host}/query",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                print(f"Index host query failed: {e}")
                if hasattr(e, 'response') and e.response:
                    print(f"Response: {e.response.text}")

                # Try standard API endpoint as fallback
                print("Trying standard API endpoint...")
                response = requests.post(
                    f"{self.base_url}/databases/{self.index_name}/query",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            print(f"Query vectors failed: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Response: {e.response.text}")
            # Return empty results instead of raising an exception
            print("Returning empty results due to query error")
            return {"matches": []}

# === PDF PROCESSING FUNCTIONS ===
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

# === CUSTOM VECTOR STORE SETUP ===
def setup_vector_store(pdf_chunks, pinecone_api):
    """Set up a custom vector store with the provided PDF chunks"""
    try:
        # First, ensure the index exists
        print("Ensuring Pinecone index exists...")
        try:
            # Try to create the index (this will return early if it already exists)
            result = pinecone_api.create_index()
            print(f"Index creation result: {result}")

            # Wait for the index to be ready
            print("Waiting for index to be ready...")
            import time
            time.sleep(20)  # Wait 20 seconds for the index to be ready
        except Exception as e:
            print(f"Warning during index creation: {e}")
            st.warning(f"There was an issue with index creation: {e}")
            st.info("Attempting to proceed anyway...")

        # Clear all existing vectors before uploading new content
        print("Clearing existing vectors from Pinecone index...")
        st.info("🗑️ Clearing previous document data from Pinecone...")
        try:
            pinecone_api.delete_all_vectors()
            print("Successfully cleared all existing vectors")
            st.success("✅ Previous document data cleared successfully!")
            # Wait longer for the deletion to propagate
            time.sleep(10)  # Increased wait time
            print("Waited 10 seconds for deletion to propagate")
        except Exception as e:
            print(f"Warning during vector deletion: {e}")
            st.warning(f"Could not clear existing vectors: {e}")
            st.info("Proceeding with upload anyway - new vectors will be added to existing ones")

        # Initialize the embedding model
        print("Initializing embedding model...")
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

        # Get embeddings for all chunks
        print("Generating embeddings for document chunks...")
        texts = [doc.page_content for doc in pdf_chunks]

        # Prepare metadata with text content included
        metadatas = []
        for i, doc in enumerate(pdf_chunks):
            # Include the text in the metadata to ensure it's stored
            metadata = doc.metadata.copy()
            metadata['text'] = doc.page_content
            metadata['id'] = f"chunk_{i}"  # Add an ID for reference
            metadatas.append(metadata)

        # Generate embeddings in batches to avoid memory issues
        batch_size = 10
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = embedding_model.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
            print(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

        # Upsert vectors to Pinecone in smaller batches
        print("Upserting vectors to Pinecone in batches...")
        upsert_batch_size = 50  # Smaller batch size for upserts

        for i in range(0, len(all_embeddings), upsert_batch_size):
            end_idx = min(i + upsert_batch_size, len(all_embeddings))
            batch_embeddings = all_embeddings[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]

            try:
                print(f"Upserting batch {i//upsert_batch_size + 1}/{(len(all_embeddings)-1)//upsert_batch_size + 1}...")
                pinecone_api.upsert_vectors(batch_embeddings, batch_metadatas)
            except Exception as batch_error:
                print(f"Error upserting batch: {batch_error}")
                st.warning(f"Error upserting batch {i//upsert_batch_size + 1}: {batch_error}")
                # Continue with the next batch

        print(f"Vector store created with {len(pdf_chunks)} chunks")

        # Verify vectors were uploaded by doing a test query
        print("Verifying vectors were uploaded...")
        test_embedding = embedding_model.embed_query("test")
        test_results = pinecone_api.query_vectors(test_embedding, top_k=1)
        if test_results and 'matches' in test_results and test_results['matches']:
            print(f"✅ Verification successful: Found {len(test_results['matches'])} vectors in index")
            st.success(f"✅ Verification: {len(test_results['matches'])} vectors found in Pinecone")
        else:
            print("⚠️ Warning: No vectors found in index after upload")
            st.warning("⚠️ Warning: No vectors found in index after upload")

        # Create a custom retriever function
        def custom_retriever(query):
            # Generate embedding for the query
            query_embedding = embedding_model.embed_query(query)

            # Query Pinecone
            results = pinecone_api.query_vectors(query_embedding, top_k=3)

            # Extract and return the documents
            documents = []
            if 'matches' in results:
                for match in results['matches']:
                    if 'metadata' in match:
                        # Create a simple document-like object
                        doc = type('Document', (), {})()
                        doc.page_content = match['metadata'].get('text', '')
                        doc.metadata = {k: v for k, v in match['metadata'].items() if k != 'text'}
                        documents.append(doc)

            return documents

        return custom_retriever
    except Exception as e:
        print(f"Error setting up vector store: {e}")
        st.error(f"Error details: {e}")
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
            if response and hasattr(response, 'text') and response.text:
                return response.text.strip()
            else:
                print(f"Gemini response was empty or invalid: {response}")
                return "I'm sorry, I received an empty response from the AI."
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            print(f"Error type: {type(e)}")
            print(f"Prompt length: {len(prompt)}")
            print(f"Context length: {len(context)}")
            return f"I'm sorry, I couldn't generate a response. Error: {str(e)[:100]}"

# === CUSTOM QA CHAIN ===
def setup_custom_qa_chain(retriever, model):
    try:
        gemini = CustomGeminiLLM(model)

        def run_qa(question):
            print(f"Processing question: {question}")
            docs = retriever(question)
            print(f"Retrieved {len(docs)} documents")
            if docs:
                print(f"First doc preview: {docs[0].page_content[:100]}...")
            context = "\n\n".join([doc.page_content for doc in docs])
            print(f"Total context length: {len(context)}")
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

    /* Basic button styling */
    .stButton button {
        border-radius: 20px !important;
        padding: 10px 15px !important;
        transition: all 0.3s ease !important;
    }

    /* Removed play-again button */

    /* Clear button */
    .clear-btn {
        text-align: center;
        margin: 10px;
    }
    .clear-btn button {
        background-color: #FF5722 !important;
        color: white !important;
        width: 40px !important;
        height: 40px !important;
        border-radius: 50% !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 16px !important;
        padding: 0 !important;
    }

    /* Send button */
    .send-btn {
        text-align: center;
        margin: 10px;
    }
    .send-btn button {
        background-color: #4CAF50 !important;
        color: white !important;
        width: 40px !important;
        height: 40px !important;
        border-radius: 50% !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 16px !important;
        padding: 0 !important;
    }

    /* Record button */
    .record-btn {
        text-align: center;
        margin: 10px;
    }
    .record-btn button {
        background-color: #f44336 !important;
        color: white !important;
        width: 40px !important;
        height: 40px !important;
        border-radius: 50% !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 16px !important;
        padding: 0 !important;
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
        page_title="💬 Interactive Chat QnA with Direct Pinecone API",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Apply custom CSS
    apply_custom_css()

    # Initialize session state
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'qa_ready' not in st.session_state:
        st.session_state.qa_ready = False
    if 'audio_paths' not in st.session_state:
        st.session_state.audio_paths = {}
    if 'pinecone_initialized' not in st.session_state:
        st.session_state.pinecone_initialized = False
    if 'pinecone_api' not in st.session_state:
        st.session_state.pinecone_api = None
    if 'current_pdf_name' not in st.session_state:
        st.session_state.current_pdf_name = None

    # Sidebar for settings and PDF upload
    with st.sidebar:
        st.title("📋 Document & Settings")

        # Pinecone Connection Status
        st.header("� Pinecone Connection")

        # Check if already connected
        if st.session_state.get('pinecone_initialized', False):
            st.success("✅ Connected to Pinecone successfully!")
            st.info("Using your 'qna' index with 768 dimensions")

            # Option to reconnect if needed
            if st.button("🔄 Reconnect", help="Reconnect to Pinecone if experiencing issues"):
                st.session_state.pinecone_initialized = False
                st.rerun()
        else:
            # Get API key from Streamlit secrets, environment, or user input
            pinecone_api_key = None

            # Try Streamlit secrets first (most secure)
            try:
                pinecone_api_key = st.secrets["pinecone"]["api_key"]
                st.info("🔐 Using API key from Streamlit secrets")
            except (KeyError, FileNotFoundError):
                # Fall back to environment variable
                pinecone_api_key = os.environ.get("PINECONE_API_KEY")
                if pinecone_api_key:
                    st.info("🔐 Using API key from environment variable")

            if not pinecone_api_key:
                st.info("💡 For security, add your API key to .streamlit/secrets.toml or set PINECONE_API_KEY environment variable")
                pinecone_api_key = st.text_input(
                    "Or enter your Pinecone API Key here:",
                    type="password",
                    help="Your API key will not be stored and is only used for this session"
                )

            # Auto-initialize if API key is available
            if pinecone_api_key:
                if st.button("🔗 Connect to Pinecone"):
                    with st.spinner("Connecting to Pinecone..."):
                        try:
                            # Create a new PineconeDirectAPI instance
                            pinecone_api = PineconeDirectAPI(pinecone_api_key)

                            # Test the connection
                            pinecone_api.check_connection()
                            st.success("✅ Connected to Pinecone successfully!")

                            # Store the API instance in session state
                            st.session_state.pinecone_api = pinecone_api
                            st.session_state.pinecone_initialized = True
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Connection failed: {str(e)[:100]}...")
                            st.session_state.pinecone_initialized = False
            else:
                st.warning("⚠️ Pinecone API key required to continue")

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

    # Main content area
    st.title("💬 Interactive Chat with Direct Pinecone API")

    # Auto-initialize Pinecone if not already initialized
    if not st.session_state.pinecone_initialized:
        with st.spinner("Automatically initializing Pinecone..."):
            try:
                # Get API key from Streamlit secrets, environment, or use None
                pinecone_api_key = None

                # Try Streamlit secrets first (most secure)
                try:
                    pinecone_api_key = st.secrets["pinecone"]["api_key"]
                    st.info("🔐 Using Pinecone API key from Streamlit secrets")
                except (KeyError, FileNotFoundError):
                    # Fall back to environment variable
                    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
                    if pinecone_api_key:
                        st.info("🔐 Using Pinecone API key from environment variable")

                if not pinecone_api_key:
                    st.warning("⚠️ No Pinecone API key found. Please configure it in the sidebar.")
                    return

                # Create a new PineconeDirectAPI instance
                pinecone_api = PineconeDirectAPI(pinecone_api_key)

                # Test the connection
                whoami = pinecone_api.check_connection()
                st.success(f"✅ Connected to Pinecone API automatically!")

                # Store the API instance in session state
                st.session_state.pinecone_api = pinecone_api
                st.session_state.pinecone_initialized = True

                # Try to list indexes
                try:
                    indexes = pinecone_api.list_indexes()
                    if indexes:
                        st.success(f"✅ Found {len(indexes)} existing indexes in your Pinecone account.")
                except Exception as e:
                    st.info(f"No existing indexes found. A new one will be created when you upload a document.")
            except Exception as e:
                st.warning(f"⚠️ Could not auto-initialize Pinecone. Please check your API key in the sidebar.")
                st.error(f"Error details: {e}")
                return

    # Check if a new PDF is uploaded (different from current one)
    new_pdf_uploaded = False
    if pdf_file:
        if st.session_state.current_pdf_name != pdf_file.name:
            # New PDF detected - reset the QA system
            st.session_state.current_pdf_name = pdf_file.name
            st.session_state.qa_ready = False
            st.session_state.conversation = []  # Clear conversation history for new document
            st.session_state.audio_paths = {}   # Clear audio paths
            new_pdf_uploaded = True
            st.info(f"🔄 New document detected: {pdf_file.name}")
        elif not st.session_state.qa_ready:
            # Same PDF but QA not ready (probably after refresh)
            st.info(f"🔄 Re-processing document after page refresh: {pdf_file.name}")
            new_pdf_uploaded = True

    # Process PDF when uploaded (new upload or first time)
    if pdf_file and (not st.session_state.qa_ready or new_pdf_uploaded):
        with st.spinner("Processing PDF and storing in Pinecone..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                tmp_pdf.write(pdf_file.read())
                pdf_path = tmp_pdf.name

            # Import Gemini model
            import google.generativeai as genai

            # Get Google API key from secrets or environment
            google_api_key = None
            try:
                google_api_key = st.secrets["google"]["api_key"]
            except (KeyError, FileNotFoundError):
                google_api_key = os.environ.get("GOOGLE_API_KEY")

            if not google_api_key:
                st.error("❌ Google API key not found. Please add it to .streamlit/secrets.toml or set GOOGLE_API_KEY environment variable.")
                return

            genai.configure(api_key=google_api_key)

            model = genai.GenerativeModel("models/gemini-1.5-flash-8b")

            # Process PDF and setup QA chain with Pinecone
            pdf_chunks = load_and_split_pdf(pdf_path)
            if not pdf_chunks:
                st.error("❌ Failed to process PDF. Please try another file.")
                return

            st.info(f"PDF processed into {len(pdf_chunks)} chunks. Storing in Pinecone...")

            # Setup custom vector store with direct Pinecone API
            retriever = setup_vector_store(pdf_chunks, st.session_state.pinecone_api)
            if not retriever:
                st.error("❌ Failed to store document in Pinecone. Please check your API key and try again.")
                return

            # Setup QA chain
            qa_chain = setup_custom_qa_chain(retriever, model)

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

        st.markdown('</div>', unsafe_allow_html=True)

    # Chat input area (only show if PDF is processed)
    if st.session_state.qa_ready:
        # Create columns for the chat input components
        col1, col2, col3 = st.columns([0.8, 0.1, 0.1])

        with col1:
            question = st.text_input("Type your message...", key="text_question", label_visibility="collapsed")

        with col2:
            # Record audio button with custom styling
            st.markdown('<div class="record-btn">', unsafe_allow_html=True)
            try:
                from streamlit_mic_recorder import mic_recorder

                if 'audio_recorder_key' not in st.session_state:
                    st.session_state.audio_recorder_key = 0

                # Track if we've already processed the current recording
                if 'last_processed_recording' not in st.session_state:
                    st.session_state.last_processed_recording = None

                # Use mic_recorder with unique key and error handling
                try:
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
                except Exception as e:
                    st.warning("⚠️ Microphone component not fully loaded. Please refresh the page if recording doesn't work.")
                    st.button("🎤", disabled=True, help="Microphone component loading...")
                    audio_result = None

            except ImportError:
                # Try alternative audio recorder
                try:
                    from streamlit_audio_recorder import audio_recorder
                    st.info("🎤 Using alternative audio recorder")
                    audio_bytes = audio_recorder(
                        text="Click to record",
                        recording_color="#e8b62c",
                        neutral_color="#6aa36f",
                        icon_name="microphone-lines",
                        icon_size="2x",
                    )
                    if audio_bytes:
                        audio_result = {"bytes": audio_bytes}
                    else:
                        audio_result = None
                except ImportError:
                    st.button("🎤", disabled=True, help="Microphone recording not available in this environment")
                    st.info("💡 You can still use text input to ask questions!")
                    audio_result = None
            except Exception as e:
                st.warning(f"Audio recording temporarily unavailable: {str(e)}")
                st.button("🎤", disabled=True, help="Audio recording temporarily unavailable")
                audio_result = None
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            # Send button with custom styling
            st.markdown('<div class="send-btn">', unsafe_allow_html=True)
            send_pressed = st.button("📤", help="Send message")
            st.markdown('</div>', unsafe_allow_html=True)

        # Process text input
        if send_pressed and question:
            process_text_question(question)

        # Process recorded audio
        if audio_result and isinstance(audio_result, dict) and 'bytes' in audio_result:
            process_recorded_audio(audio_result['bytes'])

        # Option to clear conversation
        if st.session_state.conversation:
            col1, col2, col3 = st.columns([0.4, 0.2, 0.4])
            with col2:
                st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
                if st.button("🗑️", help="Clear conversation history"):
                    st.session_state.conversation = []
                    if 'audio_paths' in st.session_state:
                        st.session_state.audio_paths = {}
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
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

# Removed process_audio_file function as it's no longer needed

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