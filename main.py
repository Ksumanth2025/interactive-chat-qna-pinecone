# main.py
# Audio QnA system using Gemini 1.5 Flash, Vosk, ChromaDB, and gTTS

import os
import json
import librosa
import soundfile as sf
from gtts import gTTS
from vosk import Model, KaldiRecognizer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import google.generativeai as genai

# === CONFIGURATION ===

# Set Gemini API key from environment variable
google_api_key = os.environ.get("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("Please set GOOGLE_API_KEY environment variable")
genai.configure(api_key=google_api_key)

# Path to local Vosk model
vosk_model_path = "D:/p2/vosk-model-small-en-us-0.15"
if not os.path.exists(vosk_model_path):
    raise Exception(f"Vosk model not found at {vosk_model_path}. Please download and extract it.")
vosk_model = Model(vosk_model_path)

# === AUDIO TRANSCRIPTION ===

def transcribe_audio(audio_path):
    try:
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        tmp_wav = "temp.wav"
        sf.write(tmp_wav, audio, 16000)

        rec = KaldiRecognizer(vosk_model, 16000)
        with open(tmp_wav, "rb") as f:
            data = f.read()
            rec.AcceptWaveform(data)
            result = rec.Result()
        os.remove(tmp_wav)

        text = json.loads(result).get('text', '')
        return text
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return ""

# === PDF CHUNKING ===

def load_and_split_pdf(pdf_path, chunk_size=1000, chunk_overlap=200):

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

# === VECTOR STORE SETUP ===

def setup_vector_store(pdf_chunks, persist_directory="./chroma_db"):
    try:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

        vector_store = Chroma.from_documents(
            documents=pdf_chunks,
            embedding=embedding_model,
            persist_directory=persist_directory
        )
        return vector_store
    except Exception as e:
        print(f"Error setting up vector store: {e}")
        return None

# === GEMINI INTEGRATION ===

class GeminiLLM:
    def __init__(self):
        self.model = genai.GenerativeModel("models/gemini-1.5-flash-8b")

    def ask(self, question, context):
        prompt = f"""You are an expert Python tutor.

Use the knowledge base context below to answer the question. If the context is incomplete or incorrect, use your own knowledge to improve the answer. If no context is found, still answer fully using your own knowledge.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

        try:
            response = self.model.generate_content(prompt)  # Plain string prompt
            return response.text.strip()
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return "I'm sorry, I couldn't generate a response."

# === QA CHAIN ===

def setup_qa_chain(vector_store):
    try:
        gemini = GeminiLLM()
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        def run_qa(question):
            docs = retriever.invoke(question)  # Updated for LangChain 0.1.46+
            context = "\n\n".join([doc.page_content for doc in docs])
            print("Retrieved context:\n", context)

            return gemini.ask(question, context)

        return run_qa
    except Exception as e:
        print(f"Error setting up Gemini QA: {e}")
        return None

# === TEXT TO SPEECH ===

def text_to_speech(text, output_path):
    try:
        tts = gTTS(text)
        tts.save(output_path)
        print(f"Audio saved to {output_path}")
    except Exception as e:
        print(f"Error generating audio: {e}")

# === MAIN WORKFLOW ===

def main(audio_input, pdf_path, audio_output):
    try:
        pdf_chunks = load_and_split_pdf(pdf_path)
        if not pdf_chunks:
            print("No PDF chunks loaded. Exiting.")
            return

        vector_store = setup_vector_store(pdf_chunks)
        if not vector_store:
            print("Failed to set up vector store. Exiting.")
            return

        question = transcribe_audio(audio_input)
        if not question:
            print("No question transcribed. Exiting.")
            return
        print(f"Question: {question}")

        qa_chain = setup_qa_chain(vector_store)
        if not qa_chain:
            print("Failed to set up QA chain. Exiting.")
            return

        answer = qa_chain(question)
        print(f"Answer: {answer}")

        text_to_speech(answer, audio_output)

    except Exception as e:
        print(f"Error in main workflow: {e}")

# === RUN ===

if __name__ == "__main__":
    main("D:/p2/question2.wav", "D:/p2/python.pdf", "D:/p2/answer.mp3")
