import streamlit as st
import requests
import time
import os

API_KEY = os.environ.get("DID_API_KEY", "your-d-id-api-key-here")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

TALK_URL = "https://api.d-id.com/talks"

st.title("🧑‍💬 D-ID Talking Avatar")

image_url = st.text_input("Image URL", "https://images.pexels.com/photos/614810/pexels-photo-614810.jpeg")
text_input = st.text_area("What should the avatar say?", "Hello! I'm your AI avatar.")

if st.button("Generate Avatar Video"):
    with st.spinner("Creating avatar..."):
        payload = {
            "source_url": image_url,
            "script": {
                "type": "text",
                "input": text_input,
                "provider": {
                    "type": "microsoft",
                    "voice_id": "en-US-JennyNeural"
                }
            }
        }

        try:
            res = requests.post(TALK_URL, headers=HEADERS, json=payload)
            res.raise_for_status()
            talk_id = res.json()["id"]

            # Poll for result_url
            for _ in range(15):
                time.sleep(2)
                status_res = requests.get(f"{TALK_URL}/{talk_id}", headers=HEADERS)
                status_res.raise_for_status()
                result = status_res.json()
                if result.get("result_url"):
                    st.video(result["result_url"])
                    break
            else:
                st.error("Video generation timed out. Try again.")
        except Exception as e:
            st.error(f"Error: {e}")
