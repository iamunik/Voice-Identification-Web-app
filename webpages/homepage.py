from funcs import open_picture
import streamlit as st


st.set_page_config(
    page_title='G.L.K',
    layout='centered',
    page_icon='🌐',
    initial_sidebar_state="auto"
)

st.title("Voice Identification Web App")
st.markdown(f"""
<img style="border: 2px solid powderblue" src="data:image/jpeg;base64,{open_picture("voice.jpg")}" width="80%"><br>
""", unsafe_allow_html=True)

st.markdown("""
## Overview

This project is a **Voice Identification System** built using Python, leveraging **SpeechBrain** and **ECAPA-TDNN** for 
speaker verification. The system identifies users by comparing their voice embeddings with stored data, providing a 
secure and efficient method for user recognition.

## Project Goals

The main goals of the project are:
- To build a system that accurately recognizes users based on voice data.
- To store and retrieve user embeddings for speaker verification.
- To create a streamlined user experience with a web app built using **Streamlit**.

## Key Technologies

### Python
The core language for developing this voice identification system, Python, offers flexibility with libraries and tools
for machine learning and audio processing.

### SpeechBrain
[SpeechBrain](https://speechbrain.github.io/) is an open-source toolkit for speech processing. This project utilizes 
SpeechBrain for feature extraction and speaker verification. 

### ECAPA-TDNN
[ECAPA-TDNN](https://arxiv.org/abs/2005.07143) (Emphasized Channel Attention, Propagation and Aggregation Time-Delay
Neural Network) is a state-of-the-art neural network architecture for speaker recognition tasks. It provides robust 
and accurate speaker embeddings used in the verification process.

### Streamlit
[Streamlit](https://streamlit.io/) is used to create a user-friendly web interface for the project. It allows users to 
upload audio files, view verification results, and navigate through different sections of the app seamlessly.

## Project Workflow

1. **Audio Upload**: Users upload or record audio directly in the Streamlit app.
2. **Embedding Extraction**: SpeechBrain, using the ECAPA-TDNN model, extracts audio embeddings from the uploaded file.
3. **Database Matching**:
   - The embeddings are compared against stored user embeddings.
   - If a match is found, the system retrieves and displays the user’s profile details.
4. **Verification Results**: Verification results and details are displayed on a separate page for each identified user.

## Implementation Details

### Audio Processing
The system processes audio uploaded through Streamlit, normalizing and preparing it for embedding extraction using 
SpeechBrain.

### Speaker Embeddings
The core of the verification process lies in generating reliable **speaker embeddings** using the ECAPA-TDNN model. 
These embeddings serve as the unique voiceprint for each user.

### Database Storage and Retrieval
The extracted embeddings are stored in a SQLite database as binary blobs. Upon verification, embeddings are retrieved 
and compared to new input embeddings to confirm the user’s identity.
""", unsafe_allow_html=True)
