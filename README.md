# Voice Identification System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) 
![SpeechBrain](https://img.shields.io/badge/SpeechBrain-v0.5%2B-green) 
![Streamlit](https://img.shields.io/badge/Streamlit-v1.12%2B-orange)

This project is a **Voice Identification System** built using Python, leveraging **SpeechBrain** and **ECAPA-TDNN** for speaker verification. The system identifies users by comparing their voice embeddings with stored data, providing a secure and efficient method for user recognition.

## Table of Contents
1. [Overview](#overview)
2. [Project Goals](#project-goals)
3. [Key Technologies](#key-technologies)
4. [Project Workflow](#project-workflow)
5. [Implementation Details](#implementation-details)

---

## Overview

The **Voice Identification System** is designed to recognize users based on their voice data. It uses state-of-the-art technologies like **SpeechBrain** and **ECAPA-TDNN** to extract robust speaker embeddings and perform accurate speaker verification. The system also features a **Streamlit-based web app** for seamless interaction, allowing users to upload audio files, verify their identity, and view results in real-time.

---

## Project Goals

The main goals of this project are:
- **Accurate Speaker Recognition**: Build a system that accurately identifies users based on their voice data.
- **Efficient Storage and Retrieval**: Store and retrieve speaker embeddings efficiently for quick verification.
- **User-Friendly Interface**: Create an intuitive web application using **Streamlit** for easy interaction.

---

## Key Technologies

### Python
Python serves as the core language for this project, offering flexibility and a rich ecosystem of libraries for machine learning, audio processing, and web development.

### SpeechBrain
[SpeechBrain](https://speechbrain.github.io/) is an open-source toolkit for speech processing. This project leverages SpeechBrain for feature extraction and speaker verification tasks.

### ECAPA-TDNN
[ECAPA-TDNN](https://arxiv.org/abs/2005.07143) (Emphasized Channel Attention, Propagation, and Aggregation Time-Delay Neural Network) is a state-of-the-art neural network architecture for speaker recognition. It generates robust and discriminative speaker embeddings, which are used as unique voiceprints for each user.

### Streamlit
[Streamlit](https://streamlit.io/) creates a user-friendly web interface. It allows users to upload audio files, view verification results, and navigate through different sections of the app seamlessly.

---

## Project Workflow

1. **Audio Upload**: Users upload or record audio directly in the Streamlit app.
2. **Embedding Extraction**: SpeechBrain, powered by the ECAPA-TDNN model, extracts speaker embeddings from the uploaded audio file.
3. **Database Matching**:
   - The extracted embeddings are compared against stored user embeddings in the database.
   - The system retrieves and displays the user’s profile details if a match is found.
4. **Verification Results**: Verification results and user details are displayed on a dedicated page.

---

## Implementation Details

### Audio Processing
The system processes audio files uploaded through the Streamlit interface. It normalizes and prepares the audio for embedding extraction using SpeechBrain.

### Speaker Embeddings
The core of the verification process lies in generating reliable **speaker embeddings** using the ECAPA-TDNN model. These embeddings serve as unique voiceprints for each user, enabling accurate speaker verification.

### Database Storage and Retrieval
- Extracted embeddings are stored in a **SQLite database** as binary blobs.
- During verification, embeddings are retrieved and compared to new input embeddings using cosine similarity to confirm the user’s identity.

---
