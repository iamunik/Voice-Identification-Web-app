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
6. [Prerequisites](#prerequisites)
7. [Installation](#installation)
8. [Running the Application](#running-the-application)
9. [Usage Guide](#usage-guide)
10. [Project Structure](#project-structure)
11. [Troubleshooting](#troubleshooting)

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

## Prerequisites

Before installing and running the Voice Identification System, ensure you have the following prerequisites:

### System Requirements
- **Operating System**: Windows, macOS, or Linux
- **Python**: Version 3.8 or higher (3.10 recommended)
- **Memory**: At least 4GB RAM (8GB recommended for better performance)
- **Storage**: At least 2GB free disk space for dependencies and models

### Required Software
- **Python 3.8+**: Download from [python.org](https://www.python.org/downloads/)
- **Git**: For cloning the repository (optional if downloading as ZIP)
- **Audio drivers**: Ensure your system can record and play audio

### Hardware Requirements
- **Microphone**: For recording voice samples
- **Speakers/Headphones**: For audio playback (optional)

---

## Installation

Follow these step-by-step instructions to install and set up the Voice Identification System:

### Step 1: Clone or Download the Repository

**Option A: Using Git (Recommended)**
```bash
git clone https://github.com/your-username/Voice-Identification-Web-app.git
cd Voice-Identification-Web-app
```

**Option B: Download ZIP**
1. Download the repository as a ZIP file
2. Extract it to your desired location
3. Navigate to the extracted folder

### Step 2: Create a Virtual Environment (Recommended)

Creating a virtual environment helps isolate project dependencies:

**On Windows:**
```bash
python -m venv voice_id_env
voice_id_env\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv voice_id_env
source voice_id_env/bin/activate
```

### Step 3: Install Dependencies

The project includes a requirements.txt file with all necessary dependencies. Install them using:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: If you encounter issues with the requirements.txt file, install the core dependencies manually:

```bash
pip install streamlit speechbrain torch torchaudio librosa soundfile numpy pandas sqlite3
pip install streamlit-js-eval
```

### Step 4: Verify Installation

Test that the main dependencies are installed correctly:

```python
python -c "import streamlit, speechbrain, torch, torchaudio; print('All dependencies installed successfully!')"
```

---
## Running the Application

### Step 1: Navigate to Project Directory

Ensure you're in the project root directory and your virtual environment is activated:

```bash
cd Voice-Identification-Web-app
# Activate virtual environment if not already active
# Windows: voice_id_env\Scripts\activate
# macOS/Linux: source voice_id_env/bin/activate
```

### Step 2: Initialize the Database

The application will automatically create the SQLite database and tables when you first run it. The database file `Voice_Recognition.db` will be created in the project root.

### Step 3: Start the Streamlit Application

Run the main application using Streamlit:

```bash
streamlit run pagination.py
```

### Step 4: Access the Web Application

After running the command, Streamlit will:
1. Start a local web server (usually on port 8501)
2. Automatically open your default web browser
3. Display the application URL (typically `http://localhost:8501`)

If the browser doesn't open automatically, manually navigate to the displayed URL.

### Alternative Running Methods

**Run with specific port:**
```bash
streamlit run pagination.py --server.port 8502
```

**Run without auto-opening browser:**
```bash
streamlit run pagination.py --server.headless true
```

**Run with CORS disabled (for development):**
```bash
streamlit run pagination.py --server.enableCORS false --server.enableXsrfProtection false
```

---

## Usage Guide

### 1. Homepage
- The application opens to a welcome page with project overview
- Navigate between pages using the sidebar menu

### 2. Registering New Users

**Step 2.1: Navigate to "New Users" Page**
- Click on "New Users" in the sidebar navigation

**Step 2.2: Fill User Information**
- **Personal Details**: Enter first name, other name, last name
- **Date of Birth**: Select using the date picker
- **Contact**: Provide phone number
- **About**: Write a brief description
- **Demographics**: Select gender, employment status, and marital status

**Step 2.3: Upload Profile Picture**
- Click "Browse files" to upload a profile picture
- Supported formats: JPG, JPEG, PNG

**Step 2.4: Record or Upload Voice Sample**
- Choose between "Upload audio" or "Record audio"
- **For Recording**:
  - A random sentence will be displayed
  - Click the microphone icon to record
  - Read the displayed sentence clearly
- **For Upload**:
  - Upload a WAV format audio file
  - Ensure clear voice quality

**Step 2.5: Complete Registration**
- Click "Register User" button
- Wait for processing (includes audio enhancement)
- Success message will confirm registration
- Page will automatically refresh

### 3. User Verification

**Step 3.1: Navigate to "User Verification" Page**
- Click on "User Verification" in the sidebar

**Step 3.2: Voice Verification Process**
- A random sentence will be displayed
- Record your voice saying the displayed sentence
- The system will:
  - Extract voice embeddings
  - Compare against stored user data
  - Display verification results

**Step 3.3: Verification Results**
- **Success**: User details will be displayed including:
  - Full name and age
  - Date of birth
  - Phone number
  - Personal information
- **Failure**: "USER NOT RECOGNISED" message will appear

---

## Project Structure

```
Voice-Identification-Web-app/
│
├── pagination.py              # Main application entry point
├── funcs.py                  # Core functions and database operations
├── voice.py                  # Audio spoof detection utilities
├── requirements.txt          # Python dependencies
├── Voice_Recognition.db      # SQLite database (created automatically)
├── README.md                 # Project documentation
│
├── webpages/                 # Streamlit page components
│   ├── homepage.py          # Welcome/overview page
│   ├── new_users.py         # User registration page
│   └── verify_users.py      # User verification page
│
├── images/                   # Static images
│   └── voice.jpg            # Homepage banner image
│
├── pretrained_models/        # Downloaded AI models (created automatically)
│   └── spkrec-ecapa-voxceleb/
│
└── __pycache__/             # Python cache files
```

### Key Files Description

- **`pagination.py`**: Main entry point that sets up Streamlit navigation
- **`funcs.py`**: Contains all core functions including:
  - Database connection and table creation
  - Audio processing and enhancement
  - Voice embedding extraction and comparison
  - User registration and verification logic
- **`webpages/`**: Contains individual Streamlit pages
- **`Voice_Recognition.db`**: SQLite database storing user data and voice embeddings

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Installation Issues

**Problem**: `pip install` fails with dependency conflicts
**Solution**:
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --no-cache-dir
```

**Problem**: SpeechBrain installation fails
**Solution**:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install speechbrain
```

#### 2. Runtime Issues

**Problem**: "ModuleNotFoundError" when running the application
**Solution**:
- Ensure virtual environment is activated
- Reinstall missing modules: `pip install [module_name]`

**Problem**: Database errors
**Solution**:
- Delete `Voice_Recognition.db` file and restart the application
- The database will be recreated automatically

**Problem**: Audio recording not working
**Solution**:
- Check microphone permissions in your browser
- Ensure microphone is connected and working
- Try using Chrome or Firefox browsers

#### 3. Performance Issues

**Problem**: Slow model loading
**Solution**:
- Models are downloaded on first use (this is normal)
- Subsequent runs will be faster
- Ensure stable internet connection for initial model download

**Problem**: High memory usage
**Solution**:
- Close other applications to free up RAM
- Consider using a machine with more memory for better performance

#### 4. Browser Issues

**Problem**: Application not opening in browser
**Solution**:
- Manually navigate to `http://localhost:8501`
- Try a different browser
- Check if port 8501 is available

**Problem**: Audio features not working in browser
**Solution**:
- Use Chrome, Firefox, or Edge (Safari may have limitations)
- Ensure browser has microphone permissions
- Try refreshing the page

### Getting Help

If you encounter issues not covered here:

1. **Check the terminal/console** for error messages
2. **Verify all dependencies** are installed correctly
3. **Ensure Python version** is 3.8 or higher
4. **Check system requirements** are met
5. **Try running in a fresh virtual environment**

### System-Specific Notes

**Windows Users**:
- Use Command Prompt or PowerShell
- Ensure Windows Defender doesn't block the application

**macOS Users**:
- May need to install Xcode command line tools: `xcode-select --install`
- Grant microphone permissions in System Preferences

**Linux Users**:
- Install system audio libraries: `sudo apt-get install portaudio19-dev python3-pyaudio`
- Ensure microphone permissions are granted

---

**Note**: The first run may take longer as the system downloads pre-trained models (ECAPA-TDNN and MetricGAN+) automatically. These models are cached locally for faster subsequent runs.
