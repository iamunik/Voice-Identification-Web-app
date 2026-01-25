import numpy as np
import librosa
import torchaudio
import tempfile
import datetime
import sqlite3
import random
import base64
import os
import io
import logging
import torch
from scipy import signal
import streamlit as st

try:
    from tensorflow.keras.models import load_model

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# SPOOF DETECTION CONFIGURATION
# ============================================================================

SPOOF_MODEL = None
SPOOF_THRESHOLD = 0.5
MFCC_PARAMS = {
    'sr': 16000,
    'n_mfcc': 40,
    'max_pad_len': 200
}


def load_spoof_model(model_path):
    """
    Load the trained LSTM spoof detection model once at startup.

    Args:
        model_path (str): Path to the saved Keras model (.keras file)

    Returns:
        bool: True if model loaded successfully
    """
    global SPOOF_MODEL

    if SPOOF_MODEL is not None:
        logger.info("Spoof model already loaded, using cached version")
        return True

    if not TENSORFLOW_AVAILABLE:
        logger.warning("TensorFlow not available, spoof detection will be disabled")
        return False

    try:
        if not os.path.exists(model_path):
            logger.error(f"Spoof model file not found at {model_path}")
            return False

        SPOOF_MODEL = load_model(model_path)
        logger.info(f"✅ Spoof detection model loaded successfully from {model_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to load spoof detection model: {str(e)}")
        return False


def set_spoof_threshold(threshold):
    """Set the spoof detection threshold (0-1). Higher = stricter."""
    global SPOOF_THRESHOLD
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    SPOOF_THRESHOLD = threshold
    logger.info(f"Spoof detection threshold set to {threshold}")


def extract_mfcc_features(audio_bytes):
    """
    Extract MFCC features from audio bytes.
    Matches the training preprocessing exactly.

    Args:
        audio_bytes (bytes): Raw audio data

    Returns:
        np.ndarray: MFCC features with shape (n_mfcc, max_pad_len) or None if failed
    """
    sr = MFCC_PARAMS['sr']
    n_mfcc = MFCC_PARAMS['n_mfcc']
    max_pad_len = MFCC_PARAMS['max_pad_len']

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name

        try:
            # Load audio
            y, _ = librosa.load(tmp_file_path, sr=sr)

            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

            # Pad or truncate to consistent length
            if mfccs.shape[1] > max_pad_len:
                mfccs = mfccs[:, :max_pad_len]
            else:
                pad_width = max_pad_len - mfccs.shape[1]
                mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

            return mfccs

        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

    except Exception as e:
        logger.error(f"MFCC extraction failed: {str(e)}")
        return None


def detect_spoof(audio_bytes):
    """
    Detect if audio is spoofed or genuine using trained LSTM model.

    Args:
        audio_bytes (bytes): Raw audio data

    Returns:
        dict: {
            'is_spoof': bool,
            'confidence': float (0-1),
            'label': str ('genuine' or 'spoof')
        }
        or None if detection is unavailable/fails
    """
    global SPOOF_MODEL, SPOOF_THRESHOLD

    if SPOOF_MODEL is None:
        logger.warning("⚠️ Spoof model not loaded, skipping spoof detection")
        return None

    try:
        # Extract MFCC features
        mfccs = extract_mfcc_features(audio_bytes)

        if mfccs is None:
            logger.error("Failed to extract MFCC features for spoof detection")
            return None

        # Add batch dimension: (40, 200) → (1, 40, 200)
        features = np.expand_dims(mfccs, axis=0)

        # Run inference
        prediction = SPOOF_MODEL.predict(features, verbose=0)[0][0]

        # Determine if spoof
        is_spoof = prediction > SPOOF_THRESHOLD

        result = {
            'is_spoof': bool(is_spoof),
            'confidence': float(prediction),
            'label': 'spoof' if is_spoof else 'genuine'
        }

        logger.info(f"Spoof detection result: {result['label']} (confidence: {result['confidence']:.4f})")
        return result

    except Exception as e:
        logger.error(f"Spoof detection inference failed: {str(e)}")
        return None


# ============================================================================
# AUDIO ENHANCEMENT
# ============================================================================

@st.cache_resource
def load_enhancement_model():
    """Load the enhancement model once and cache it."""
    try:
        from speechbrain.inference.enhancement import SpectralMaskEnhancement

        logger.info("Loading SpectralMaskEnhancement model...")
        model = SpectralMaskEnhancement.from_hparams(
            source="speechbrain/metricgan-plus-voicebank",
            savedir="pretrained_models/metricgan-plus-voicebank",
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )
        logger.info("✅ Enhancement model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load enhancement model: {e}")
        return None


def simple_noise_reduction(audio, sr, noise_duration=1.0):
    """Fallback: Simple noise reduction using spectral gating."""
    noise_sample_count = int(sr * noise_duration)
    noise_profile = np.mean(np.abs(audio[:noise_sample_count]))

    threshold = noise_profile * 1.5
    mask = np.abs(audio) > threshold
    smoothed_mask = signal.medfilt(mask.astype(float), kernel_size=5)
    reduced_audio = audio * smoothed_mask

    return reduced_audio


def enhance_audio_to_blob(audio_bytes):
    """
    Enhance audio by removing noise using MetricGAN+ model.
    Falls back to simple noise reduction if model loading fails.

    Args:
        audio_bytes (bytes): Raw audio data from streamlit

    Returns:
        bytes: Enhanced audio as WAV bytes
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_file_path = tmp_file.name

    try:
        # Try to load and use the enhancement model
        model = load_enhancement_model()

        if model is not None:
            try:
                logger.info("Enhancing audio with MetricGAN+...")
                with torch.no_grad():
                    enhanced_speech = model.enhance_file(tmp_file_path)

                # Convert to bytes
                buffer = io.BytesIO()
                torchaudio.save(
                    buffer,
                    enhanced_speech.view(1, -1),
                    16000,
                    format="wav"
                )
                audio_blob = buffer.getvalue()
                logger.info("✅ Audio enhanced successfully with MetricGAN+")
                return audio_blob

            except Exception as e:
                logger.warning(f"MetricGAN+ enhancement failed: {e}. Using fallback method.")
                # Fallback to simple noise reduction
                audio, sr = torchaudio.load(tmp_file_path)
                audio_np = audio.numpy().flatten()
                reduced_audio = simple_noise_reduction(audio_np, sr)

                buffer = io.BytesIO()
                reduced_tensor = torch.FloatTensor(reduced_audio).unsqueeze(0)
                torchaudio.save(buffer, reduced_tensor, sr, format="wav")
                logger.info("✅ Audio enhanced with fallback noise reduction")
                return buffer.getvalue()
        else:
            logger.info("Enhancement model not available. Using fallback noise reduction.")
            audio, sr = torchaudio.load(tmp_file_path)
            audio_np = audio.numpy().flatten()
            reduced_audio = simple_noise_reduction(audio_np, sr)

            buffer = io.BytesIO()
            reduced_tensor = torch.FloatTensor(reduced_audio).unsqueeze(0)
            torchaudio.save(buffer, reduced_tensor, sr, format="wav")
            return buffer.getvalue()

    except Exception as e:
        logger.error(f"Audio enhancement error: {e}. Returning original audio.")
        audio, sr = torchaudio.load(tmp_file_path)
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio, sr, format="wav")
        return buffer.getvalue()

    finally:
        if os.path.exists(tmp_file_path):
            try:
                os.remove(tmp_file_path)
            except:
                pass


# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def create_connection():
    """Create and return a connection to the database."""
    conn = sqlite3.connect('Voice_Recognition.db')
    return conn


def create_tables():
    """Create database tables if they don't exist."""
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            other_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            dob DATE NOT NULL,
            phone TEXT NOT NULL,
            about LONGTEXT NOT NULL,
            sex TEXT NOT NULL,
            occupation TEXT NOT NULL,
            marital_status TEXT NOT NULL,
            picture BLOB NOT NULL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS voice_print (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            voice_embedding BLOB NOT NULL,
            FOREIGN KEY (user_id) REFERENCES user(user_id)
        )
    ''')

    conn.commit()
    conn.close()


def insert_user(first_name, other_name, last_name, dob, phone, about, sex, occupation, marital_status, picture):
    """Insert a new user into the database."""
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO user (first_name, other_name, last_name, dob, phone, about, sex, occupation, marital_status, picture)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (first_name, other_name, last_name, dob, phone, about, sex, occupation, marital_status, picture))

    conn.commit()
    user_id = cursor.lastrowid
    conn.close()

    logger.info(f"User {user_id} inserted: {first_name} {last_name}")
    return user_id


def insert_voice_embedding(user_id, voice_embedding):
    """Insert voice embedding for a user."""
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO voice_print (user_id, voice_embedding)
        VALUES (?, ?)
    ''', (user_id, voice_embedding))

    conn.commit()
    conn.close()
    logger.info(f"Voice embedding inserted for user {user_id}")


# ============================================================================
# VOICE RECOGNITION & VERIFICATION
# ============================================================================

def find_best_matching_user(input_audio_blob, recognizer):
    """
    Find the best matching user from the database.

    FLOW:
    1. Check for spoof attacks (rejects if detected)
    2. Enhance audio
    3. Compare against all users in DB
    4. Return best match

    Returns:
        tuple: (best_user_id, best_score, prediction)
               Returns (None, 0.0, False) if spoof detected or error
    """

    # STEP 1: Spoof Detection Gate
    spoof_result = detect_spoof(input_audio_blob)

    if spoof_result is not None:
        if spoof_result['is_spoof']:
            logger.warning(f"🚨 SPOOF ATTACK DETECTED! Confidence: {spoof_result['confidence']:.4f}")
            return None, 0.0, False
        else:
            logger.info(f"✅ Audio verified as genuine. Confidence: {spoof_result['confidence']:.4f}")
    else:
        logger.warning("⚠️ Spoof detection unavailable, proceeding without spoof check")

    # STEP 2: Audio passed spoof check, proceed with voice verification
    try:
        input_embedding = enhance_audio_to_blob(input_audio_blob)
        audio_tensor, _ = torchaudio.load(io.BytesIO(input_embedding))

        best_user_id = None
        best_score = float("-inf")
        prediction = False

        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, voice_embedding FROM voice_print")
        results = cursor.fetchall()
        conn.close()

        for user_id, stored_audio_blob in results:
            stored_embedding, _ = torchaudio.load(io.BytesIO(stored_audio_blob))
            score, pred = recognizer.verify_batch(audio_tensor, stored_embedding)

            if score > best_score:
                best_score = score
                best_user_id = user_id
                prediction = pred

        logger.info(f"Voice match: User {best_user_id}, Score: {best_score:.4f}")
        return best_user_id, best_score, prediction

    except Exception as e:
        logger.error(f"Error during voice matching: {str(e)}")
        return None, 0.0, False


def show_result(user_id):
    """Retrieve user details from database."""
    connection = create_connection()
    cursor = connection.cursor()
    cursor.execute("""
        SELECT * FROM user as t1 
        LEFT JOIN voice_print as t2 ON t1.user_id = t2.user_id 
        WHERE t1.user_id = ?
    """, (user_id,))

    execute_rows = cursor.fetchall()
    connection.close()
    return execute_rows


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def dob_to_age(date_of_birth):
    """Calculate age from date of birth."""
    if isinstance(date_of_birth, str):
        date_of_birth = datetime.datetime.strptime(date_of_birth, "%Y-%m-%d").date()

    today = datetime.date.today()
    age = (today - date_of_birth).days // 365
    return age


def open_picture(image_name):
    """Load and encode image as base64."""
    cwd = os.path.dirname(__file__)
    image_path = os.path.join(cwd, "images", image_name)
    image_path = os.path.abspath(image_path)
    try:
        with open(image_path, "rb") as file:
            images = base64.b64encode(file.read()).decode()
        return images
    except Exception as e:
        logger.error(f"Error opening picture: {str(e)}")
        return None


def test_train_sentences():
    """Return a random sentence for voice recording."""
    sentences = [
        "The Greeks used to imagine that it was a sign from the gods to foretell war",
        "The Norsemen considered the rainbow as a bridge over which the gods passed",
        "Others have tried to explain the phenomenon physically",
        "The difference in the rainbow depends considerably upon the size of the drops",
        "The actual primary rainbow observed is said to be the effect of super-imposition",
        "The wise men used to believe that fate could change the course of history.",
        "In the quiet woods, they heard strange sounds that seemed to echo from the past.",
        "She thought the stars above might hold the answers to the mysteries of life.",
        "The journey across the seas was seen as a test of strength and endurance.",
        "Many cultures have stories that speak of heroes who rise in times of need."
    ]
    return random.choice(sentences)