from speechbrain.inference import SpectralMaskEnhancement
import torchaudio
import tempfile
import datetime
import sqlite3
import random
import base64
import os
import io


# Connect to the database
def create_connection():
    conn = sqlite3.connect('Voice_Recognition.db')
    return conn


# Create tables
def create_tables():
    conn = create_connection()
    cursor = conn.cursor()

    # Create users table
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
        picture BLOB NOT NULL)
    ''')

    # Create voice_prints table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS voice_print (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            voice_embedding BLOB NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    conn.commit()
    conn.close()


# Insert a new user
def insert_user(first_name, other_name, last_name, dob, phone, about, sex, occupation, marital_status, picture):
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute('''INSERT INTO user (
        first_name, other_name, last_name, dob, phone, about, sex, occupation, marital_status, picture)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (first_name, other_name, last_name, dob, phone, about, sex, occupation, marital_status, picture))

    conn.commit()
    user_id = cursor.lastrowid  # Get the inserted user's ID
    conn.close()

    return user_id


# Insert voice embedding
def insert_voice_embedding(user_id, voice_embedding):
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO voice_print (user_id, voice_embedding)
        VALUES (?, ?)
    ''', (user_id, voice_embedding))

    conn.commit()
    conn.close()


# Extract audio embeddings and enhance them
def enhance_audio_to_blob(audio_bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_file_path = tmp_file.name

    try:
        model = SpectralMaskEnhancement.from_hparams(
            source="speechbrain/metricgan-plus-voicebank",
            savedir="pretrained_models/metricgan-plus-voicebank"
        )

        enhanced_speech = model.enhance_file(tmp_file_path)

        # Convert to a binary object (blob)
        buffer = io.BytesIO()
        torchaudio.save(buffer, enhanced_speech.view(1, -1), 16000, format="wav", backend="soundfile")
        audio_blob = buffer.getvalue()

    finally:
        # Delete the temporary file
        os.remove(tmp_file_path)

    return audio_blob


# Calculate age from date of birth
def dob_to_age(date_of_birth):
    if type(date_of_birth) is str:
        date_of_birth = datetime.datetime.strptime(date_of_birth, "%Y-%m-%d")
        date_of_birth = date_of_birth.date()
        today = datetime.date.today()
        age_ = today - date_of_birth
        return round(age_.days/365)

    else:
        today = datetime.date.today()
        age_ = today - date_of_birth
        return round(age_.days/365)


# Open picture
def open_picture(image_name):
    cwd = os.path.dirname(__file__)
    image_path = os.path.join(cwd, "images", image_name)
    image_path = os.path.abspath(image_path)
    file = open(image_path, "rb")
    images = base64.b64encode(file.read()).decode()
    return images


# Find the best matching user from dB
def find_best_matching_user(input_audio_blob, recognizer):
    # Extract embedding for the input audio
    input_embedding = enhance_audio_to_blob(input_audio_blob)
    audio_tensor, sample_rate = torchaudio.load(io.BytesIO(input_embedding))

    # Initialize variables to store the best match
    best_user_id = None
    best_score = float("-inf")
    prediction = bool

    # Connect to the database to retrieve all user audio blobs
    conn = create_connection()
    cursor = conn.cursor()

    # Query to retrieve all user_id and audio_blob pairs
    cursor.execute("SELECT user_id, voice_embedding FROM voice_print")
    results = cursor.fetchall()
    conn.close()

    # Loop through each user in the database
    for user_id, stored_audio_blob in results:
        # Extract embedding for the stored user's audio
        stored_embedding = torchaudio.load(io.BytesIO(stored_audio_blob))

        # Use verify_batch() to compare
        score, prediction = recognizer.verify_batch(audio_tensor, stored_embedding[0])

        # Check if this score is the highest so far
        if score > best_score:
            best_score = score
            best_user_id = user_id

    return best_user_id, best_score, prediction


# Show the result of the user extraction from dB
def show_result(user_id):
    connection = create_connection()
    cursor = connection.cursor()
    cursor.execute("""
    SELECT * FROM user as t1 
    LEFT JOIN voice_print as t2 ON t1.user_id = t2.user_id 
    WHERE t1.user_id = ?
    """, (user_id, ))

    execute_rows = cursor.fetchall()
    connection.close()
    return execute_rows


# Randomly generated sentences for audio recording
def test_train_sentences():
    sentences = ["The Greeks used to imagine that it was a sign from the gods to foretell war",
                 "The Norsemen considered the rainbow as a bridge over which the gods passed",
                 "Others have tried to explain the phenomenon physically",
                 "The difference in the rainbow depends considerably upon the size of the drops",
                 "The actual primary rainbow observed is said to be the effect of super-imposition",
                 "The wise men used to believe that fate could change the course of history.",
                 "In the quiet woods, they heard strange sounds that seemed to echo from the past.",
                 "She thought the stars above might hold the answers to the mysteries of life.",
                 "The journey across the seas was seen as a test of strength and endurance.",
                 "Many cultures have stories that speak of heroes who rise in times of need."]
    return random.choice(sentences)
