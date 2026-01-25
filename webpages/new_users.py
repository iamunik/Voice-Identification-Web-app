from funcs import (create_tables, enhance_audio_to_blob, insert_voice_embedding,
                   insert_user, test_train_sentences, detect_spoof, load_spoof_model)
from streamlit_js_eval import streamlit_js_eval
import streamlit as st
import datetime
import time
import os

st.set_page_config(
    page_title='New users',
    layout='centered',
    page_icon='🆕',
    initial_sidebar_state="auto"
)


# Initialize spoof model if available
@st.cache_resource
def init_spoof_detector():
    """Initialize spoof detection model if it exists."""
    # Navigate up one level from webpages/ to parent directory
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))  # webpages/
    parent_dir = os.path.dirname(current_dir)  # parent/
    spoof_model_path = os.path.join(parent_dir, "lstm_spoof_detector_model.keras")

    if os.path.exists(spoof_model_path):
        success = load_spoof_model(spoof_model_path)
        return success
    return False


spoof_model_loaded = init_spoof_detector()

# Initialize the database and tables
create_tables()

# Streamlit App Interface
st.title("Voice identification Portal")
st.subheader("Register a New User")

# Show spoof detection status
if spoof_model_loaded:
    st.info("✅ Spoof detection is enabled")
else:
    st.warning("⚠️ Spoof detection is disabled (model not found)")

col1, col2 = st.columns(2)

# collect input for new registration
with col1:
    first_name = st.text_input("First name")
    other_name = st.text_input("other name")
    last_name = st.text_input("last name")
    dob = st.date_input("Your date of birth", value=None,
                        min_value=datetime.date(year=1960, month=1, day=1),
                        max_value=datetime.date.today())
    phone = st.text_input("Phone number")

with col2:
    about = st.text_area("Tell me about yourself")
    sex = st.radio("Gender", ['Male', 'Female'])
    occupation = st.radio("Employment Status", ['Student', 'Self-employed', 'Freelance'])
    marital_status = st.radio("Marital status", ['Single', 'Married', 'Taken'])

picture_file = st.file_uploader("Upload a picture:", type=["jpg", "jpeg", "png"])

voice = st.radio("How would you like to process your audio", ["Upload audio", "Record audio"])
voice_file = None
audio_is_genuine = None

if voice == "Upload audio":
    voice_file_upload = st.file_uploader("Upload a voice recording (WAV format):", type=["wav"])
    if voice_file_upload:
        with st.spinner("Checking audio authenticity..."):
            spoof_result = detect_spoof(voice_file_upload.getvalue())

        if spoof_result is not None:
            if spoof_result['is_spoof']:
                st.error(f"❌ SPOOFED AUDIO DETECTED - Confidence: {spoof_result['confidence']:.2%}")
                st.error("Audio appears to be synthetic, replayed, or voice-converted. Please provide genuine audio.")
                audio_is_genuine = False
            else:
                st.success(f"✅ Audio verified as genuine - Confidence: {spoof_result['confidence']:.2%}")
                audio_is_genuine = True

                with st.spinner("Enhancing audio..."):
                    voice_file = enhance_audio_to_blob(voice_file_upload.getvalue())
                st.success("Audio enhanced successfully!")
        else:
            # Spoof model not loaded, proceed without check
            st.warning("⚠️ Spoof detection unavailable, proceeding without authenticity check")
            with st.spinner("Enhancing audio..."):
                voice_file = enhance_audio_to_blob(voice_file_upload.getvalue())
            st.success("Audio enhanced successfully!")
            audio_is_genuine = True

else:
    st.divider()
    st.markdown("**Please record the following text displayed below:**")
    st.success(f"{test_train_sentences()}")
    voice_file_recorded = st.audio_input('Record the text displayed above')
    if voice_file_recorded:
        with st.spinner("Checking audio authenticity..."):
            spoof_result = detect_spoof(voice_file_recorded.getvalue())

        if spoof_result is not None:
            if spoof_result['is_spoof']:
                st.error(f"❌ SPOOFED AUDIO DETECTED - Confidence: {spoof_result['confidence']:.2%}")
                st.error("Audio appears to be synthetic, replayed, or voice-converted. Please record again.")
                audio_is_genuine = False
            else:
                st.success(f"✅ Audio verified as genuine - Confidence: {spoof_result['confidence']:.2%}")
                audio_is_genuine = True

                with st.spinner("Enhancing audio..."):
                    voice_file = enhance_audio_to_blob(voice_file_recorded.getvalue())
                st.success("Voice capture and enhancement complete!")
        else:
            # Spoof model not loaded, proceed without check
            st.warning("⚠️ Spoof detection unavailable, proceeding without authenticity check")
            with st.spinner("Enhancing audio..."):
                voice_file = enhance_audio_to_blob(voice_file_recorded.getvalue())
            st.success("Voice capture and enhancement complete!")
            audio_is_genuine = True

submitted = st.button("Register User")

try:
    if submitted:
        # Check if audio is genuine (only if spoof detection is available)
        if spoof_model_loaded and audio_is_genuine is False:
            st.error("Cannot register with spoofed audio. Please provide genuine audio.")

        # Check all required fields
        elif all((first_name, other_name, last_name, dob, phone, about, sex, occupation,
                  marital_status, picture_file, voice_file)):

            # Save the uploaded picture as BLOB
            picture_data = picture_file.read()

            # Insert into user_db
            user_id = insert_user(first_name, other_name, last_name, dob, phone, about, sex, occupation,
                                  marital_status, picture_data)

            # Insert into voice_db
            insert_voice_embedding(user_id, voice_file)

            with st.spinner("Registering..."):
                time.sleep(2)

            st.success(f"✅ User {first_name} {last_name} registered successfully!")
            st.balloons()

            # Refresh the page after user registers
            time.sleep(2)
            streamlit_js_eval(js_expressions="parent.window.location.reload()")
        else:
            st.error("Please fill in all fields and upload both picture and voice.")
except Exception as e:
    st.error(f"Registration error: {str(e)}")