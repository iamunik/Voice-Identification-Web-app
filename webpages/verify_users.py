from funcs import test_train_sentences, dob_to_age, find_best_matching_user, show_result, load_spoof_model
from speechbrain.inference.speaker import SpeakerRecognition
import streamlit as st
import base64
import time
import os

# Page configuration
st.set_page_config(
    page_title="Verify users",
    layout='centered',
    page_icon='✅',
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


# Initialize SpeechBrain's speaker recognition model
@st.cache_resource
def load_recognizer():
    """Load the speaker recognition model once and cache it."""
    recognizer = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )
    return recognizer


recognizer = load_recognizer()

st.title("Verification Page")
st.markdown("**Please record the following text displayed below:**")

# Show spoof detection status
if spoof_model_loaded:
    st.info("✅ Spoof detection is enabled")
else:
    st.warning("⚠️ Spoof detection is disabled (model not found)")

st.success(f"{test_train_sentences()}")
audio_file = st.audio_input("Record or upload your audio")

if audio_file:
    with st.spinner("Processing audio..."):
        time.sleep(1)

    # Find the best matching user in the database (includes spoof detection in the flow)
    bestUser_id, bestScore, predict = find_best_matching_user(audio_file.getvalue(), recognizer)

    if bestUser_id is None and bestScore == 0.0 and not predict:
        # Check if it was a spoof detection failure or no match
        st.error("❌ VERIFICATION FAILED")
        st.error("This audio was detected as spoofed or could not be verified.")

    elif bestUser_id:
        with st.spinner("Checking Database"):
            time.sleep(2)

        if not predict:
            st.error("❌ USER NOT RECOGNISED!!!")
            st.info("Your voice does not match any registered user in our database.")
        else:
            with st.spinner("Verifying User"):
                time.sleep(2)

            rows = show_result(bestUser_id)
            st.success("✅ USER FOUND AND VERIFIED!")
            st.header("USER INFORMATION")

            col1, col2 = st.columns(2)

            for row in rows:
                fullname = f"<p>Fullname:<br><b>{row[3]} {row[1]} {row[2]}</b></p>"
                age = f"<p>Age:<br><b>{dob_to_age(row[4])} years</b></p>"
                dob = f"<p>Date of birth:<br><b>{row[4]}</b></p>"
                phone = f"<p>Phone number:<br><b>{row[5]}</b></p>"
                about = f"<p>About user:<br><b>{row[6]}</b></p>"
                sex = f"<p>Gender:<br><b>{row[7]}</b></p>"
                occupation = f"<p>Occupation:<br><b>{row[8]}</b></p>"
                marital_status = f"<p>Marital Status:<br><b>{row[9]}</b></p>"

                with col1:
                    if row[10]:  # Check if image exists
                        images = base64.b64encode(row[10]).decode()
                        st.markdown(f'<img style="border: 2px solid powderblue" src="data:image/jpeg;base64,{images}" '
                                    f'width="80%" height="30%">',
                                    unsafe_allow_html=True)
                    st.html(sex)
                    st.html(occupation)

                with col2:
                    st.html(fullname)
                    st.html(dob)
                    st.html(age)
                    st.html(phone)
                    st.html(about)
                    st.html(marital_status)

            st.divider()
            st.metric("Voice Match Score", f"{bestScore:.4f}", delta="Higher is better")

    else:
        st.error("❌ Couldn't connect to database, please try again.")

else:
    st.warning("🎤 Record your audio for verification")