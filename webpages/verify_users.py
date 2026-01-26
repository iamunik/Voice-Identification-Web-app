from funcs import test_train_sentences, dob_to_age, find_best_matching_user, show_result
from speechbrain.inference.speaker import SpeakerRecognition
import streamlit as st
import base64
import time

# Page configuration
st.set_page_config(
    page_title="Verify users",
    layout='centered',
    page_icon='✅',
    initial_sidebar_state="auto"
)

# Initialize SpeechBrain's speaker recognition model
recognizer = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )

st.title("Verification Page")
st.markdown("**Please record the following text displayed below:**")


st.success(f"{test_train_sentences()}")
audio_file = st.experimental_audio_input("Record or upload your audio")

if audio_file:
    # Find the best matching user in the database
    bestUser_id, bestScore, predict = find_best_matching_user(audio_file.getvalue(), recognizer)
    if bestUser_id:
        with st.spinner("Checking Database"):       # Aesthetics
            time.sleep(2)
        if not predict:
            st.error("USER NOT RECOGNISED!!!")
        else:
            with st.spinner("Verifying User"):      # Aesthetics
                time.sleep(2)
            rows = show_result(bestUser_id)
            st.header("USER FOUND\n")
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
    else:
        st.error("Couldn't connect to database, please try again.")
else:
    st.warning("Record your audio for auto verification")
