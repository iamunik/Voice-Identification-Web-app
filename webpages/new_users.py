from funcs import (create_tables, enhance_audio_to_blob, insert_voice_embedding,
                   insert_user, test_train_sentences)
from streamlit_js_eval import streamlit_js_eval
import streamlit as st
import datetime
import time

st.set_page_config(
    page_title='New users',
    layout='centered',
    page_icon='🆕',
    initial_sidebar_state="auto"
)

# Initialize the database and tables
create_tables()

# Streamlit App Interface
st.title("Voice identification Portal")

# Upload Picture and Voice
st.subheader("Register a New User")


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
if voice == "Upload audio":
    voice_file = st.file_uploader("Upload a voice recording (WAV format):", type=["wav"])
    if voice_file:
        voice_file = enhance_audio_to_blob(voice_file.getvalue())
    else:
        pass

else:
    st.divider()
    st.markdown("**Please record the following text displayed below:**")
    st.success(f"{test_train_sentences()}")
    voice_file = st.experimental_audio_input('Record the text displayed above')
    if voice_file:
        voice_file = enhance_audio_to_blob(voice_file.getvalue())
        st.text("Voice capture complete!!!!")
    else:
        pass


submitted = st.button("Register User")

try:
    if submitted:
        if all((first_name, other_name, last_name, dob, phone, about, sex, occupation,
                marital_status, picture_file, voice_file[0])):
            # Save the uploaded picture as BLOB
            picture_data = picture_file.read()

            # Insert into user_db
            user_id = insert_user(first_name, other_name, last_name, dob, phone, about, sex, occupation,
                                  marital_status, picture_data)

            # Insert into voice_db
            insert_voice_embedding(user_id, voice_file)

            with st.spinner("Registering"):
                time.sleep(3)

            st.success(f"User {first_name} registered successfully!")

            # Refresh the page after user registers
            streamlit_js_eval(js_expressions="parent.window.location.reload()")
        else:
            st.error("Please fill in all fields and upload both picture and voice.")
except TypeError:
    st.error("Please fill the form appropriately.")
