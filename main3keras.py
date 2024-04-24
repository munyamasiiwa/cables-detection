import streamlit as st
import mysql.connector
from mysql.connector import Error
from passlib.hash import bcrypt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pandas as pd
import cv2
import os
import re  # Regular expression library for validation
import tempfile
from tensorflow.keras.layers import InputLayer
import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope

def custom_layer_deserializer(cls, cls_config):
    # Remove the 'batch_shape' from config if it exists since it's causing the error
    cls_config.pop('event_shape', None)  # Safely remove the batch_shape key if it exists
    # Create the layer with the modified configuration
    return cls(**cls_config)

model_path = 'model3.h5'  # Update with the correct path

# Define custom objects if any other custom layers are used
custom_objects = {
    'InputLayer': custom_layer_deserializer,
    # include other custom layers if necessary
}

with custom_object_scope(custom_objects):
     model = load_model(model_path)


def create_database_connection():
    """Create database connection."""
    try:
        connection = mysql.connector.connect(
            host="localhost", user="root", passwd="", database="detection")
        return connection
    except Error as e:
        st.error(f"Database connection failed due to {e}")
        return None

def hash_password(password):
    """Hash the password using bcrypt."""
    return bcrypt.hash(password)

def check_password(password, hashed):
    """Check hashed password."""
    return bcrypt.verify(password, hashed)

def validate_signup(username, email, password):
    """Validates signup form data."""
    if not username:
        return "Username cannot be empty."
    if len(username) < 5:
        return "Username must be at least 5 characters long."
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return "Invalid email format."
    if len(password) < 8:
        return "Password must be at least 8 characters long."
    if not re.search(r"[A-Z]", password):
        return "Password must contain at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return "Password must contain at least one lowercase letter."
    if not re.search(r"\d", password):
        return "Password must contain at least one digit."
    if not re.search(r"[^a-zA-Z\d]", password):
        return "Password must contain at least one special character."
    return ""

def signup_page():
    """Signup page logic."""
    st.image("caf.png", width=150)
    st.subheader("Signup Form")
    with st.form("signup_form"):
        username = st.text_input("Username", key="signup_username")
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_password")
        role = st.selectbox("Role", ["user", "admin"], key="signup_role")
        submit_button = st.form_submit_button("Signup")
        if submit_button:
            error_message = validate_signup(username, email, password)
            if error_message:
                st.error(error_message)
            else:
                hashed_password = hash_password(password)
                conn = create_database_connection()
                if conn:
                    cursor = conn.cursor()
                    cursor.execute("INSERT INTO users (username, email, password, role) VALUES (%s, %s, %s, %s)",
                                   (username, email, hashed_password, role))
                    conn.commit()
                    cursor.close()
                    conn.close()
                    st.success("You have successfully signed up.")

def login_page():
    """Login page logic."""
    st.image("caf.png", width=150)
    st.subheader("Login Form")
    with st.form("login_form"):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        # Fetching roles from the database to include in the dropdown
        conn = create_database_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT role FROM users")
            roles = [role[0] for role in cursor.fetchall()]
            cursor.close()
            conn.close()
        role = st.selectbox("Role", roles, key="login_role")  # Role dropdown

        submit_button = st.form_submit_button("Login")
        if submit_button:
            if not username or not password:
                st.error("Username and password cannot be empty.")
            else:
                conn = create_database_connection()
                if conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM users WHERE username=%s AND role=%s", (username, role))
                    user = cursor.fetchone()
                    cursor.close()
                    conn.close()
                    if user and check_password(password, user[2]):
                        st.success("Logged in successfully.")
                        st.session_state['logged_in'] = True
                        st.session_state['user_role'] = role
                    else:
                        st.error("Login failed. Please check your username and password.")
def prediction_page():
    """Prediction page logic."""
    st.image("caf.png", width=150)
    st.title('Cable Condition Prediction')

    uploaded_file = st.file_uploader("Upload an image or video of the cable", type=["jpg", "jpeg", "png", "mp4"])

    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1]
        if file_type in ['mp4']:
            st.video(uploaded_file)
            frames = extract_frames(uploaded_file)
            defect_found = False

            for i, frame in enumerate(frames):
                predicted_class_index = predict_on_frame(frame, model)
                class_labels = {0: 'DEFECTIVE', 1: 'NON DEFECTIVE'}

                if class_labels[predicted_class_index] == 'DEFECTIVE':
                    st.error('Prediction: DEFECTIVE CABLE (DAMAGED INSULATION)')
                    st.image(frame, caption='Defective Frame', use_column_width=True)
                    defect_found = True
                    break  # Stop after finding the first defective frame

            if not defect_found:
                st.success('Prediction: WELL INSULATED CABLE')
                if frames:
                    st.image(frames[-1], caption='Last Inspected Frame', use_column_width=True)  # Show last frame if no defects

        else:
            # Image handling as previously defined
            img = image.load_img(uploaded_file, target_size=(260, 260))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.  # Normalize the image
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions[0])
            class_labels = {0: 'DEFECTIVE', 1: 'NON DEFECTIVE'}

            if class_labels[predicted_class_index] == 'NON DEFECTIVE':
                st.success('Prediction: WELL INSULATED CABLE')
                st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
            else:
                st.error('Prediction: DEFECTIVE CABLE (DAMAGED INSULATION)')
                st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

def extract_frames(video_file):
    """Extract frames from the uploaded video file."""
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    tfile.flush()
    vidcap = cv2.VideoCapture(tfile.name)

    success, frame = vidcap.read()
    frames = []
    while success:
        frames.append(frame)
        success, frame = vidcap.read()
    vidcap.release()
    tfile.close()
    os.unlink(tfile.name)
    return frames

def predict_on_frame(frame, model):
    """Predict on a single frame using the loaded model."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (260, 260))
    img_array = image.img_to_array(frame)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Normalize the image
    prediction = model.predict(img_array)
    return np.argmax(prediction[0])


    predicted_class_label = class_labels[predicted_class_index]
        # st.write(f"Predicted class label: {predicted_class_label}")  # Optionally display the class label


def user_management_page():
    """User management page logic."""
    st.title('User Management')
    conn = create_database_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, email, role FROM users")
        users = cursor.fetchall()
        cursor.close()
        conn.close()

        # Display user information in a table
        df = pd.DataFrame(users, columns=['ID', 'Username', 'Email', 'Role'])
        st.write("### User Information", df)

        if st.session_state.get('user_role') == 'admin':
            selected_indices = st.multiselect("Select rows:", df.index)

            # Edit user information
            if selected_indices and st.button("Edit Selected"):
                for i in selected_indices:
                    user_id = df.loc[i, 'ID']
                    with st.form(f"form_edit_{user_id}"):
                        new_email = st.text_input("Email", value=df.loc[i, 'Email'], key=f"email_{i}")
                        new_role = st.selectbox("Role", ["user", "admin"], index=["user", "admin"].index(df.loc[i, 'Role']), key=f"role_{i}")
                        submit_button = st.form_submit_button("Update")
                        if submit_button:
                            conn = create_database_connection()
                            if conn:
                                cursor = conn.cursor()
                                cursor.execute("UPDATE users SET email=%s, role=%s WHERE id=%s", (new_email, new_role, user_id))
                                conn.commit()
                                cursor.close()
                                conn.close()
                                st.success(f"Updated user {df.loc[i, 'Username']}")
                                st.experimental_rerun()

            # Delete user
            if selected_indices and st.button("Delete Selected"):
                delete_confirmed = st.checkbox("Confirm delete?")
                if delete_confirmed and st.button("Confirm Delete"):
                    conn = create_database_connection()
                    if conn:
                        for i in selected_indices:
                            user_id = df.loc[i, 'ID']
                            cursor = conn.cursor()
                            cursor.execute("DELETE FROM users WHERE id=%s", (user_id,))
                            conn.commit()
                            cursor.close()
                            conn.close()
                            st.success(f"User {df.loc[i, 'Username']} deleted successfully.")
                            df.drop(index=i, inplace=True)  # Update the dataframe after deletion
                            st.experimental_rerun()

def logout():
    """Logout function."""
    st.session_state['logged_in'] = False
    st.session_state.pop('user_role', None)  # Optionally remove the role from session state
    st.write("You have been logged out.")

def main():
    st.sidebar.title("Navigation")

    if 'logged_in' in st.session_state and st.session_state['logged_in']:
        page_options = ["Predictions", "User Management"]
        page = st.sidebar.radio("Choose a page:", page_options)
        if st.sidebar.button("Logout"):
            logout()
    else:
        page = st.sidebar.radio("Choose a page:", ["Login", "Signup"])

    if page == "Login":
        login_page()
    elif page == "Signup":
        signup_page()
    elif page == "Predictions":
        prediction_page()
    elif page == "User Management":
        user_management_page()

if __name__ == "__main__":
    main()

