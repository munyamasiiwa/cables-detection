import streamlit as st
import mysql.connector
from mysql.connector import Error
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pandas as pd
import cv2
import os
from datetime import datetime

# Define paths
model_path = 'model3.keras'  # Update with the path to your model
uploads_dir = 'uploads/'  # Directory to save uploaded files

# Ensure the uploads directory exists
os.makedirs(uploads_dir, exist_ok=True)

# Load the saved model
model = load_model(model_path)

def create_database_connection():
    """Create database connection."""
    try:
        connection = mysql.connector.connect(
            host="localhost", user="root", passwd="", database="cafca")
        return connection
    except Error as e:
        st.error(f"Database connection failed due to {e}")
        return None

def extract_frames(video_path, max_frames=50):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

def predict_on_frame(frame, model):
    img_array = cv2.resize(frame, (260, 260))
    img_array = image.img_to_array(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    predictions = model.predict(img_array)
    return np.argmax(predictions[0])

def prediction_page():
    """Prediction page logic."""
    st.image("caf.png", width=150)
    st.title('Cable Condition Prediction')

    uploaded_file = st.file_uploader("Upload an image or video of the cable", type=["jpg", "jpeg", "png", "mp4"])

    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1]
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        defective_frame_path = None  # Initialize the path for the defective frame
        defect_found = False  # Initialize defect_found here

        # Save uploaded file to the uploads directory
        file_path = os.path.join(uploads_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        if file_type == 'mp4':
            st.video(file_path)
            frames = extract_frames(file_path)

            for i, frame in enumerate(frames):
                predicted_class_index = predict_on_frame(frame, model)
                class_labels = {0: 'DEFECTIVE', 1: 'NON DEFECTIVE'}

                if class_labels[predicted_class_index] == 'DEFECTIVE':
                    st.error('Prediction: DEFECTIVE CABLE (DAMAGED INSULATION)')
                    st.image(frame, caption='Defective Frame', use_column_width=True)
                    
                    defective_frame_path = os.path.join(uploads_dir, f"defective_frame_{timestamp}_{i}.jpg")
                    cv2.imwrite(defective_frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    defect_found = True
                    break

            if not defect_found:
                st.success('Prediction: WELL INSULATED CABLE')
                if frames:
                    st.image(frames[-1], caption='Last Inspected Frame', use_column_width=True)
        else:
            # Image handling
            img = image.load_img(file_path, target_size=(260, 260))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize the image
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions[0])
            class_labels = {0: 'DEFECTIVE', 1: 'NON DEFECTIVE'}

            if class_labels[predicted_class_index] == 'NON DEFECTIVE':
                st.success('Prediction: WELL INSULATED CABLE')
                st.image(file_path, caption='Uploaded Image', use_column_width=True)
            else:
                st.error('Prediction: DEFECTIVE CABLE (DAMAGED INSULATION)')
                st.image(file_path, caption='Uploaded Image', use_column_width=True)
                defect_found = True

        # Save prediction result and paths to the database
        conn = create_database_connection()
        if conn:
            cursor = conn.cursor()
            prediction = 'DEFECTIVE' if defect_found else 'NON DEFECTIVE'
            cursor.execute(
                "INSERT INTO predictions (filename, prediction, timestamp, file_path, defective_frame_path) VALUES (%s, %s, %s, %s, %s)",
                (uploaded_file.name, prediction, timestamp, file_path, defective_frame_path)
            )
            conn.commit()
            cursor.close()
            conn.close()

def reports_page():
    """Reports page logic."""
    st.image("caf.png", width=150)
    st.title('Prediction Reports')
    
    conn = create_database_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM predictions")
        records = cursor.fetchall()
        cursor.close()
        conn.close()

        # Make sure the column names match the database table structure
        df = pd.DataFrame(records, columns=['ID', 'Filename', 'Prediction','Timestamp','File Path' ,'Defective Frame Path'])
        st.write("### Prediction Records")
        st.dataframe(df)

        for index, row in df.iterrows():
            st.write(f"**Filename**: {row['Filename']}")
            st.write(f"**Prediction**: {row['Prediction']}")
            st.write(f"**Timestamp**: {row['Timestamp']}")
            file_path = row['File Path']
            
            defective_frame_path = row['Defective Frame Path']

            if file_path:
                st.write(f"**File Path**: {file_path}")
                try:
                    if file_path.endswith(('jpg', 'jpeg', 'png')):
                        st.image(file_path, caption=row['Filename'])
                    elif file_path.endswith('mp4'):
                        st.video(file_path)
                except Exception as e:
                    st.error(f"Error displaying file: {e}")

            if defective_frame_path and os.path.exists(defective_frame_path):
                st.write(f"**Defective Frame Path**: {defective_frame_path}")
                try:
                    st.image(defective_frame_path, caption='Defective Frame')
                except Exception as e:
                    st.error(f"Error displaying defective frame: {e}")
            else:
                st.write("Defective frame not available or not found.")

def main():
    st.sidebar.title("Navigation")
    page_options = ["Predictions", "Reports"]
    page = st.sidebar.radio("Choose a page:", page_options, key="main_radio")

    if page == "Predictions":
        prediction_page()
    elif page == "Reports":
        reports_page()

if __name__ == "__main__":
    main()
