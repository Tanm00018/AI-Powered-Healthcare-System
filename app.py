import streamlit as st
import pickle
import numpy as np
import bcrypt
import datetime
import base64
from io import BytesIO
from uuid import uuid4

# --- Load ML Model ---
with open("disease_predictor.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

all_symptoms = vectorizer.get_feature_names_out()

# --- Session Initialization ---
if "users" not in st.session_state:
    st.session_state.users = {}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.role = ""

# --- Auth Functions ---
def signup():
    st.subheader("Signup")
    username = st.text_input("Choose Username")
    password = st.text_input("Choose Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    role = st.selectbox("Role", ["Patient", "Doctor"])
    
    if st.button("Signup"):
        if not username or not password:
            st.error("Username and password are required")
        elif password != confirm_password:
            st.error("Passwords don't match")
        elif len(password) < 8:
            st.error("Password must be at least 8 characters")
        elif username in st.session_state.users:
            st.error("Username already exists.")
        else:
            salt = bcrypt.gensalt()
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
            
            st.session_state.users[username] = {
                "password": hashed_password.decode('utf-8'),
                "role": role,
                "medical_history": {} if role == "Patient" else None,
                "patients": [] if role == "Doctor" else None,
            }
            st.success("Signup successful! Please login.")

def login():
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        user = st.session_state.users.get(username)
        if not user:
            st.error("Invalid credentials.")
            return
            
        stored_password = user["password"].encode('utf-8')
        provided_password = password.encode('utf-8')
        
        if bcrypt.checkpw(provided_password, stored_password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = user["role"]
            st.rerun()
        else:
            st.error("Invalid credentials.")

def logout_button():
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.role = ""
        st.rerun()

# --- UI Components ---
def display_record(record):
    """Display a single medical record"""
    with st.container(border=True):
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f"**Date:** {record.get('timestamp', 'N/A')}")
        with col2:
            st.markdown(f"**Doctor:** Dr. {record.get('doctor', 'N/A')}")
        
        st.divider()
        
        if record.get("treatment", "").strip():
            st.markdown("### Treatment Details")
            st.markdown(record["treatment"])
        
        if record.get("medications", "").strip():
            st.markdown("### Prescribed Medications")
            st.markdown(record["medications"])
        
        if record.get("allergies", "").strip():
            st.markdown("### Recorded Allergies")
            st.markdown(record["allergies"])
        
        if "file" in record and record["file"]:
            file_bytes = record["file"]["bytes"]
            file_name = record["file"]["name"]
            st.markdown("### Attachments")
            
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                st.image(BytesIO(file_bytes), caption=file_name)
            elif file_name.lower().endswith('.pdf'):
                base64_pdf = base64.b64encode(file_bytes).decode('utf-8')
                pdf_display = f"""
                <iframe src="data:application/pdf;base64,{base64_pdf}" 
                        width="100%" height="500px" 
                        style="border:1px solid #444;"></iframe>
                """
                st.markdown(pdf_display, unsafe_allow_html=True)
                st.download_button("Download PDF", file_bytes, file_name, "application/pdf")

# --- Patient Portal ---
def patient_portal():
    st.title("Patient Portal")
    st.write(f"Welcome, {st.session_state.username}")
    logout_button()

    user_data = st.session_state.users[st.session_state.username]
    
    # Medical Summary
    with st.container(border=True):
        st.subheader("📋 Medical Summary")
        
        # Get all records from all folders
        all_records = []
        for folder in user_data["medical_history"].values():
            all_records.extend(folder)
        
        # Sort all records by date (newest first)
        all_records_sorted = sorted(all_records, 
                                  key=lambda x: x.get("timestamp", ""), 
                                  reverse=True)
        
        # Extract summary data
        combined_allergies = set()
        current_medications = []
        
        for record in all_records_sorted:
            if record.get("allergies", "").strip():
                combined_allergies.update(a.strip().title() for a in record["allergies"].split(","))
            if record.get("medications", "").strip():
                current_medications.append({
                    "medications": record["medications"],
                    "doctor": record.get("doctor", "Unknown"),
                    "date": record.get("timestamp", "Unknown date")
                })
        
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("### 🚨 Known Allergies")
                if combined_allergies:
                    for allergy in sorted(combined_allergies):
                        st.markdown(f"- {allergy}")
                else:
                    st.info("No allergies recorded")
        
        with col2:
            with st.container(border=True):
                st.markdown("### 💊 Current Medications")
                if current_medications:
                    for med in current_medications[:3]:  # Show most recent 3
                        st.markdown(f"- {med['medications']} (Prescribed by Dr. {med['doctor']})")
                else:
                    st.info("No medications recorded")

    # AI Symptom Checker
    st.divider()
    with st.container(border=True):
        st.subheader("🤖 AI Symptom Analysis")
        with st.expander("Get AI Assessment", expanded=False):
            user_input = st.text_area("Describe your symptoms (comma separated)", 
                                    placeholder="e.g., fever, headache, cough",
                                    height=100)
            
            if st.button("Analyze Symptoms", type="primary"):
                if not user_input.strip():
                    st.error("Please enter your symptoms")
                else:
                    with st.spinner("Analyzing symptoms..."):
                        user_symptoms = [s.strip().lower() for s in user_input.split(",")]
                        symptoms_vector = np.array([
                            1 if symptom in user_symptoms else 0 for symptom in all_symptoms
                        ]).reshape(1, -1)
                        
                        prediction = model.predict(symptoms_vector)
                        predicted_condition = str(prediction[0])
                        
                        st.success("Analysis Complete")
                        with st.container(border=True):
                            st.markdown(f"### 🔍 Predicted Condition")
                            st.markdown(f"**{predicted_condition}**")
                            st.markdown("""
                            <div style="background-color: #333; padding: 10px; border-radius: 5px;">
                            <small>Note: This AI assessment is not a medical diagnosis. 
                            Please consult your doctor for professional medical advice.</small>
                            </div>
                            """, unsafe_allow_html=True)

    # Medical History Section
    st.divider()
    with st.container(border=True):
        st.subheader("📁 Medical History Folders")
        
        if not user_data["medical_history"]:
            st.info("No medical records available yet.")
        else:
            # Display folder selection buttons
            cols = st.columns(4)
            for i, folder_name in enumerate(user_data["medical_history"].keys()):
                with cols[i % 4]:
                    if st.button(f"📂 {folder_name}"):
                        st.session_state.selected_folder = folder_name
            
            # Display records from selected folder
            if "selected_folder" in st.session_state:
                st.subheader(f"📜 {st.session_state.selected_folder} Records")
                folder_records = user_data["medical_history"][st.session_state.selected_folder]
                sorted_records = sorted(folder_records, 
                                      key=lambda x: x.get("timestamp", ""), 
                                      reverse=True)
                
                for record in sorted_records:
                    display_record(record)

# --- Doctor Portal ---
def doctor_portal():
    st.title("Doctor Portal")
    st.write(f"Welcome, Dr. {st.session_state.username}")
    logout_button()

    # Get all patients (users with role "Patient")
    patient_list = [u for u in st.session_state.users if st.session_state.users[u]["role"] == "Patient"]
    
    if not patient_list:
        st.info("No patients registered.")
        return
    
    selected_patient = st.selectbox("Select Patient", patient_list)
    patient_data = st.session_state.users[selected_patient]
    
    # Summary
    with st.container(border=True):
        st.subheader(f"👤 Patient Summary: {selected_patient}")
        
        # Get all records from all folders
        all_records = []
        for folder in patient_data["medical_history"].values():
            all_records.extend(folder)
        
        combined_allergies = set()
        treatments_count = 0
        
        for record in all_records:
            if record.get("allergies", "").strip():
                combined_allergies.update(a.strip().title() for a in record["allergies"].split(","))
            if record.get("treatment", "").strip():
                treatments_count += 1

        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("### 🚨 Known Allergies")
                if combined_allergies:
                    for allergy in sorted(combined_allergies):
                        st.markdown(f"- {allergy}")
                else:
                    st.info("No allergies recorded")
        
        with col2:
            with st.container(border=True):
                st.markdown("### 📅 Treatment History")
                if treatments_count > 0:
                    st.markdown(f"Total treatments: {treatments_count}")
                    if all_records:
                        latest = sorted(all_records, key=lambda x: x.get("timestamp", ""), reverse=True)[0]
                        st.markdown(f"Last treatment on {latest.get('timestamp', 'Unknown date')}")
                else:
                    st.info("No treatments recorded")

    # Full Records
    st.divider()
    with st.container(border=True):
        st.subheader("📁 Patient Medical Folders")
        
        if not patient_data["medical_history"]:
            st.info("No medical records available.")
        else:
            # Display folder selection
            folder_names = list(patient_data["medical_history"].keys())
            selected_folder = st.selectbox("Select Folder", folder_names)
            
            # Display records from selected folder
            if selected_folder:
                st.subheader(f"📋 {selected_folder} Records")
                folder_records = patient_data["medical_history"][selected_folder]
                
                # Filter options
                col1, col2 = st.columns(2)
                with col1:
                    show_all = st.checkbox("Show all records", value=True)
                with col2:
                    if not show_all:
                        doctor_filter = st.selectbox(
                            "Filter by doctor",
                            ["All"] + list(sorted({r.get("doctor", "Unknown") for r in folder_records})))
                
                # Filter records
                if show_all:
                    filtered_records = folder_records
                else:
                    if doctor_filter == "All":
                        filtered_records = folder_records
                    else:
                        filtered_records = [r for r in folder_records if r.get("doctor", "Unknown") == doctor_filter]
                
                # Display filtered records
                sorted_records = sorted(filtered_records, 
                                      key=lambda x: x.get("timestamp", ""), 
                                      reverse=True)
                for record in sorted_records:
                    display_record(record)

    # Update Form
    st.divider()
    with st.container(border=True):
        st.subheader("✍️ Add New Medical Record")
        with st.form("record_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                # Create new folder or select existing
                folder_option = st.radio("Folder", ["Existing", "New"])
                if folder_option == "Existing":
                    if patient_data["medical_history"]:
                        folder_name = st.selectbox("Select folder", list(patient_data["medical_history"].keys()))
                    else:
                        st.info("No existing folders. Please create a new one.")
                        folder_name = st.text_input("New folder name")
                else:
                    folder_name = st.text_input("New folder name")
                
                allergies = st.text_input("Allergies (comma separated)", 
                                         value=", ".join(sorted(combined_allergies)) if combined_allergies else "")
                medications = st.text_input("Prescribed Medications")
            with col2:
                treatment = st.text_area("Treatment Plan", height=150)
                uploaded_file = st.file_uploader("Upload Report", type=["pdf", "jpg", "jpeg", "png"])

            if st.form_submit_button("Add Medical Record", type="primary"):
                new_record = {
                    "id": str(uuid4()),
                    "doctor": st.session_state.username,
                    "allergies": allergies,
                    "treatment": treatment,
                    "medications": medications,
                    "timestamp": str(datetime.datetime.now())
                }
                if uploaded_file:
                    new_record["file"] = {
                        "name": uploaded_file.name,
                        "bytes": uploaded_file.read()
                    }
                
                # Initialize folder if it doesn't exist
                if folder_name not in patient_data["medical_history"]:
                    patient_data["medical_history"][folder_name] = []
                
                # Add record to folder
                patient_data["medical_history"][folder_name].append(new_record)
                
                # Update doctor-patient relationship
                if st.session_state.username not in patient_data.get("doctors", []):
                    if "doctors" not in patient_data:
                        patient_data["doctors"] = []
                    patient_data["doctors"].append(st.session_state.username)
                
                st.success(f"Record added to '{folder_name}' successfully")
                st.rerun()

# --- Main App ---
def main_app():
    st.set_page_config(page_title="AI Healthcare", page_icon="🩺", layout="wide")
    
    # Dark theme CSS
    st.markdown("""
    <style>
        /* Main background - black */
        .stApp {
            background-color: #000000 !important;
            color: #ffffff !important;
        }
        
        /* All text white */
        body, .stMarkdown, .stMarkdown p, .stMarkdown li, 
        .stTextInput>div>div>input, .stTextArea>div>div>textarea,
        .stSelectbox>div>div>select, .stRadio>div {
            color: #ffffff !important;
        }
        
        /* Headers - white and bold */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #ffffff !important;
            font-weight: 700 !important;
        }
        
        /* Containers - dark gray with lighter gray border */
        .stContainer, .st-bb, .st-at, .st-expander {
            background-color: #1a1a1a !important;
            border: 1px solid #444 !important;
            color: #ffffff !important;
        }
        
        /* Buttons - blue with white text */
        .stButton>button {
            background-color: #1e88e5 !important;
            color: white !important;
            border: none !important;
        }
        
        /* Input fields - dark gray with gray border */
        .stTextInput>div>div>input, 
        .stTextArea>div>div>textarea,
        .stSelectbox>div>div>select,
        .stFileUploader>div>div {
            background-color: #333 !important;
            border: 1px solid #444 !important;
            color: white !important;
        }
        
        /* Info boxes - dark blue background, white text */
        .stAlert, .stInfo {
            background-color: #0d47a1 !important;
            color: #ffffff !important;
            border: 1px solid #1976d2 !important;
        }
        
        /* Success messages - dark green background, white text */
        .stSuccess {
            background-color: #2e7d32 !important;
            color: #ffffff !important;
            border: 1px solid #4caf50 !important;
        }
        
        /* Error messages - dark red background, white text */
        .stError {
            background-color: #c62828 !important;
            color: #ffffff !important;
            border: 1px solid #f44336 !important;
        }
        
        /* Dividers - gray */
        .stDivider {
            border-color: #444 !important;
        }
        
        /* Radio buttons - white text */
        .stRadio label {
            color: white !important;
        }
        
        /* Selectbox dropdown - dark background */
        .stSelectbox div[data-baseweb="select"] > div {
            background-color: #333 !important;
            color: white !important;
        }
        
        /* Dropdown options - dark background */
        .stSelectbox div[role="listbox"] {
            background-color: #333 !important;
            color: white !important;
        }
        
        /* File uploader - dark background */
        .stFileUploader>div>div {
            background-color: #333 !important;
        }
        
        /* Folder buttons */
        .stButton>button {
            width: 100%;
            margin-bottom: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🩺 AI Healthcare Assistant")
    if not st.session_state.logged_in:
        choice = st.radio("Choose Action", ["Login", "Signup"], horizontal=True)
        signup() if choice == "Signup" else login()
    else:
        patient_portal() if st.session_state.role == "Patient" else doctor_portal()

if __name__ == "__main__":
    main_app()
