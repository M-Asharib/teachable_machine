import streamlit as st
import requests
import os
import time
import base64
import pandas as pd
from io import BytesIO
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="Teachable Machine Pro",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configuration Constants ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# --- Custom Premium CSS Injection ---
st.markdown("""
    <style>
        /* Import outfit font */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }

        /* Glassmorphism card container */
        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        }

        /* Subtle gradient titles */
        .gradient-text {
            background: linear-gradient(135deg, #00C6FF 0%, #0072FF 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }

        .gradient-accent {
            background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }

        /* Metric/Card labels */
        .card-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: #FFFFFF;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .sample-badge {
            background: linear-gradient(135deg, #00C6FF 0%, #0072FF 100%);
            color: white;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: bold;
            box-shadow: 0 4px 10px rgba(0, 198, 255, 0.3);
        }

        /* Custom progress bar styles */
        .meter-container {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.06);
            padding: 16px;
            border-radius: 12px;
            margin-bottom: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease;
        }

        .meter-container:hover {
            transform: translateY(-2px);
            border-color: rgba(255, 255, 255, 0.15);
        }

        .meter-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
        }

        .meter-label {
            font-weight: 600;
            color: #E2E8F0;
            font-size: 1rem;
        }

        .meter-value {
            font-weight: 700;
            color: #00E5FF;
            font-size: 1rem;
        }

        .meter-track {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 6px;
            height: 10px;
            width: 100%;
            overflow: hidden;
        }

        .meter-fill {
            background: linear-gradient(90deg, #00C6FF 0%, #0072FF 100%);
            height: 100%;
            border-radius: 6px;
            transition: width 0.4s cubic-bezier(0.1, 0.8, 0.25, 1);
        }

        .winner-banner {
            background: rgba(0, 229, 255, 0.1);
            border: 1px solid rgba(0, 229, 255, 0.3);
            border-radius: 12px;
            padding: 16px;
            text-align: center;
            margin-bottom: 20px;
            color: #00E5FF;
            font-weight: 700;
            font-size: 1.2rem;
            box-shadow: 0 4px 20px rgba(0, 229, 255, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "classes" not in st.session_state:
    st.session_state.classes = []
if "is_trained" not in st.session_state:
    st.session_state.is_trained = False
if "backend_active" not in st.session_state:
    st.session_state.backend_active = False
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# --- Helper Functions ---
def check_backend_connection():
    """Checks FastAPI backend availability and status."""
    try:
        response = requests.get(BACKEND_URL, timeout=2)
        if response.status_code == 200:
            data = response.json()
            st.session_state.backend_active = True
            st.session_state.is_trained = data.get("model_trained", False)
            # Sync active classes from backend database
            backend_classes = data.get("active_classes", [])
            for c in backend_classes:
                if c not in st.session_state.classes:
                    st.session_state.classes.append(c)
            return data
    except Exception:
        st.session_state.backend_active = False
    return None

def fetch_class_sample_counts():
    """Queries backend folder list to count files in each class."""
    counts = {}
    backend_info = check_backend_connection()
    if backend_info and st.session_state.backend_active:
        # Check folders inside the dataset
        try:
            # We can use backend info or fetch a list of classes
            # Since main.py only returns active class names in GET /,
            # We can run query or do individual scans.
            # To avoid adding complex API, we do standard directory listing
            # if we have backend access. Alternatively, we can let FastAPI expose sample counts.
            # Let's see: FastAPI returns health details containing dataset_directory and active_classes.
            # Wait, can we fetch the list of files from backend?
            # To make it robust, let's keep track locally or get it from backend.
            # Let's inspect e:\SMIT\teachable_machine\backend\main.py:
            # It doesn't have an explicit /counts endpoint but / has 'active_classes'.
            # Wait, we can fetch counts by checking the backend dataset folder if we are on the same machine!
            # Since Streamlit and FastAPI run on the same filesystem in our local development workspace,
            # we can read the directories directly! This is robust and fast.
            backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend", "dataset")
            if os.path.exists(backend_dir):
                for c in st.session_state.classes:
                    class_path = os.path.join(backend_dir, c)
                    if os.path.exists(class_path):
                        files = [
                            f for f in os.listdir(class_path)
                            if os.path.splitext(f)[1].lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
                        ]
                        counts[c] = len(files)
                    else:
                        counts[c] = 0
            else:
                for c in st.session_state.classes:
                    counts[c] = 0
        except Exception:
            for c in st.session_state.classes:
                counts[c] = 0
    else:
        for c in st.session_state.classes:
            counts[c] = 0
    return counts

# --- Perform initial connection check ---
backend_info = check_backend_connection()
class_counts = fetch_class_sample_counts()

# --- HEADER SECTION ---
st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <h1 style='margin-bottom: 5px; font-weight: 800; font-size: 2.8rem;'>
            ✨ <span class='gradient-text'>Teachable Machine Pro</span>
        </h1>
        <p style='color: #94A3B8; font-size: 1.15rem; max-width: 700px; margin: 0 auto 10px auto;'>
            A powerful decoupled AI platform. Define custom categories, upload samples or use your webcam to 
            collect training data, and train a transfer learning model in seconds!
        </p>
    </div>
""", unsafe_allow_html=True)

# --- SIDEBAR: SYSTEM CONTROLS ---
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 10px 0;'>
            <h3 style='margin: 0; font-weight: 700;'><span class='gradient-text'>System Console</span></h3>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Connection status display
    if st.session_state.backend_active:
        st.success("🟢 API Server: Connected")
        st.markdown(f"**Backend Host:** `{BACKEND_URL}`")
        st.markdown(f"**Model Status:** {'✓ Trained & Ready' if st.session_state.is_trained else '⚡ Needs Training'}")
        if backend_info:
            st.markdown(f"**Classes in System:** `{len(backend_info.get('active_classes', []))}`")
    else:
        st.error("🔴 API Server: Disconnected")
        st.warning("Please ensure the FastAPI backend is running. Start it with:\n\n`uvicorn backend.main:app --reload --port 8000`")
        if st.button("🔄 Retry Connection"):
            st.rerun()

    st.markdown("---")
    st.markdown("### 🛠️ Workspace Actions")
    
    # Reset everything button
    if st.button("🗑️ Reset Workspace", use_container_width=True, type="secondary"):
        if st.session_state.backend_active:
            try:
                res = requests.delete(f"{BACKEND_URL}/reset")
                if res.status_code == 200:
                    st.session_state.classes = []
                    st.session_state.is_trained = False
                    st.session_state.last_prediction = None
                    st.session_state.prediction_history = []
                    st.toast("Workspace reset successfully!", icon="🗑️")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Failed to reset backend dataset.")
            except Exception as e:
                st.error(f"Error resetting: {str(e)}")
        else:
            st.error("Cannot reset: Backend is down.")

    st.markdown("---")
    st.markdown("""
        <div style='font-size: 0.85rem; color: #64748B; text-align: center; margin-top: 50px;'>
            Teachable Machine Clone v1.0<br>
            Powered by PyTorch, MobileNetV3, and FastAPI
        </div>
    """, unsafe_allow_html=True)

# --- MAIN PAGE LAYOUT ---
if not st.session_state.backend_active:
    st.info("👋 **Welcome!** Please launch the FastAPI backend to start defining classes and uploading training images.")
    st.stop()

# Layout split: Left side for data collection & training, Right side for live preview/inference
left_col, right_col = st.columns([1.1, 0.9], gap="large")

# ==========================================
# LEFT COLUMN: DATASET DEFINITION & INGESTION
# ==========================================
with left_col:
    st.markdown("### 📂 1. Define Categories")
    
    # Custom Class Addition
    with st.form("add_class_form", clear_on_submit=True):
        new_class_input = st.text_input("Enter category name:", placeholder="e.g. Mug, Hand, Book, Remote")
        submit_class = st.form_submit_button("＋ Add Category", use_container_width=True)
        
        if submit_class and new_class_input:
            clean_name = new_class_input.strip()
            # Basic validation
            if clean_name.lower() in [c.lower() for c in st.session_state.classes]:
                st.error(f"Category '{clean_name}' already exists.")
            elif len(clean_name) < 2:
                st.error("Category name must be at least 2 characters.")
            else:
                # Add locally, directory created automatically on sample upload
                st.session_state.classes.append(clean_name)
                st.toast(f"Category '{clean_name}' added!", icon="✨")
                st.rerun()

    # Category Grid Cards Display
    if st.session_state.classes:
        st.markdown("<div style='margin-bottom: 10px; font-weight: 600; color: #94A3B8;'>Active Categories</div>", unsafe_allow_html=True)
        cols = st.columns(min(len(st.session_state.classes), 3))
        for idx, c in enumerate(st.session_state.classes):
            col_target = cols[idx % 3]
            count = class_counts.get(c, 0)
            
            with col_target:
                st.markdown(f"""
                    <div class='glass-card' style='padding: 16px; text-align: center; border-radius: 12px;'>
                        <div class='card-title' style='justify-content: center; font-size: 1.1rem; margin-bottom: 10px;'>
                            {c}
                        </div>
                        <span class='sample-badge'>{count} samples</span>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No categories added yet. Add a category above to start collecting training samples!")
        st.stop()

    st.markdown("---")
    st.markdown("### 📷 2. Ingest Training Samples")
    
    # Choose category to upload to
    selected_class = st.selectbox(
        "Select category to collect samples for:",
        options=st.session_state.classes,
        index=0
    )

    # Image collection method selection
    input_method = st.radio(
        "Select capture method:",
        options=["📷 Live Webcam Capture", "📁 Bulk Upload Files"],
        horizontal=True
    )

    # Ingestion Core
    if input_method == "📁 Bulk Upload Files":
        uploaded_files = st.file_uploader(
            "Drag & drop custom images for this class:",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            accept_multiple_files=True,
            key=f"uploader_{selected_class}"
        )
        
        if uploaded_files:
            if st.button(f"📤 Upload {len(uploaded_files)} image(s) to '{selected_class}'", use_container_width=True, type="primary"):
                files_payload = []
                for file in uploaded_files:
                    files_payload.append(("files", (file.name, file.read(), file.type)))
                
                with st.spinner("Uploading samples..."):
                    try:
                        res = requests.post(
                            f"{BACKEND_URL}/upload-sample",
                            data={"class_name": selected_class},
                            files=files_payload
                        )
                        if res.status_code == 200:
                            st.success(f"Uploaded {len(uploaded_files)} samples successfully!")
                            st.toast("Dataset updated!", icon="📥")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"Upload failed: {res.json().get('detail', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Error during upload: {str(e)}")
                        
    else: # Webcam Capture Mode
        st.markdown("*Use your webcam to take custom snapshots. Images will be automatically named with secure UUIDs.*")
        webcam_image = st.camera_input("Smile & click capture:")
        
        if webcam_image:
            # Show active button to save the captured frame
            if st.button(f"📥 Save Snapshot to '{selected_class}'", use_container_width=True, type="primary"):
                file_bytes = webcam_image.read()
                files_payload = [("files", ("webcam.jpg", file_bytes, "image/jpeg"))]
                
                with st.spinner("Saving snapshot..."):
                    try:
                        res = requests.post(
                            f"{BACKEND_URL}/upload-sample",
                            data={"class_name": selected_class},
                            files=files_payload
                        )
                        if res.status_code == 200:
                            st.success(f"Snapshot added to '{selected_class}' successfully!")
                            st.toast("Snapshot saved!", icon="📷")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"Failed to save snapshot: {res.json().get('detail', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Error saving snapshot: {str(e)}")

    st.markdown("---")
    st.markdown("### 🚂 3. Transfer Learning Engine")
    
    # Show active checklist for training
    ready_to_train = True
    active_classes_count = 0
    
    st.markdown("#### Training Eligibility Checklist:")
    
    # Criterion 1: At least 2 classes
    if len(st.session_state.classes) >= 2:
        st.markdown("✅ Defined at least 2 categories.")
    else:
        st.markdown("❌ Defined at least 2 categories. *(Need more)*")
        ready_to_train = False
        
    # Criterion 2: Each class must contain images
    for c in st.session_state.classes:
        count = class_counts.get(c, 0)
        if count > 0:
            st.markdown(f"✅ Category '{c}' has {count} sample(s).")
            active_classes_count += 1
        else:
            st.markdown(f"❌ Category '{c}' has 0 samples. *(Upload data first)*")
            ready_to_train = False

    st.markdown("")
    
    # Train Button Trigger
    if st.button("🚂 Train Custom Model", use_container_width=True, disabled=not ready_to_train, type="primary"):
        with st.spinner("Extracting deep visual representations using MobileNetV3 and fitting classifier... Please wait."):
            try:
                res = requests.post(f"{BACKEND_URL}/train")
                if res.status_code == 200:
                    data = res.json()
                    st.success("🎉 Custom classifier trained successfully!")
                    st.session_state.is_trained = True
                    st.session_state.prediction_history = []
                    st.balloons()
                    time.sleep(2)
                    st.rerun()
                else:
                    detail = res.json().get("detail", "Unknown server-side training error")
                    st.error(f"Training Failed: {detail}")
            except Exception as e:
                st.error(f"Error triggering training: {str(e)}")

# ==========================================
# RIGHT COLUMN: STATE-GATED PREVIEW & PREDICTION
# ==========================================
with right_col:
    st.markdown("### 🔍 4. Live Model Testing")
    
    if not st.session_state.is_trained:
        st.markdown("""
            <div class='glass-card' style='text-align: center; border-color: rgba(255, 75, 75, 0.2); padding: 40px 20px;'>
                <h3 style='margin: 0 0 10px 0; color: #FF4B4B;'>Testing Panel Locked</h3>
                <p style='color: #94A3B8; font-size: 0.95rem; margin: 0;'>
                    To prevent system errors, you cannot run visual predictions until a custom model has been successfully trained.
                </p>
                <p style='color: #64748B; font-size: 0.85rem; margin-top: 10px;'>
                    Complete the 3 milestones on the left side to unlock.
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style='background: rgba(0, 229, 255, 0.05); border: 1px solid rgba(0, 229, 255, 0.15); border-radius: 12px; padding: 12px; margin-bottom: 20px; font-size: 0.95rem; color: #E2E8F0;'>
                🎉 <b>Classifier Unlocked!</b> Feed testing inputs to get real-time confidence scores and predictions.
            </div>
        """, unsafe_allow_html=True)

        test_input_method = st.radio(
            "Select testing input source:",
            options=["📷 Live Webcam Test", "📁 Test Image File"],
            horizontal=True
        )

        test_image_file = None
        
        if test_input_method == "📁 Test Image File":
            test_image_file = st.file_uploader(
                "Upload a test image:",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
                key="test_uploader"
            )
        else:
            test_image_file = st.camera_input("Capture live test snapshot:")

        # Predict Trigger
        if test_image_file:
            img_bytes = test_image_file.read()
            files_payload = {"file": ("test_frame.jpg", img_bytes, "image/jpeg")}
            
            with st.spinner("Analyzing image..."):
                try:
                    res = requests.post(f"{BACKEND_URL}/predict", files=files_payload)
                    if res.status_code == 200:
                        st.session_state.last_prediction = res.json()
                    else:
                        st.error(f"Inference failed: {res.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Inference endpoint error: {str(e)}")

        # Render Prediction Results beautifully using custom HTML progress bars
        if st.session_state.last_prediction:
            pred_data = st.session_state.last_prediction
            winner = pred_data.get("predicted_class", "None")
            probabilities = pred_data.get("probabilities", {})

            # 1. Update rolling prediction history tracker
            if "prediction_history" not in st.session_state:
                st.session_state.prediction_history = []
            
            st.session_state.prediction_history.append(probabilities)
            if len(st.session_state.prediction_history) > 15:
                st.session_state.prediction_history.pop(0)

            # Helper to load base64 images
            def load_b64_image(b64_str):
                return Image.open(BytesIO(base64.b64decode(b64_str)))

            # 2. Render Live Visual Scanning Overlays Side-by-Side in Tabs
            st.markdown("#### 📺 Live Visual Telemetry")
            tab_bbox, tab_saliency = st.tabs(["🎯 Bounding Box (Square Overlay)", "🔥 Neural Attention Map"])
            
            with tab_bbox:
                if "bounding_box_image" in pred_data:
                    st.image(load_b64_image(pred_data["bounding_box_image"]), use_column_width=True, caption="Dynamic Foreground Bounding Box")
                else:
                    st.info("No bounding box metadata returned from model.")
            
            with tab_saliency:
                if "saliency_image" in pred_data:
                    st.image(load_b64_image(pred_data["saliency_image"]), use_column_width=True, caption="Gradient-based MobileNetV3 Saliency Focus Map")
                else:
                    st.info("No saliency heatmap metadata returned from model.")

            # 3. Render Latency & Hardware Telemetry Cards
            col_lat, col_dev = st.columns(2)
            with col_lat:
                st.metric("Inference Latency", f"{pred_data.get('inference_time_ms', 0)} ms")
            with col_dev:
                st.metric("Computation Engine", "CPU (MobileNetV3)")

            st.markdown(f"""
                <div class='winner-banner' style='margin-top: 15px;'>
                    ✨ Best Prediction: {winner}
                </div>
                <h4 style='margin-top: 20px; margin-bottom: 15px;'>Confidence Metrics</h4>
            """, unsafe_allow_html=True)

            # Sort probabilities by score descending
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

            for label, prob in sorted_probs:
                # Set dynamic visual styling based on winner
                is_winner = (label == winner)
                bar_color = "linear-gradient(90deg, #00E5FF 0%, #0072FF 100%)" if is_winner else "linear-gradient(90deg, #64748B 0%, #475569 100%)"
                val_color = "#00E5FF" if is_winner else "#94A3B8"
                weight = "bold" if is_winner else "normal"

                st.markdown(f"""
                    <div class='meter-container'>
                        <div class='meter-header'>
                            <span class='meter-label' style='font-weight: {weight};'>{label}</span>
                            <span class='meter-value' style='color: {val_color}; font-weight: bold;'>{prob}%</span>
                        </div>
                        <div class='meter-track'>
                            <div class='meter-fill' style='width: {prob}%; background: {bar_color};'></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            # 4. Display Rolling Timeline Graph
            st.markdown("#### 📈 Probability History Timeline")
            history_df = pd.DataFrame(st.session_state.prediction_history)
            st.line_chart(history_df)
            
        else:
            st.markdown("""
                <div class='glass-card' style='text-align: center; color: #64748B; padding: 40px 10px; font-size: 0.95rem;'>
                    💡 Feed an image file or take a camera snapshot to display live prediction metrics.
                </div>
            """, unsafe_allow_html=True)
