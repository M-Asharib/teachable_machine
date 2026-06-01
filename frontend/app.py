import streamlit as st
import requests
import os
import time
import base64
import pandas as pd
import re
from io import BytesIO
from PIL import Image
import streamlit.components.v1 as components

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
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "train_summary" not in st.session_state:
    st.session_state.train_summary = None

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

@st.cache_data(ttl=3, show_spinner=False)
def fetch_class_sample_counts():
    """Queries backend /dataset-info endpoint to get class file counts."""
    counts = {}
    for c in st.session_state.classes:
        counts[c] = 0
        
    if st.session_state.backend_active:
        try:
            response = requests.get(f"{BACKEND_URL}/dataset-info", timeout=2)
            if response.status_code == 200:
                data = response.json()
                classes_data = data.get("classes", {})
                for c, count in classes_data.items():
                    counts[c] = count
                    if c not in st.session_state.classes:
                        st.session_state.classes.append(c)
        except Exception:
            pass
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
        
        # Load and display model pickle metadata if trained
        if st.session_state.is_trained:
            with st.expander("📦 Model File (.pkl) Inspector", expanded=False):
                try:
                    res = requests.get(f"{BACKEND_URL}/model-info", timeout=2)
                    if res.status_code == 200:
                        meta = res.json()
                        st.write(f"**Backbone:** `{meta.get('backbone_name', 'MobileNetV3')}`")
                        st.write(f"**Feature Dims:** `{meta.get('features_dim', 576)}`")
                        st.write("**Classes Map:**")
                        st.json(meta.get("label_map", {}))
                        
                        clf_type = meta.get("classifier_type")
                        if clf_type:
                            st.write(f"**Classifier:** `{clf_type}`")
                            st.write(f"**Penalty:** `{meta.get('penalty', 'l2')}`")
                            st.write(f"**Regularizer C:** `{meta.get('c_value', 1.0)}`")
                    else:
                        st.info("No trained model details available.")
                except Exception as e:
                    st.error(f"Error reading model info: {e}")
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

    # ── Model Download Button ────────────────────────────────────────────────────
    if st.session_state.get("is_trained", False):
        try:
            model_bytes = requests.get(f"{BACKEND_URL}/export-model", timeout=5)
            if model_bytes.status_code == 200:
                st.download_button(
                    label="⬇️ Download Trained Model (.pkl)",
                    data=model_bytes.content,
                    file_name="teachable_machine_model.pkl",
                    mime="application/octet-stream",
                    use_container_width=True,
                    help="Download your trained model to use it in other Python applications."
                )
        except Exception:
            pass

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

# Define main view tabs
tab_workspace, tab_theory, tab_analytics = st.tabs(["✨ Model Training Workspace", "📖 Theory & Simulators", "📊 Analytics Dashboard"])

with tab_workspace:
    # Layout split: Left side for data collection & training, Right side for live preview/inference
    left_col, right_col = st.columns([1.1, 0.9], gap="large")

    # ==========================================
    # LEFT COLUMN: DATASET DEFINITION & INGESTION
    # ==========================================
    with left_col:
        st.markdown("### 📂 1. Define Categories")
        
        # Custom Class Addition
        new_class = st.text_input("Enter category name (e.g., 'Apple', 'Banana'):", placeholder="Class Name")
        if st.button("➕ Add Category", use_container_width=True):
            if new_class:
                # Regex sanitize class name
                safe_class_name = re.sub(r'[^a-zA-Z0-9_\-]', '', new_class.strip())
                if safe_class_name and safe_class_name not in st.session_state.classes:
                    st.session_state.classes.append(safe_class_name)
                    st.toast(f"Category '{safe_class_name}' added!", icon="➕")
                    st.rerun()
                elif safe_class_name in st.session_state.classes:
                    st.warning("Category already exists.")
                else:
                    st.error("Invalid category name.")
            else:
                st.error("Please enter a category name.")

        # Active Categories Grid Display
        if st.session_state.classes:
            st.markdown("<div style='margin-top: 20px; margin-bottom: 10px; font-weight: 600; color: #94A3B8;'>Active Categories</div>", unsafe_allow_html=True)
            cols = st.columns(min(len(st.session_state.classes), 3))
            for idx, c in enumerate(st.session_state.classes):
                col_target = cols[idx % 3]
                count = class_counts.get(c, 0)
                
                with col_target:
                    st.markdown(f"""
                        <div class='glass-card' style='padding: 16px; text-align: center; border-radius: 12px; margin-bottom: 10px;'>
                            <div class='card-title' style='justify-content: center; font-size: 1.1rem; margin-bottom: 10px;'>
                                {c}
                            </div>
                            <span class='sample-badge'>{count} samples</span>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Draw PCA dataset separation chart if we have at least 3 total samples
            total_samples = sum(class_counts.values())
            if total_samples >= 3:
                st.markdown("---")
                st.markdown("#### 📊 Dataset Latent Space (PCA Projection)")
                st.markdown("Projects high-dimensional CNN feature vectors to 2D to show dataset clustering quality.")
                
                try:
                    sel_backbone = st.session_state.get("backbone_opt", "MobileNetV3")
                    pca_res = requests.get(f"{BACKEND_URL}/features-pca", params={"backbone_name": sel_backbone})
                    if pca_res.status_code == 200:
                        pca_data = pca_res.json()
                        if pca_data:
                            pca_df = pd.DataFrame(pca_data)
                            st.scatter_chart(
                                pca_df,
                                x="x",
                                y="y",
                                color="class",
                                size=100,
                                use_container_width=True
                            )
                        else:
                            st.info("Insufficient samples processed to run PCA projection.")
                    else:
                        st.error("Error retrieving PCA coordinates.")
                except Exception as e:
                    st.error(f"Error computing PCA plot: {str(e)}")

        st.markdown("---")
        st.markdown("### 📷 2. Upload / Capture Training Samples")
        
        if not st.session_state.classes:
            st.info("💡 Add at least one category above to start collecting training samples.")
        else:
            # Let user select which class they are uploading to
            selected_class = st.selectbox(
                "Select category to add samples to:",
                options=st.session_state.classes
            )
            
            # Sub-tabs for input selection
            upload_tab, camera_tab = st.tabs(["📁 Upload Images", "📷 Live Webcam Captures"])
            
            with upload_tab:
                uploaded_files = st.file_uploader(
                    f"Choose images for '{selected_class}':",
                    type=["jpg", "jpeg", "png", "webp", "bmp"],
                    accept_multiple_files=True,
                    key=f"upload_{selected_class}"
                )
                
                if st.button(f"📥 Save Uploaded Images to '{selected_class}'", use_container_width=True, key=f"btn_upload_{selected_class}"):
                    if uploaded_files:
                        saved_count = 0
                        with st.spinner(f"Uploading files to '{selected_class}'..."):
                            for file in uploaded_files:
                                try:
                                    img_bytes = file.read()
                                    files_payload = {"files": (file.name, img_bytes, file.type)}
                                    data_payload = {"class_name": selected_class}
                                    
                                    res = requests.post(f"{BACKEND_URL}/upload-sample", files=files_payload, data=data_payload)
                                    if res.status_code == 200:
                                        saved_count += 1
                                except Exception as e:
                                    st.error(f"Failed to upload {file.name}: {str(e)}")
                        
                        if saved_count > 0:
                            st.success(f"Successfully uploaded {saved_count} sample(s) to category '{selected_class}'!")
                            time.sleep(1)
                            st.rerun()
                    else:
                        st.error("Please choose files to upload first.")
            
            with camera_tab:
                st.markdown("#### 📸 Continuous Streaming & Burst Capture")
                st.warning("⚠️ **Webcam Sandboxing Limit:** Due to modern browser security policies, custom HTML5 webcam components embedded in Streamlit iframes are often blocked from accessing the camera. If you see a 'Camera Blocked' error, please use the **Manual Backup Capture** below, which uses the native Streamlit camera input and works perfectly.")
                st.markdown("Use the premium high-speed capture engine below to record samples. Burst mode will take 10 consecutive frames at 500ms intervals.")
                
                # HTML5 camera burst script
                camera_burst_html = f"""
                <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.06); padding: 16px; border-radius: 12px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: white;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <span style="font-weight: 600; font-size: 1rem; color: #00E5FF;">📷 Continuous Webcam Stream</span>
                        <span id="burst-status" style="font-size: 0.85rem; color: #94A3B8; font-weight: bold;">Ready</span>
                    </div>
                    
                    <video id="webcam-feed" autoplay playsinline style="width: 100%; border-radius: 8px; border: 1px solid rgba(0, 229, 255, 0.2); background: #000; height: 200px; object-fit: cover; transform: scaleX(-1);"></video>
                    <canvas id="capture-canvas" style="display: none;" width="640" height="480"></canvas>
                    
                    <div style="display: flex; gap: 8px; margin-top: 12px;">
                        <button id="btn-snapshot" style="flex: 1; padding: 10px; background: linear-gradient(135deg, #00C6FF 0%, #0072FF 100%); border: none; border-radius: 8px; color: white; font-weight: bold; cursor: pointer; transition: opacity 0.2s;">📸 Single Shot</button>
                        <button id="btn-burst" style="flex: 1; padding: 10px; background: linear-gradient(135deg, #7f00ff 0%, #e100ff 100%); border: none; border-radius: 8px; color: white; font-weight: bold; cursor: pointer; transition: opacity 0.2s;">🚀 Burst (10 Shots)</button>
                    </div>
                </div>

                <script>
                    const video = document.getElementById('webcam-feed');
                    const canvas = document.getElementById('capture-canvas');
                    const ctx = canvas.getContext('2d');
                    const statusText = document.getElementById('burst-status');
                    const btnSnapshot = document.getElementById('btn-snapshot');
                    const btnBurst = document.getElementById('btn-burst');
                    
                    const backendUrl = "{BACKEND_URL}";
                    const className = "{selected_class}";

                    // Initialize webcam
                    navigator.mediaDevices.getUserMedia({{ video: true }})
                        .then(stream => {{
                            video.srcObject = stream;
                        }})
                        .catch(err => {{
                            console.error("Camera access failed: ", err);
                            statusText.textContent = "Camera Blocked";
                            statusText.style.color = "#FF4B4B";
                        }});

                    function uploadFrame(filename) {{
                        ctx.save();
                        ctx.translate(canvas.width, 0);
                        ctx.scale(-1, 1);
                        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                        ctx.restore();
                        
                        return new Promise((resolve, reject) => {{
                            canvas.toBlob(blob => {{
                                const formData = new FormData();
                                formData.append('files', blob, filename);
                                formData.append('class_name', className);
                                
                                fetch(backendUrl + '/upload-sample', {{
                                    method: 'POST',
                                    body: formData
                                }})
                                .then(res => res.json())
                                .then(data => resolve(data))
                                .catch(err => reject(err));
                            }}, 'image/jpeg', 0.9);
                        }});
                    }}

                    btnSnapshot.onclick = () => {{
                        statusText.textContent = "Saving...";
                        uploadFrame('snapshot_' + Date.now() + '.jpg')
                            .then(data => {{
                                statusText.textContent = "Saved!";
                                setTimeout(() => {{
                                    window.parent.location.reload();
                                }}, 600);
                            }})
                            .catch(err => {{
                                statusText.textContent = "Upload Failed";
                                console.error(err);
                            }});
                    }};

                    btnBurst.onclick = async () => {{
                        btnSnapshot.disabled = true;
                        btnBurst.disabled = true;
                        btnBurst.style.opacity = '0.5';
                        
                        const totalShots = 10;
                        for (let i = 0; i < totalShots; i++) {{
                            statusText.textContent = `Capturing ${{i + 1}}/${{totalShots}}...`;
                            try {{
                                await uploadFrame(`burst_${{i + 1}}_${{Date.now()}}.jpg`);
                            }} catch(err) {{
                                console.error("Burst frame failed: ", err);
                            }}
                            await new Promise(r => setTimeout(r, 500));
                        }}
                        
                        statusText.textContent = "Done! Reloading...";
                        setTimeout(() => {{
                            window.parent.location.reload();
                        }}, 800);
                    }};
                </script>
                """
                components.html(camera_burst_html, height=330, scrolling=False)
                
                st.markdown("*(If the video stream is not loading due to browser sandboxing, use the manual backup capture console below)*")
                camera_file = st.camera_input(f"Manual Backup Capture for '{selected_class}':", key=f"cam_{selected_class}")
                if camera_file:
                    img_bytes = camera_file.read()
                    if st.button(f"📸 Save Manual Snapshot to '{selected_class}'", use_container_width=True):
                        try:
                            files_payload = {"files": ("webcam_capture.jpg", img_bytes, "image/jpeg")}
                            data_payload = {"class_name": selected_class}
                            res = requests.post(f"{BACKEND_URL}/upload-sample", files=files_payload, data=data_payload)
                            if res.status_code == 200:
                                st.success("Snapshot saved successfully!")
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
        
        # Hyperparameter Selection controls
        st.markdown("#### ⚙️ Hyperparameter Playground:")
        classifier_opt = st.selectbox(
            "🤖 Classifier Algorithm",
            options=["LogisticRegression", "SVM", "RandomForest", "KNN"],
            index=0,
            help="LogisticRegression is fast and interpretable. SVM handles non-linear boundaries. RandomForest is robust. KNN is simple and non-parametric."
        )
        backbone_opt = st.selectbox(
            "Visual Feature Extractor (Backbone)",
            options=["MobileNetV3", "ResNet18"],
            index=0,
            help="MobileNetV3 is highly optimized and fast. ResNet18 is deeper and provides larger embeddings."
        )
        col_c, col_pen = st.columns(2)
        with col_c:
            c_val = st.slider("Regularizer C", min_value=0.01, max_value=10.0, value=1.0, step=0.1, help="Inverse of regularization strength. Smaller values specify stronger regularization.")
        with col_pen:
            penalty_opt = st.radio("Penalty constraint", ["L2 (Ridge)", "L1 (Lasso)"], index=0, help="L2 penalty uses weight squaring. L1 penalty enforces sparsity.")
            penalty_val = "l1" if "L1" in penalty_opt else "l2"
        
        # SVM and tree classifiers ignore penalty — clarify to user
        if classifier_opt in ["SVM", "RandomForest", "KNN"]:
            st.info(f"ℹ️ `{classifier_opt}` ignores L1/L2 penalty. The C value and Backbone selection still apply.")

        st.markdown("")
        
        # Train Button Trigger
        if st.button("🚂 Train Custom Model", use_container_width=True, disabled=not ready_to_train, type="primary"):
            with st.spinner(f"Extracting features using {backbone_opt} and training {classifier_opt}..."):
                try:
                    payload = {
                        "backbone_name": backbone_opt,
                        "classifier_type": classifier_opt,
                        "c_value": c_val,
                        "penalty": penalty_val
                    }
                    res = requests.post(f"{BACKEND_URL}/train", data=payload)
                    if res.status_code == 200:
                        data = res.json()
                        st.success(f"🎉 Custom classifier ({classifier_opt}) trained successfully!")
                        st.session_state.is_trained = True
                        st.session_state.prediction_history = []
                        st.session_state.train_summary = data.get("details", {})
                        st.balloons()
                        time.sleep(2)
                        st.rerun()
                    else:
                        detail = res.json().get("detail", "Unknown server-side training error")
                        st.error(f"Training Failed: {detail}")
                except Exception as e:
                    st.error(f"Error triggering training: {str(e)}")

        # Display Validation & Training Report
        if "train_summary" in st.session_state and st.session_state.train_summary:
            summary = st.session_state.train_summary
            val_metrics = summary.get("validation_metrics", {})
            
            st.markdown("---")
            st.markdown("### 📊 Model Performance Report")
            
            if val_metrics and val_metrics.get("split_executed"):
                st.success("🤖 Stratified Validation Split Executed (80% Train, 20% Val)")
                
                # Metrics cards
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Val Accuracy", f"{val_metrics.get('accuracy')}%")
                col2.metric("Precision", f"{val_metrics.get('precision')}%")
                col3.metric("Recall", f"{val_metrics.get('recall')}%")
                col4.metric("F1-Score", f"{val_metrics.get('f1_score')}%")
                
                # Confusion matrix display
                st.markdown("#### 🎯 Validation Confusion Matrix")
                cm_data = val_metrics.get("confusion_matrix", {})
                labels = cm_data.get("labels", [])
                matrix = cm_data.get("matrix", [])
                
                cm_df = pd.DataFrame(matrix, index=[f"Actual {l}" for l in labels], columns=[f"Predicted {l}" for l in labels])
                st.dataframe(cm_df, use_container_width=True)
            else:
                warning_msg = val_metrics.get("warning", "No validation split executed.") if val_metrics else "Low sample count: all images used for training."
                st.warning(f"⚠️ {warning_msg}")
                st.info(f"Total samples trained: {summary.get('samples_trained', 0)}")

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
                            pred_json = res.json()
                            st.session_state.last_prediction = pred_json
                            
                            # Log prediction details to backend analytics
                            try:
                                best_class = pred_json.get("predicted_class", "Unknown")
                                confidence = pred_json.get("probabilities", {}).get(best_class, 0.0)
                                log_payload = {
                                    "predicted_class": best_class,
                                    "confidence": confidence,
                                    "probabilities": pred_json.get("probabilities", {}),
                                    "backbone_used": pred_json.get("backbone_used")
                                }
                                requests.post(f"{BACKEND_URL}/log-prediction", json=log_payload, timeout=2)
                            except Exception:
                                pass
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
                        st.image(load_b64_image(pred_data["saliency_image"]), use_column_width=True, caption="Gradient-based Saliency Attention Map")
                    else:
                        st.info("No saliency heatmap metadata returned from model.")

                # 3. Render Latency & Hardware Telemetry Cards
                col_lat, col_dev = st.columns(2)
                with col_lat:
                    st.metric("Inference Latency", f"{pred_data.get('inference_time_ms', 0)} ms")
                with col_dev:
                    st.metric("Computation Engine", f"CPU ({pred_data.get('backbone_used', 'MobileNetV3')})")

                st.markdown(f"""
                    <div class='winner-banner' style='margin-top: 15px;'>
                        ✨ Best Prediction: {winner}
                    </div>
                """, unsafe_allow_html=True)

                # Sort probabilities by score descending
                sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

                # Low-confidence warning for closed-set validation
                if len(sorted_probs) >= 2:
                    top_prob = sorted_probs[0][1]
                    if top_prob < 65.0:
                        st.warning("⚠️ **Low Confidence / Unknown Object:** The model is uncertain about this input. Because the model was only trained to distinguish between your active categories, any new object (like a human or blank wall) is forced into the closest match. To solve this, add a **'Background'** category containing random environment samples.")

                st.markdown("<h4 style='margin-top: 20px; margin-bottom: 15px;'>Confidence Metrics</h4>", unsafe_allow_html=True)

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

with tab_theory:
    st.markdown("### 📖 Artificial Intelligence & Machine Learning Theory Hub")
    st.markdown("Explore core ML concepts, interact with simulators (Gradient Descent & Overfitting), and test your knowledge directly inside the dashboard.")
    
    # Inline and load index.html from learning_hub
    try:
        html_path = "learning_hub/index.html"
        css_path = "learning_hub/styles.css"
        js_path = "learning_hub/script.js"
        
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        with open(js_path, "r", encoding="utf-8") as f:
            js_content = f.read()
            
        # Programmatic inlining of CSS and JS
        inlined_html = html_content.replace(
            '<link rel="stylesheet" href="styles.css">',
            f'<style>{css_content}</style>'
        ).replace(
            '<script src="script.js"></script>',
            f'<script>{js_content}</script>'
        )
        
        # Point relative image paths to local absolute path for iframe resolution
        workspace_dir = os.path.abspath("learning_hub")
        workspace_dir_url = workspace_dir.replace("\\", "/")
        inlined_html = inlined_html.replace('src="assets/', f'src="file:///{workspace_dir_url}/assets/')
        
        # Render using Streamlit component
        import streamlit.components.v1 as components
        components.html(inlined_html, height=900, scrolling=True)
    except Exception as e:
        st.error(f"Failed to embed Learning Hub: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICS TAB (appended after theory tab)
# ─────────────────────────────────────────────────────────────────────────────
try:
    with tab_analytics:
        st.markdown("### 📊 Prediction Analytics Dashboard")
        st.markdown("Real-time summary of all inference sessions. Data is logged automatically after each prediction.")

        if not st.session_state.get("backend_active", False):
            st.warning("Backend must be connected to view analytics.")
        else:
            try:
                res = requests.get(f"{BACKEND_URL}/analytics", timeout=3)
                if res.status_code == 200:
                    data = res.json()
                    total = data.get("total_predictions", 0)
                    dist = data.get("class_distribution", {})
                    avg_conf = data.get("avg_confidence", {})
                    recents = data.get("recent_predictions", [])

                    if total == 0:
                        st.info("No predictions logged yet. Run some inferences in the 🔮 Inference Engine tab first!")
                    else:
                        st.metric("Total Predictions Logged", total)
                        st.markdown("---")

                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown("#### 🗂️ Class Prediction Distribution")
                            if dist:
                                import pandas as pd
                                dist_df = pd.DataFrame(list(dist.items()), columns=["Class", "Count"])
                                st.bar_chart(dist_df.set_index("Class"))

                        with col_b:
                            st.markdown("#### 🎯 Average Confidence per Class")
                            if avg_conf:
                                conf_df = pd.DataFrame(list(avg_conf.items()), columns=["Class", "Avg Confidence (%)"])
                                st.bar_chart(conf_df.set_index("Class"))

                        st.markdown("---")
                        st.markdown("#### 🕒 Last 10 Predictions")
                        if recents:
                            import pandas as pd
                            log_df = pd.DataFrame([{
                                "Time": e.get("timestamp", "")[:19].replace("T", " "),
                                "Predicted": e.get("predicted_class", ""),
                                "Confidence": f"{e.get('confidence', 0):.1f}%",
                                "Backbone": e.get("backbone", "")
                            } for e in reversed(recents)])
                            st.dataframe(log_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("Could not load analytics data from backend.")
            except Exception as ex:
                st.error(f"Analytics error: {ex}")
except Exception:
    pass  # tab_analytics may not be defined in older versions
