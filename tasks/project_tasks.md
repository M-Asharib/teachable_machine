# Teachable Machine: Phase-Wise Detailed Task List

This file tracks the developmental milestones of the decoupled Teachable Machine project. Mark items as complete (`[x]`) as we progress.

---


## Phase 0: UX Benchmark & Reference Study
Before writing any code, experience the target UX to understand exactly what we are building toward.

- [ ] **Task 0.1: Experience Google's Teachable Machine Firsthand**
  * Open the official benchmark URL: `https://teachablemachine.withgoogle.com/train/image`
  * Perform a full end-to-end test:
    * Define at least 2 custom class names.
    * Collect 10+ webcam samples per class.
    * Click "Train Model" and observe the instant training speed.
    * Switch to the preview panel and test accuracy with live camera input.
  * Observe and note down: the UI layout, the progress bar style, the performance meter behavior.
- [ ] **Task 0.2: Define UI Parity Goals**
  * Write down exactly which visual elements from the Google tool we must replicate:
    * Class naming cards.
    * Sample count display per class.
    * "Train Model" button style and loader behavior.
    * Live confidence bar chart output.

---

## Phase 1: Environment & Directory Setup
Set up the structural skeleton of the project and ensure all machine learning and web dependencies are configured correctly.

- [ ] **Task 1.1: Create Project Structure**
  * Create `backend/` and `frontend/` folders under the root workspace.
  * Create a placeholder `backend/dataset/` directory (required for the `os` module folder-scan logic later).
- [ ] **Task 1.2: Define Requirements**
  * Create `requirements.txt` on the root of the project with all exact library names:
    * `fastapi`
    * `uvicorn[standard]`
    * `python-multipart`
    * `streamlit`
    * `requests`
    * `torch` (lightweight CPU-only version)
    * `torchvision` (lightweight CPU-only version)
    * `scikit-learn`
    * `numpy`
    * `pillow`
- [ ] **Task 1.3: Verification of Python Environment**
  * Validate that Python 3.9+ is available via `python --version`.
  * Install dependencies using `pip install -r requirements.txt`.
  * Confirm PyTorch can be imported: run `python -c "import torch; print(torch.__version__)"`.
  * Confirm Streamlit can launch: run `streamlit hello`.

---

## Phase 2: FastAPI Backend Core & Image Ingestion
Implement the core FastAPI infrastructure and write the endpoint to accept and store custom training samples.

- [ ] **Task 2.1: Implement Server Base (`backend/main.py`)**
  * Initialize the FastAPI application instance.
  * Add standard `CORSMiddleware` to allow all origins (required for cross-domain Streamlit communication).
  * Add a basic health-check endpoint `GET /` returning `{"status": "Teachable Machine API is running"}`.
- [ ] **Task 2.2: Implement Ingestion API (`POST /upload-sample`)** *(Explicit guide requirements applied)*
  * Accept form parameters: `class_name` (string) and `files` (list of `UploadFile`).
  * Sanitize `class_name` to strip spaces and special characters (make safe for directory naming).
  * **Use Python's built-in `os` module** (as mandated by the project guide) to:
    * Check if a folder named `backend/dataset/{class_name}/` exists.
    * If it doesn't, create it automatically with `os.makedirs()`.
  * **Use the `uuid` module** (as mandated by the project guide) to generate randomized filenames:
    * Format: `str(uuid.uuid4()) + file_extension` (e.g., `a7d8c2b5-4e8c-8f92.jpg`).
    * This prevents any file overwriting between uploads.
  * Write each uploaded file's raw bytes to disk inside the correct class folder.
  * Return a JSON response: `{"message": "Uploaded {n} images to class '{class_name}'"}`.
- [ ] **Task 2.3: Verification of Ingestion**
  * Start the FastAPI server: `uvicorn backend.main:app --reload --port 8000`.
  * Send a mock upload using `curl` or a Python test script.
  * Inspect the `backend/dataset/` directory and confirm:
    * A class subfolder was created with the correct name.
    * Each image file has a unique UUID-based filename.

---

## Phase 3: Transfer Learning Training Engine
Implement the machine learning pipeline that extracts deep visual features and trains a custom classification head on the fly.

- [ ] **Task 3.1: Initialize PyTorch MobileNetV3 Backbone (`backend/ml_engine.py`)**
  * Import and load `mobilenet_v3_small` from `torchvision.models` with pretrained `DEFAULT` weights.
  * Set the model to evaluation mode using `.eval()` — this disables Dropout and BatchNorm training behavior.
  * Freeze all parameter gradients using `param.requires_grad = False` (prevents any weights from updating).
  * Slice the model to remove its final classification head — keep only the feature extraction layers so the output is a raw 1D feature vector per image.
- [ ] **Task 3.2: Create the Mandatory Preprocessing Pipeline**
  * Write a reusable `preprocess_image(image_bytes)` function using `torchvision.transforms`:
    * Convert raw bytes → PIL Image (RGB mode).
    * Resize to exactly **224 × 224 pixels** (as mandated by the project guide).
    * Convert PIL Image → PyTorch Tensor.
    * Normalize using ImageNet channel statistics:
      * Means: `[0.485, 0.456, 0.406]`
      * Std Devs: `[0.229, 0.224, 0.225]`
  * ⚠️ **CRITICAL**: This exact same function must be used in both training AND inference — any difference will break prediction accuracy.
- [ ] **Task 3.3: Load Dataset & Extract Feature Vectors**
  * Use `os.listdir()` to scan all subdirectories inside `backend/dataset/`.
  * For each class folder, load all image files, run `preprocess_image()`, and pass them through the MobileNetV3 backbone using `torch.no_grad()`.
  * Collect all feature vectors into a numpy array `X` and corresponding integer labels into array `y`.
  * Build and store a `label_map` dictionary: `{0: "class_a", 1: "class_b", ...}`.
- [ ] **Task 3.4: Train and Save the Classifier**
  * Train a `sklearn.linear_model.LogisticRegression` model on `X` and `y`.
  * Bundle the trained classifier and the `label_map` into a single Python dictionary:
    ```python
    model_bundle = {"classifier": clf, "label_map": label_map}
    ```
  * Serialize and save using `pickle.dump()` to `backend/model.pkl`.
  * Return a training summary: classes found, number of samples per class, training duration in seconds.

---

## Phase 4: Training & Validation Endpoint
Expose the training logic as a secure POST endpoint and incorporate robust error handling.

- [ ] **Task 4.1: Integrate `POST /train` Endpoint**
  * Add `/train` route inside `backend/main.py`.
  * Call the training function from `backend/ml_engine.py`.
  * Return a success JSON response with training summary on completion.
- [ ] **Task 4.2: Add Strict Validation Checks (Graceful Error Control)** *(Mandated by project guide)*
  * Before training, count all class subfolders inside `backend/dataset/` that contain at least 1 image.
  * **Check 1**: If total distinct classes is **less than 2**, return `HTTP 400`:
    * Message: `"At least 2 distinct classes with images are required to train a model."`
  * **Check 2**: If `backend/dataset/` directory is empty or missing, return `HTTP 400`:
    * Message: `"No dataset found. Please upload images first."`
  * These errors must be readable and clean — never let the server crash or return a 500 error to the user.
- [ ] **Task 4.3: Prevent Server Lockup During Training**
  * Training must not block the server's ability to handle other requests.
  * Consider running training in a background thread if processing large datasets.

---

## Phase 5: Inference & Prediction API
Expose the trained model through a precise prediction endpoint that returns percentage probability scores.

- [ ] **Task 5.1: Create Prediction Core in `backend/ml_engine.py`**
  * Load `backend/model.pkl` and extract the `classifier` and `label_map`.
  * Accept raw image bytes, apply the **exact same** `preprocess_image()` function used during training.
  * Extract features using MobileNetV3 backbone (inside `torch.no_grad()`).
  * Use `classifier.predict_proba()` to get probability scores for all trained classes.
  * **Return format** *(as per project guide — percentage format)*:
    * `predicted_class`: the label with highest probability (string).
    * `probabilities`: a dictionary of `{class_name: percentage_score}` for every trained class.
    * Example: `{"predicted_class": "Cup", "probabilities": {"Cup": 87.3, "Hand": 12.7}}`
- [ ] **Task 5.2: Integrate `POST /predict` Endpoint**
  * Add `/predict` route in `backend/main.py` that accepts a single `UploadFile`.
  * Pass the image bytes to the prediction core and return the formatted result.
- [ ] **Task 5.3: Error Handling for Missing Model** *(Mandated by project guide)*
  * If `model.pkl` is not found on disk, return `HTTP 400`:
    * Message: `"No trained model found. Please train the model first."`
  * Never let the server crash with a 500 error on this common case.

---

## Phase 6: Streamlit Frontend Base & Data Collection
Build the user interface elements for naming custom classes, selecting image acquisition types, and sending samples to the backend.

- [ ] **Task 6.1: Initialize Streamlit Layout (`frontend/app.py`)**
  * Configure page settings: `st.set_page_config(page_title="Teachable Machine", layout="wide")`.
  * Inject custom CSS for premium dark mode, rounded card containers, gradient buttons, and smooth hover animations.
  * Build a styled header section with the project title and subtitle.
- [ ] **Task 6.2: Class & Dataset Management UI**
  * Add a text input for users to type custom class label names.
  * Add an "Add Class" button that appends the new label to the active class list stored in `st.session_state.classes`.
  * Display all active classes in a visual card grid showing the class name and the current image sample count.
- [ ] **Task 6.3: Multi-Method Sample Gathering**
  * Add a selector to toggle between two input modes: **File Uploader** or **Webcam Capture**.
  * In File mode: use `st.file_uploader(accept_multiple_files=True)` to support bulk image drag-and-drop.
  * In Webcam mode: use `st.camera_input()` to capture a single live frame from the user's camera.
  * When the user confirms an upload action, use the `requests` library to POST the image(s) and `class_name` to `http://localhost:8000/upload-sample`.
  * Update the sample count display in real-time after each successful upload.

---

## Phase 7: Frontend Training & Prediction Interface
Wire up the exact Streamlit components named in the project guide and manage state correctly.

- [ ] **Task 7.1: Training Action Component**
  * Use **`st.button("Train Model")`** *(as explicitly named in the project guide)* to trigger the training flow.
  * Show `st.spinner("Training in progress...")` while the POST request to `/train` is running.
  * On success, display `st.success("Model trained successfully!")`.
  * On failure (e.g., less than 2 classes), display the backend error using `st.error(message)`.
- [ ] **Task 7.2: Session State Management** *(Mandated: "State Management Checkpoints")*
  * Initialize `st.session_state.is_trained = False` on app start.
  * Set `st.session_state.is_trained = True` only after a successful `/train` API response.
  * **Do NOT render** any prediction or testing components until `is_trained` is `True` — this keeps the dashboard clean and professional.
- [ ] **Task 7.3: State-Gated Live Prediction Interface**
  * Only render this section when `st.session_state.is_trained == True`.
  * Provide two testing input options: webcam snapshot or file upload.
  * On "Predict" action, POST the image to `http://localhost:8000/predict` using `requests`.
  * Display results using **both** *(as named in guide)*:
    * `st.progress(value)` — a styled progress bar per class showing percentage confidence.
    * `st.bar_chart(data)` — a clean horizontal bar chart of all class probabilities.
  * Highlight the top predicted class with a prominent success banner.

---

## Phase 8: Robustness Testing & Refinement
Iron out any remaining system bugs, optimize performance, and write full documentation.

- [ ] **Task 8.1: Conduct End-to-End Integration Tests**
  * Verify the complete user journey from start to finish:
    * Add 2+ class names → Upload 5+ samples per class → Train → Predict with new image.
  * Verify Streamlit UI stays interactive and never freezes during training.
  * Verify the "Add Sample" counter updates correctly after each webcam capture and file upload.
- [ ] **Task 8.2: Test All Graceful Error Controls** *(Mandated by project guide)*
  * Test: Click "Train Model" with zero classes — expect clean error banner, no crash.
  * Test: Click "Train Model" with only 1 class — expect readable 400 error message.
  * Test: Click "Predict" before a model is trained — confirm the test panel is hidden by `session_state`.
  * Test: Send a corrupt/unsupported image file to `/predict` — expect a clean error, not a 500 crash.
- [ ] **Task 8.3: Backend Performance Optimization**
  * Cache the loaded `model.pkl` in server memory (e.g., use a module-level variable) so it is not reloaded from disk on every single `/predict` call.
  * Cache the MobileNetV3 backbone model in memory at server startup, not per-request.
- [ ] **Task 8.4: Add Reset Functionality**
  * Add a "Reset Everything" button in the Streamlit sidebar.
  * On click, call a new `DELETE /reset` endpoint that clears the `backend/dataset/` directory and deletes `backend/model.pkl`.
  * Reset `st.session_state.is_trained = False` and clear the class list.
- [ ] **Task 8.5: Write README.md** *(Specific content required)*
  * Section 1: Project overview and architecture diagram (two services).
  * Section 2: Prerequisites (Python 3.9+, pip, webcam for webcam mode).
  * Section 3: Installation — `pip install -r requirements.txt`.
  * Section 4: Running the backend — `uvicorn backend.main:app --reload --port 8000`.
  * Section 5: Running the frontend — `streamlit run frontend/app.py`.
  * Section 6: Full user workflow walkthrough with screenshots.

---

## Phase 9: Docker & Containerization
*"Final Challenge Tip"* from the project guide: Pack both services into Dockerfiles and launch with a single `docker-compose.yml`.

- [ ] **Task 9.1: Write Backend Dockerfile (`backend/Dockerfile`)**
  * Use a Python 3.11 slim base image.
  * Copy `requirements.txt` and install dependencies.
  * Copy the `backend/` source code.
  * Expose port `8000`.
  * Set the startup command: `uvicorn main:app --host 0.0.0.0 --port 8000`.
- [ ] **Task 9.2: Write Frontend Dockerfile (`frontend/Dockerfile`)**
  * Use a Python 3.11 slim base image.
  * Install Streamlit and requests from `requirements.txt`.
  * Copy the `frontend/` source code.
  * Expose port `8501`.
  * Set the startup command: `streamlit run app.py --server.port 8501 --server.address 0.0.0.0`.
- [ ] **Task 9.3: Write `docker-compose.yml` on Root**
  * Define two services: `backend` and `frontend`.
  * Mount `backend/dataset/` as a persistent Docker volume so uploaded images survive container restarts.
  * Set the frontend service to depend on (`depends_on`) the backend service.
  * Map ports: `8000:8000` for backend, `8501:8501` for frontend.
- [ ] **Task 9.4: Verify Docker Build & Run**
  * Run `docker-compose up --build` and confirm both containers start without errors.
  * Open browser at `http://localhost:8501` and complete the full user journey inside Docker.
  * Confirm that uploaded dataset images are persisted via the volume between container restarts.
