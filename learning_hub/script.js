/* ==========================================
   AETHERML INTERACTIVE ENGINE (script.js)
   ========================================== */

document.addEventListener("DOMContentLoaded", () => {
    initParticles();
    initRoadmap();
    initGradientDescentSim();
    initOverfittingSim();
    initModals();
    initQuiz();
});

/* ==========================================
   1. FLOATING PARTICLE BACKGROUND
   ========================================== */
function initParticles() {
    const canvas = document.getElementById("particleCanvas");
    const ctx = canvas.getContext("2d");
    
    let particles = [];
    const particleCount = 65;

    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    window.addEventListener("resize", resize);
    resize();

    class Particle {
        constructor() {
            this.reset();
        }
        reset() {
            this.x = Math.random() * canvas.width;
            this.y = Math.random() * canvas.height;
            this.size = Math.random() * 2 + 0.5;
            this.speedX = Math.random() * 0.4 - 0.2;
            this.speedY = Math.random() * 0.4 - 0.2;
            this.opacity = Math.random() * 0.5 + 0.15;
        }
        update() {
            this.x += this.speedX;
            this.y += this.speedY;

            if (this.x < 0 || this.x > canvas.width || this.y < 0 || this.y > canvas.height) {
                this.reset();
            }
        }
        draw() {
            ctx.fillStyle = `rgba(0, 242, 254, ${this.opacity})`;
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    for (let i = 0; i < particleCount; i++) {
        particles.push(new Particle());
    }

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        particles.forEach(p => {
            p.update();
            p.draw();
        });
        requestAnimationFrame(animate);
    }
    animate();
}

/* ==========================================
   2. INTERACTIVE ROADMAP CONTROLLER
   ========================================== */
const ROADMAP_DATA = {
    foundations: {
        title: "Foundations of AI/ML",
        phase: "Phase 1: Math & Python Fundamentals",
        icon: "fa-calculator",
        subjects: [
            "Linear Algebra (Matrices, Matrix-Vector multiplication, Eigenvalues/Eigenvectors)",
            "Multivariable Calculus (Partial Derivatives, Gradients, Chain Rule)",
            "Probability & Inference (Bayes' Theorem, Standard Distributions, Maximum Likelihood)",
            "Scientific Python Programming (OOP, NumPy vectorization, Pandas DataFrames)"
        ],
        stack: ["Python", "NumPy", "Pandas", "Matplotlib", "Jupyter"],
        objective: "Establish the mathematical toolkit and programmatic agility required to read research papers, process structured datasets, and build custom regression modules from scratch."
    },
    classical: {
        title: "Classical Machine Learning",
        phase: "Phase 2: Supervised & Unsupervised Learning",
        icon: "fa-chart-line",
        subjects: [
            "Supervised Regressors (Ordinary Least Squares, Lasso/Ridge Regularization)",
            "Supervised Classifiers (Logistic Regression, Support Vector Machines, Random Forests)",
            "Unsupervised Clustering (K-Means Centroid placement, Hierarchical Agglomeration)",
            "Dimensionality Reduction (Principal Component Analysis (PCA), t-SNE projection)"
        ],
        stack: ["Scikit-Learn", "SciPy", "Seaborn", "Statsmodels"],
        objective: "Formulate statistical models, diagnose high bias vs high variance, perform feature engineering, and evaluate performance using precise metrics (Precision, Recall, ROC-AUC, Silhouette Scores)."
    },
    deep: {
        title: "Deep Learning & Vision",
        phase: "Phase 3: Neural Networks & Computer Vision",
        icon: "fa-network-wired",
        subjects: [
            "Neural Foundations (Multi-Layer Perceptrons, Backpropagation, Activations like ReLU/GELU)",
            "Convolutional Architectures (CNNs, Spatial Convolutions, Max Pooling, ResNet blocks)",
            "Computer Vision Tasks (Object Detection via YOLO, Semantic Segmentation, Transfer Learning)",
            "Optimization Algorithms (Adam, SGD with momentum, Learning Rate schedulers)"
        ],
        stack: ["PyTorch", "Torchvision", "TensorFlow", "Keras", "OpenCV"],
        objective: "Design deep neural layers, implement standard Convolution operations, leverage pre-trained backbone feature extractors (like MobileNetV3), and fine-tune classifiers on customized visual datasets."
    },
    generative: {
        title: "NLP & Generative AI",
        phase: "Phase 4: Sequence Processing & Transformers",
        icon: "fa-wand-magic-sparkles",
        subjects: [
            "Recurrent Paradigms (RNNs, Long Short-Term Memory (LSTM) cells)",
            "Transformer Architecture (Scaled Dot-Product Self-Attention, Multi-Head Queries, Positional Encoding)",
            "Large Language Models (Inference pipelines, Tokenization, Fine-Tuning using LoRA/QLoRA)",
            "Generative Visual Models (Diffusion Models, Variational Autoencoders, GANs)"
        ],
        stack: ["Hugging Face", "Transformers", "LangChain", "PyTorch", "Accelerate"],
        objective: "Understand self-attention weights, run large model weights local to hardware constraints, and implement prompt engineering, retrieval-augmented generation (RAG), and parameter-efficient fine-tuning."
    }
};

function initRoadmap() {
    const nodes = document.querySelectorAll(".roadmap-node");
    const detailsCard = document.getElementById("roadmapDetails");
    
    const detailsTitle = document.getElementById("detailsTitle");
    const detailsPhase = document.getElementById("detailsPhase");
    const detailsIcon = document.getElementById("detailsIcon");
    const detailsSubjects = document.getElementById("detailsSubjects");
    const detailsStack = document.getElementById("detailsStack");
    const detailsObjective = document.getElementById("detailsObjective");

    nodes.forEach(node => {
        node.addEventListener("click", () => {
            nodes.forEach(n => n.classList.remove("active"));
            node.classList.add("active");

            const nodeKey = node.getAttribute("data-node");
            const data = ROADMAP_DATA[nodeKey];
            if (!data) return;

            // Trigger animation on card
            detailsCard.style.animation = "none";
            // trigger reflow
            void detailsCard.offsetWidth;
            detailsCard.style.animation = "fadeIn 0.4s ease";

            // Update content
            detailsTitle.textContent = data.title;
            detailsPhase.textContent = data.phase;
            detailsIcon.innerHTML = `<i class="fa-solid ${data.icon}"></i>`;
            
            // Render subjects
            detailsSubjects.innerHTML = data.subjects.map(s => `<li>${s}</li>`).join("");
            
            // Render stack tags
            detailsStack.innerHTML = data.stack.map(st => `<span class="tag">${st}</span>`).join("");
            
            // Render objective
            detailsObjective.textContent = data.objective;
        });
    });
}

/* ==========================================
   3. GRADIENT DESCENT SIMULATOR
   ========================================== */
function initGradientDescentSim() {
    const canvas = document.getElementById("gdCanvas");
    const ctx = canvas.getContext("2d");

    const lrSlider = document.getElementById("lrSlider");
    const stepsSlider = document.getElementById("stepsSlider");
    const startXSlider = document.getElementById("startXSlider");
    
    const lrVal = document.getElementById("lrVal");
    const stepsVal = document.getElementById("stepsVal");
    const startXVal = document.getElementById("startXVal");
    
    const gdFeedback = document.getElementById("gdFeedback");

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        const w = canvas.width;
        const h = canvas.height;
        const padding = 30;

        // Coordinate transforms
        const mapX = x => padding + (x + 5) / 10 * (w - 2 * padding);
        const mapY = y => h - padding - (y / 25) * (h - 2 * padding);

        // Draw Axes
        ctx.strokeStyle = "rgba(255,255,255,0.15)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(mapX(-5), mapY(0));
        ctx.lineTo(mapX(5), mapY(0));
        ctx.moveTo(mapX(0), mapY(0));
        ctx.lineTo(mapX(0), mapY(25));
        ctx.stroke();

        // Draw curve: y = x^2
        ctx.strokeStyle = "#7f00ff";
        ctx.lineWidth = 3.5;
        ctx.beginPath();
        for (let x = -5; x <= 5; x += 0.1) {
            const y = x * x;
            if (x === -5) ctx.moveTo(mapX(x), mapY(y));
            else ctx.lineTo(mapX(x), mapY(y));
        }
        ctx.stroke();

        // Fetch user hyperparams
        const lr = parseFloat(lrSlider.value);
        const steps = parseInt(stepsSlider.value);
        let currentX = parseFloat(startXSlider.value);
        
        lrVal.textContent = lr;
        stepsVal.textContent = steps;
        startXVal.textContent = currentX;

        // Optimize: Gradient Descent Loop
        // Cost = X^2 -> Grad = 2*X
        let path = [{x: currentX, y: currentX * currentX}];
        let status = "success";
        let text = "Optimal parameters reached smoothly.";

        for (let i = 0; i < steps; i++) {
            const grad = 2 * currentX;
            currentX = currentX - lr * grad;
            
            // Check for divergence
            if (Math.abs(currentX) > 10) {
                status = "error";
                text = "Divergence occurred! Learning rate is too high.";
                break;
            }
            path.push({x: currentX, y: currentX * currentX});
        }

        // Check convergence status
        const lastX = path[path.length - 1].x;
        if (status !== "error") {
            if (Math.abs(lastX) < 0.05) {
                status = "success";
                text = "Converged! Cost minimized to 0.";
            } else {
                status = "warning";
                text = "Slow convergence. Try increasing iterations or learning rate.";
            }
        }

        // Draw optimization path
        ctx.strokeStyle = "rgba(0, 242, 254, 0.7)";
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(mapX(path[0].x), mapY(path[0].y));
        for (let i = 1; i < path.length; i++) {
            ctx.lineTo(mapX(path[i].x), mapY(path[i].y));
        }
        ctx.stroke();
        ctx.setLineDash([]); // Reset dash

        // Draw descent steps (nodes)
        path.forEach((pt, idx) => {
            ctx.beginPath();
            ctx.arc(mapX(pt.x), mapY(pt.y), idx === 0 ? 7 : 4, 0, Math.PI * 2);
            ctx.fillStyle = idx === 0 ? "#ffd700" : (idx === path.length - 1 ? "#00f2fe" : "rgba(0, 242, 254, 0.9)");
            ctx.fill();
        });

        // Update indicator UI
        const statusSpan = gdFeedback.querySelector(".status-indicator");
        const feedbackText = gdFeedback.querySelector(".feedback-text");
        
        statusSpan.className = `status-indicator ${status}`;
        if (status === "success") {
            statusSpan.textContent = "🟢 Converged";
        } else if (status === "warning") {
            statusSpan.textContent = "🟡 Under-optimized";
        } else {
            statusSpan.textContent = "🔴 Diverged";
        }
        feedbackText.textContent = text;
    }

    // Set dimensions on init
    function setCanvasDims() {
        const d = canvas.parentElement.getBoundingClientRect();
        canvas.width = d.width - 20;
        canvas.height = 250;
        draw();
    }

    [lrSlider, stepsSlider, startXSlider].forEach(slider => {
        slider.addEventListener("input", draw);
    });

    window.addEventListener("resize", setCanvasDims);
    // Let some time for rendering calculations
    setTimeout(setCanvasDims, 100);
}

/* ==========================================
   4. OVERFITTING & BIAS-VARIANCE SIMULATOR
   ========================================== */
function initOverfittingSim() {
    const canvas = document.getElementById("overfittingCanvas");
    const ctx = canvas.getContext("2d");

    const degreeSlider = document.getElementById("degreeSlider");
    const noiseSlider = document.getElementById("noiseSlider");
    const pointsSlider = document.getElementById("pointsSlider");
    
    const degreeVal = document.getElementById("degreeVal");
    const noiseVal = document.getElementById("noiseVal");
    const pointsVal = document.getElementById("pointsVal");
    
    const fittingFeedback = document.getElementById("fittingFeedback");

    // Static synthetic baseline data generation (independent of slider updates to keep curve persistent)
    let rawX = [];
    let rawY = [];
    const maxPoints = 50;

    // Generate base sine wave trend
    for (let i = 0; i < maxPoints; i++) {
        // x values distributed from -3 to 3
        const x = -3 + (i / (maxPoints - 1)) * 6;
        const y = Math.sin(x) * 4; // Amplitude 4
        rawX.push(x);
        rawY.push(y);
    }

    // Polynomial fitting math helper (Standard Least Squares solver)
    // Fits y = b0 + b1*x + b2*x^2 + ... + bd*x^d
    function solveOLS(xArr, yArr, degree) {
        const N = xArr.length;
        const K = degree + 1;

        // Build Design Matrix X (shape N x K)
        let X = [];
        for (let i = 0; i < N; i++) {
            let row = [];
            for (let j = 0; j < K; j++) {
                row.push(Math.pow(xArr[i], j));
            }
            X.push(row);
        }

        // Compute X_transpose (shape K x N)
        let XT = [];
        for (let j = 0; j < K; j++) {
            let row = [];
            for (let i = 0; i < N; i++) {
                row.push(X[i][j]);
            }
            XT.push(row);
        }

        // Compute A = XT * X (shape K x K)
        let A = [];
        for (let r = 0; r < K; r++) {
            let row = [];
            for (let c = 0; c < K; c++) {
                let sum = 0;
                for (let i = 0; i < N; i++) {
                    sum += XT[r][i] * X[i][c];
                }
                row.push(sum);
            }
            A.push(row);
        }

        // Compute B = XT * y (shape K x 1)
        let B = [];
        for (let r = 0; r < K; r++) {
            let sum = 0;
            for (let i = 0; i < N; i++) {
                sum += XT[r][i] * yArr[i];
            }
            B.push(sum);
        }

        // Solve OLS Matrix Equation: A * Coeffs = B using Gaussian Elimination
        const coeffs = gaussianElimination(A, B);
        return coeffs;
    }

    // Gaussian Elimination solver
    function gaussianElimination(A, B) {
        const n = B.length;
        // Make augmented matrix
        let M = [];
        for (let i = 0; i < n; i++) {
            M.push([...A[i], B[i]]);
        }

        for (let i = 0; i < n; i++) {
            // Find pivot
            let maxEl = Math.abs(M[i][i]);
            let maxRow = i;
            for (let k = i + 1; k < n; k++) {
                if (Math.abs(M[k][i]) > maxEl) {
                    maxEl = Math.abs(M[k][i]);
                    maxRow = k;
                }
            }

            // Swap maximum row
            let temp = M[maxRow];
            M[maxRow] = M[i];
            M[i] = temp;

            // Make pivot 1
            const pivot = M[i][i];
            if (Math.abs(pivot) < 1e-12) {
                // Return fallback zero coefficients if matrix is singular
                return new Array(n).fill(0);
            }
            for (let j = i; j <= n; j++) {
                M[i][j] /= pivot;
            }

            // Zero out values below/above pivot
            for (let k = 0; k < n; k++) {
                if (k !== i) {
                    const c = M[k][i];
                    for (let j = i; j <= n; j++) {
                        M[k][j] -= c * M[i][j];
                    }
                }
            }
        }

        // Extract last column
        return M.map(row => row[n]);
    }

    function evaluatePolynomial(x, coeffs) {
        let sum = 0;
        for (let i = 0; i < coeffs.length; i++) {
            sum += coeffs[i] * Math.pow(x, i);
        }
        return sum;
    }

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        const w = canvas.width;
        const h = canvas.height;
        const padding = 35;

        const mapX = x => padding + (x + 3.5) / 7 * (w - 2 * padding);
        const mapY = y => h / 2 - (y / 6) * (h / 2 - padding);

        // Grid
        ctx.strokeStyle = "rgba(255,255,255,0.05)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = -3; i <= 3; i++) {
            ctx.moveTo(mapX(i), padding);
            ctx.lineTo(mapX(i), h - padding);
        }
        ctx.stroke();

        // Fetch parameters
        const deg = parseInt(degreeSlider.value);
        const noise = parseFloat(noiseSlider.value) / 10; // Scale down noise
        const ptsCount = parseInt(pointsSlider.value);

        degreeVal.textContent = deg;
        noiseVal.textContent = Math.round(noise * 10);
        pointsVal.textContent = ptsCount;

        // Subsample points and add deterministic noise
        let samplesX = [];
        let samplesY = [];
        const step = Math.floor(maxPoints / ptsCount);
        
        for (let i = 0; i < ptsCount; i++) {
            const idx = Math.min(i * step, maxPoints - 1);
            const x = rawX[idx];
            // Deterministic noise using a pseudo-random sine formula
            const randNoise = Math.sin(idx * 45) * noise; 
            const y = rawY[idx] + randNoise;
            
            samplesX.push(x);
            samplesY.push(y);
        }

        // Fit polynomial model
        const coeffs = solveOLS(samplesX, samplesY, deg);

        // Draw Noisy Data Points
        samplesX.forEach((sx, idx) => {
            ctx.beginPath();
            ctx.arc(mapX(sx), mapY(samplesY[idx]), 4.5, 0, Math.PI * 2);
            ctx.fillStyle = "#ffd700";
            ctx.fill();
        });

        // Draw Fitted Curve
        ctx.strokeStyle = "#00f2fe";
        ctx.lineWidth = 3;
        ctx.beginPath();
        for (let x = -3.2; x <= 3.2; x += 0.05) {
            const fittedY = evaluatePolynomial(x, coeffs);
            if (x === -3.2) ctx.moveTo(mapX(x), mapY(fittedY));
            else ctx.lineTo(mapX(x), mapY(fittedY));
        }
        ctx.stroke();

        // Update feedback indicators
        let status = "success";
        let text = "Captures the underlying trend of the data with minimal noise.";

        if (deg === 1) {
            status = "error";
            text = "Underfitting (High Bias). A simple straight line cannot capture the sine curvature.";
        } else if (deg > 4) {
            status = "warning";
            text = "Overfitting (High Variance). Model is capturing noise rather than the core pattern.";
        }

        const statusSpan = fittingFeedback.querySelector(".status-indicator");
        const feedbackText = fittingFeedback.querySelector(".feedback-text");
        
        statusSpan.className = `status-indicator ${status}`;
        if (status === "success") {
            statusSpan.textContent = "🟢 Balanced Fit";
        } else if (status === "warning") {
            statusSpan.textContent = "🟡 Overfitting";
        } else {
            statusSpan.textContent = "🔴 Underfitting";
        }
        feedbackText.textContent = text;
    }

    function setCanvasDims() {
        const d = canvas.parentElement.getBoundingClientRect();
        canvas.width = d.width - 20;
        canvas.height = 250;
        draw();
    }

    [degreeSlider, noiseSlider, pointsSlider].forEach(slider => {
        slider.addEventListener("input", draw);
    });

    window.addEventListener("resize", setCanvasDims);
    setTimeout(setCanvasDims, 100);
}

/* ==========================================
   5. ARCHITECTURES DETAILS MODAL
   ========================================== */
const ARCH_MODAL_CONTENTS = {
    mlp: `
        <h3 class="gradient-text" style="font-size:1.8rem; margin-bottom:15px;"><i class="fa-solid fa-network-wired"></i> Multi-Layer Perceptron (MLP)</h3>
        <p style="margin-bottom:20px; color:#cfd8dc;">The basic structural block of deep neural networks. MLPs map an input vector to target labels through layers of linear matrix multiplications paired with non-linear activation functions.</p>
        
        <h4 style="margin-bottom:10px; color:#00f2fe;"><i class="fa-solid fa-calculator"></i> Mathematical Operations</h4>
        <p style="font-family: monospace; font-size: 0.95rem; background:rgba(0,0,0,0.4); padding:12px; border-radius:8px; border:1px solid rgba(255,255,255,0.05); margin-bottom:20px; line-height: 1.8;">
            <b>Forward Pass:</b><br>
            z<sup>[l]</sup> = W<sup>[l]</sup>a<sup>[l-1]</sup> + b<sup>[l]</sup><br>
            a<sup>[l]</sup> = &sigma;(z<sup>[l]</sup>)<br><br>
            <b>Activation (&sigma;):</b> ReLU(z) = max(0, z)
        </p>

        <h4 style="margin-bottom:10px; color:#00f2fe;"><i class="fa-solid fa-arrows-spin"></i> Error Backpropagation</h4>
        <p style="color:#cfd8dc; font-size:0.9rem; margin-bottom:15px;">By computing the derivative of a cost loss function J relative to every weight matrix W using the calculus <b>Chain Rule</b>, errors propagate backwards to update model weights:</p>
        <p style="font-family: monospace; font-size: 0.9rem; background:rgba(0,0,0,0.4); padding:10px; border-radius:8px; border:1px solid rgba(255,255,255,0.05);">
            W = W - &eta; * (&part;J / &part;W)
        </p>
    `,
    clustering: `
        <h3 class="gradient-text" style="font-size:1.8rem; margin-bottom:15px;"><i class="fa-solid fa-chart-pie"></i> Clustering Algorithms</h3>
        <p style="margin-bottom:20px; color:#cfd8dc;">Unsupervised learning organizes datasets into natural groups (clusters) without pre-defined labels. The model operates purely on spatial feature similarity.</p>
        
        <h4 style="margin-bottom:10px; color:#7f00ff;"><i class="fa-solid fa-map-pin"></i> K-Means Mechanics</h4>
        <p style="color:#cfd8dc; font-size:0.9rem; margin-bottom:15px;">K-Means positions K target centroids randomly and repeats two simple steps until convergence:</p>
        <ul style="list-style:none; padding-left:10px; margin-bottom:20px; font-size:0.9rem; color:#b0bec5; display:flex; flex-direction:column; gap:8px;">
            <li><span style="color:#7f00ff; font-weight:bold;">1. Assignment:</span> Map each data point to its closest centroid using Euclidean distance.</li>
            <li><span style="color:#7f00ff; font-weight:bold;">2. Update:</span> Recompute centroid coordinates as the arithmetic mean of all points assigned to that cluster.</li>
        </ul>

        <h4 style="margin-bottom:10px; color:#7f00ff;"><i class="fa-solid fa-arrows-left-right-to-line"></i> Dimensionality Reduction</h4>
        <p style="color:#cfd8dc; font-size:0.9rem;"><b>PCA (Principal Component Analysis):</b> Identifies orthogonal eigenvectors (principal components) along which data variance is maximized. This permits mapping a 500-dimensional space down to 2D/3D for visualization with minimal information loss.</p>
    `,
    transformer: `
        <h3 class="gradient-text" style="font-size:1.8rem; margin-bottom:15px;"><i class="fa-solid fa-wand-magic-sparkles"></i> Transformer Attention</h3>
        <p style="margin-bottom:20px; color:#cfd8dc;">Transformers replaced sequential processing (LSTMs) with a parallel attention system. Instead of checking words one-by-one, the model evaluates context relationships across all tokens simultaneously.</p>
        
        <h4 style="margin-bottom:10px; color:#ffd700;"><i class="fa-solid fa-gears"></i> Scaled Dot-Product Attention</h4>
        <p style="font-family: monospace; font-size: 0.95rem; background:rgba(0,0,0,0.4); padding:12px; border-radius:8px; border:1px solid rgba(255,255,255,0.05); margin-bottom:20px;">
            Attention(Q, K, V) = softmax( (Q * K<sup>T</sup>) / &radic;d<sub>k</sub> ) * V
        </p>

        <h4 style="margin-bottom:10px; color:#ffd700;"><i class="fa-solid fa-magnifying-glass"></i> Key Elements Explained</h4>
        <ul style="list-style:none; padding-left:10px; font-size:0.9rem; color:#b0bec5; display:flex; flex-direction:column; gap:8px;">
            <li><b>Query (Q):</b> What token details are looking for?</li>
            <li><b>Key (K):</b> What features does each other token offer?</li>
            <li><b>Value (V):</b> The actual content vectors.</li>
            <li><b>&radic;d<sub>k</sub>:</b> Scaling factor preventing vanishing gradients in Softmax for high dims.</li>
        </ul>
    `
};

function initModals() {
    const backdrop = document.getElementById("modalBackdrop");
    const contentArea = document.getElementById("modalContentArea");
    const openBtns = document.querySelectorAll(".open-modal-btn");
    const closeBtn = document.getElementById("closeModalBtn");

    openBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const key = btn.getAttribute("data-modal");
            const htmlContent = ARCH_MODAL_CONTENTS[key];
            if (!htmlContent) return;

            contentArea.innerHTML = htmlContent;
            backdrop.classList.add("active");
        });
    });

    closeBtn.addEventListener("click", () => {
        backdrop.classList.remove("active");
    });

    backdrop.addEventListener("click", (e) => {
        if (e.target === backdrop) {
            backdrop.classList.remove("active");
        }
    });
}

/* ==========================================
   6. KNOWLEDGE ASSESSMENT QUIZ
   ========================================== */
const QUIZ_QUESTIONS = [
    {
        question: "Which of the following activation functions outputs values strictly between 0 and 1?",
        options: ["ReLU", "Tanh", "Sigmoid", "Leaky ReLU"],
        correct: 2,
        explanation: "Sigmoid squashes any input value into a range between 0 and 1, making it ideal for probability representation."
    },
    {
        question: "In Gradient Descent, what happens if the learning rate is set too high?",
        options: [
            "The model converges very slowly.",
            "The weights might oscillate or diverge entirely.",
            "The loss function will instantly hit zero.",
            "The gradients automatically freeze."
        ],
        correct: 1,
        explanation: "Large step sizes can overshoot the cost function minimum, causing oscillation or divergence away from optimal parameters."
    },
    {
        question: "Which ML paradigm does K-Means clustering belong to?",
        options: [
            "Supervised Learning",
            "Unsupervised Learning",
            "Reinforcement Learning",
            "Semi-supervised Learning"
        ],
        correct: 1,
        explanation: "K-Means works on unlabeled datasets to find patterns purely through point proximity, making it Unsupervised."
    },
    {
        question: "What is the primary function of the Self-Attention mechanism in Transformers?",
        options: [
            "To speed up images convolution operations.",
            "To map visual features to text tags.",
            "To relate distant tokens in a sequence to construct semantic context.",
            "To compress the dataset dimensions."
        ],
        correct: 2,
        explanation: "Self-Attention calculates a score matrix linking every token to every other token, capturing contextual relationships regardless of distance."
    },
    {
        question: "What does high bias in a machine learning model typically indicate?",
        options: [
            "The model is overfitting.",
            "The training error is extremely low.",
            "The model is too simple (underfitting) to capture the trend.",
            "The dataset has a lot of noise."
        ],
        correct: 2,
        explanation: "High bias represents strong assumptions in the model design, causing it to underfit the target trends."
    }
];

function initQuiz() {
    const startView = document.getElementById("quizStartView");
    const questionView = document.getElementById("quizQuestionView");
    const resultView = document.getElementById("quizResultView");

    const startBtn = document.getElementById("startQuizBtn");
    const nextBtn = document.getElementById("nextQuestionBtn");
    const restartBtn = document.getElementById("restartQuizBtn");

    const progressFill = document.getElementById("quizProgressFill");
    const questionCounter = document.getElementById("questionCounter");
    const currentScoreVal = document.getElementById("currentScoreVal");
    const questionText = document.getElementById("questionText");
    const optionsContainer = document.getElementById("optionsContainer");
    
    const finalScoreVal = document.getElementById("finalScoreVal");
    const resultFeedbackText = document.getElementById("resultFeedbackText");
    const resultIcon = document.getElementById("resultIcon");

    let currentIdx = 0;
    let score = 0;
    let hasAnswered = false;

    startBtn.addEventListener("click", () => {
        startView.classList.add("hidden");
        questionView.classList.remove("hidden");
        loadQuestion();
    });

    nextBtn.addEventListener("click", () => {
        currentIdx++;
        if (currentIdx < QUIZ_QUESTIONS.length) {
            loadQuestion();
        } else {
            showResults();
        }
    });

    restartBtn.addEventListener("click", () => {
        currentIdx = 0;
        score = 0;
        resultView.classList.add("hidden");
        questionView.classList.remove("hidden");
        loadQuestion();
    });

    function loadQuestion() {
        hasAnswered = false;
        nextBtn.classList.add("hidden");
        optionsContainer.innerHTML = "";

        const q = QUIZ_QUESTIONS[currentIdx];
        questionText.textContent = q.question;
        questionCounter.textContent = `Question ${currentIdx + 1} of ${QUIZ_QUESTIONS.length}`;
        currentScoreVal.textContent = score;

        // Progress bar updates
        const progressPercentage = (currentIdx / QUIZ_QUESTIONS.length) * 100;
        progressFill.style.width = `${progressPercentage}%`;

        // Render options
        q.options.forEach((opt, idx) => {
            const btn = document.createElement("button");
            btn.className = "quiz-option";
            btn.innerHTML = `<span style="font-weight:700; color:#00f2fe; margin-right:12px;">${String.fromCharCode(65 + idx)}.</span> ${opt}`;
            
            btn.addEventListener("click", () => {
                if (hasAnswered) return;
                selectOption(btn, idx);
            });
            optionsContainer.appendChild(btn);
        });
    }

    function selectOption(selectedBtn, selectedIdx) {
        hasAnswered = true;
        const q = QUIZ_QUESTIONS[currentIdx];
        const allBtns = optionsContainer.querySelectorAll(".quiz-option");

        allBtns.forEach((btn, idx) => {
            btn.classList.add("disabled");
            if (idx === q.correct) {
                btn.classList.add("correct");
            }
        });

        if (selectedIdx === q.correct) {
            score++;
            currentScoreVal.textContent = score;
        } else {
            selectedBtn.classList.add("incorrect");
        }

        // Show Next Question button
        nextBtn.classList.remove("hidden");
    }

    function showResults() {
        questionView.classList.add("hidden");
        resultView.classList.remove("hidden");
        progressFill.style.width = "100%";

        const total = QUIZ_QUESTIONS.length;
        const percent = Math.round((score / total) * 100);

        finalScoreVal.textContent = `${percent}%`;
        resultFeedbackText.textContent = `You got ${score} out of ${total} questions correct.`;

        if (percent >= 80) {
            resultIcon.className = "quiz-result-icon";
            resultIcon.innerHTML = `<i class="fa-solid fa-circle-check" style="color:#10b981;"></i>`;
            resultTitle.textContent = "Excellent Job! ML Specialist";
        } else if (percent >= 50) {
            resultIcon.className = "quiz-result-icon";
            resultIcon.innerHTML = `<i class="fa-solid fa-circle-info" style="color:#f59e0b;"></i>`;
            resultTitle.textContent = "Keep Learning!";
        } else {
            resultIcon.className = "quiz-result-icon";
            resultIcon.innerHTML = `<i class="fa-solid fa-circle-exclamation" style="color:#f43f5e;"></i>`;
            resultTitle.textContent = "Needs Review";
        }
    }
}
