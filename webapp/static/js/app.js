/**
 * PhytoGuard AI — Frontend Application Logic
 * Handles: drag-drop upload, prediction, charts, history, samples, tabs, model info
 */

"use strict";

// ── State ─────────────────────────────────────────────────────────────────────
let currentFile = null;
let gaugeChart = null;
let trainingChart = null;
let currentTab = "symptoms";
let currentDiseaseData = null;

// Training history data (from vit_base_training_history.json — loaded server-side)
const TRAINING_EVAL_ACCURACY = [
  0.9946, 0.9971, 0.9973, 0.9988, 0.9969,
  0.9992, 0.9994, 0.9998, 0.9996, 0.9996
];
const TRAINING_EPOCHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

// Classes data (from label_mappings.json)
const CLASSES = [
  { name: "Apple___Apple_scab",            display: "Apple Scab",           plant: "Apple",   color: "#f59e0b", status: "diseased" },
  { name: "Apple___Black_rot",             display: "Black Rot",            plant: "Apple",   color: "#ef4444", status: "diseased" },
  { name: "Apple___Cedar_apple_rust",      display: "Cedar Apple Rust",     plant: "Apple",   color: "#f97316", status: "diseased" },
  { name: "Apple___healthy",               display: "Healthy",              plant: "Apple",   color: "#22c55e", status: "healthy"  },
  { name: "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", display: "Gray Leaf Spot", plant: "Corn", color: "#ef4444", status: "diseased" },
  { name: "Corn_(maize)___Common_rust_",   display: "Common Rust",          plant: "Corn",    color: "#f59e0b", status: "diseased" },
  { name: "Corn_(maize)___Northern_Leaf_Blight", display: "Northern Leaf Blight", plant: "Corn", color: "#ef4444", status: "diseased" },
  { name: "Corn_(maize)___healthy",        display: "Healthy",              plant: "Corn",    color: "#22c55e", status: "healthy"  },
  { name: "Potato___Early_blight",         display: "Early Blight",         plant: "Potato",  color: "#f59e0b", status: "diseased" },
  { name: "Potato___Late_blight",          display: "Late Blight",          plant: "Potato",  color: "#dc2626", status: "diseased" },
  { name: "Potato___healthy",              display: "Healthy",              plant: "Potato",  color: "#22c55e", status: "healthy"  },
];

// ── DOM Ready ─────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initUploadZone();
  loadSamples();
  loadModelInfo();
  buildClassesGrid();
  buildTrainingChart();
  initScrollSpy();
});

// ── Navbar scroll effect ───────────────────────────────────────────────────────
function initScrollSpy() {
  const navbar = document.getElementById("navbar");
  window.addEventListener("scroll", () => {
    navbar.classList.toggle("scrolled", window.scrollY > 60);
  }, { passive: true });
}

/* ════════════════════════════════════════════════════════════════════════════
   UPLOAD & DRAG-DROP
   ════════════════════════════════════════════════════════════════════════════ */
function initUploadZone() {
  const zone = document.getElementById("upload-zone");
  const input = document.getElementById("file-input");

  // Click on zone (excluding the button itself)
  zone.addEventListener("click", (e) => {
    if (e.target === zone || e.target.id === "upload-inner") {
      input.click();
    }
  });

  input.addEventListener("change", () => {
    if (input.files[0]) handleFile(input.files[0]);
  });

  // Drag events
  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("drag-over");
  });

  zone.addEventListener("dragleave", (e) => {
    if (!zone.contains(e.relatedTarget)) zone.classList.remove("drag-over");
  });

  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });
}

function handleFile(file) {
  const validTypes = ["image/jpeg", "image/jpg", "image/png"];
  if (!validTypes.includes(file.type)) {
    showValidationMsg("⚠️ Unsupported file type. Please upload JPG or PNG.");
    return;
  }
  if (file.size > 16 * 1024 * 1024) {
    showValidationMsg("⚠️ File too large. Maximum size is 16 MB.");
    return;
  }

  currentFile = file;
  hideValidationMsg();

  const reader = new FileReader();
  reader.onload = (e) => {
    const previewImg = document.getElementById("preview-img");
    previewImg.src = e.target.result;
    document.getElementById("upload-inner").style.display = "none";
    document.getElementById("upload-preview").style.display = "flex";
    document.getElementById("btn-analyze").disabled = false;
  };
  reader.readAsDataURL(file);
}

function resetUpload() {
  currentFile = null;
  document.getElementById("file-input").value = "";
  document.getElementById("upload-inner").style.display = "flex";
  document.getElementById("upload-preview").style.display = "none";
  document.getElementById("btn-analyze").disabled = true;
  hideValidationMsg();
}

/* ════════════════════════════════════════════════════════════════════════════
   PREDICTION
   ════════════════════════════════════════════════════════════════════════════ */
async function runPrediction() {
  if (!currentFile) return;

  setLoading(true);

  const formData = new FormData();
  formData.append("image", currentFile);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      if (data.validation_failed) {
        showValidationMsg("⚠️ " + data.error);
      } else {
        showValidationMsg("❌ " + (data.error || "Prediction failed. Please try again."));
      }
      setLoading(false);
      return;
    }

    hideValidationMsg();
    displayResults(data);
    updateHistory();

    // Scroll to results
    setTimeout(() => {
      document.getElementById("results-card").scrollIntoView({ behavior: "smooth", block: "nearest" });
    }, 300);

  } catch (err) {
    showValidationMsg("❌ Network error. Is the server running?");
    console.error("Prediction error:", err);
  }

  setLoading(false);
}

function setLoading(loading) {
  const overlay = document.getElementById("global-overlay");
  const btnText = document.getElementById("btn-text");
  const btnSpinner = document.getElementById("btn-spinner");
  const btnAnalyze = document.getElementById("btn-analyze");

  overlay.style.display = loading ? "flex" : "none";
  btnText.style.display = loading ? "none" : "flex";
  btnSpinner.style.display = loading ? "flex" : "none";
  btnAnalyze.disabled = loading;
}

/* ════════════════════════════════════════════════════════════════════════════
   DISPLAY RESULTS
   ════════════════════════════════════════════════════════════════════════════ */
function displayResults(data) {
  currentDiseaseData = data.disease_info;

  const isHealthy = data.is_healthy;
  const color = data.disease_info.color || (isHealthy ? "#22c55e" : "#ef4444");

  // Show results card
  document.getElementById("results-placeholder").style.display = "none";
  document.getElementById("results-card").style.display = "block";

  // Status icon & header
  document.getElementById("result-icon").textContent = isHealthy ? "✅" : "🔬";
  document.getElementById("result-disease").textContent = data.display_name;
  document.getElementById("result-disease").style.color = color;
  document.getElementById("result-plant").textContent = data.plant;

  // Badge
  const badge = document.getElementById("result-badge");
  badge.textContent = isHealthy ? "Healthy" : (data.disease_info.severity || "Diseased");
  badge.style.background = color + "20";
  badge.style.color = color;
  badge.style.border = `1px solid ${color}50`;

  // Header background tint
  document.getElementById("result-header").style.background =
    isHealthy ? "rgba(34, 197, 94, 0.04)" : "rgba(239, 68, 68, 0.04)";

  // Confidence gauge
  drawGauge(data.confidence, color);

  // Top-3 bars
  drawTop3Bars(data.top3);

  // Inference info
  const infer = document.getElementById("infer-info");
  infer.innerHTML = `
    <div class="infer-item">
      <div class="infer-val">${data.inference_time_ms}ms</div>
      <div class="infer-lbl">Inference</div>
    </div>
    <div class="infer-item">
      <div class="infer-val">${data.device.toUpperCase()}</div>
      <div class="infer-lbl">Device</div>
    </div>
    <div class="infer-item">
      <div class="infer-val">${data.confidence.toFixed(1)}%</div>
      <div class="infer-lbl">Confidence</div>
    </div>
  `;

  // Disease info panel
  showDiseasePanel(data);
}

/* ── Confidence Gauge (Chart.js doughnut) ────────────────────────────────── */
function drawGauge(confidence, color) {
  const ctx = document.getElementById("gauge-chart").getContext("2d");
  if (gaugeChart) gaugeChart.destroy();

  const pct = Math.min(100, Math.max(0, confidence));
  const remaining = 100 - pct;

  gaugeChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      datasets: [{
        data: [pct, remaining],
        backgroundColor: [color, "rgba(255,255,255,0.05)"],
        borderWidth: 0,
        circumference: 180,
        rotation: 270,
      }]
    },
    options: {
      responsive: false,
      cutout: "75%",
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      animation: { duration: 1000, easing: "easeOutQuart" },
    },
  });

  // Animate value counter
  const valueEl = document.getElementById("gauge-value");
  animateCounter(valueEl, 0, pct, 1000, (v) => `${v.toFixed(1)}%`);
}

/* ── Top-3 Bars ─────────────────────────────────────────────────────────────── */
function drawTop3Bars(top3) {
  const container = document.getElementById("top3-bars");
  container.innerHTML = "";

  top3.forEach((item, i) => {
    const classInfo = CLASSES.find(c => c.name === item.class_name);
    const color = classInfo ? classInfo.color : "#6b7280";
    const pct = item.confidence.toFixed(1);

    const el = document.createElement("div");
    el.className = "top3-bar-item";
    el.style.animationDelay = `${i * 0.1}s`;
    el.innerHTML = `
      <div class="top3-bar-header">
        <span class="top3-class-name ${i === 0 ? "top1" : ""}">${item.display_name}</span>
        <span class="top3-conf">${pct}%</span>
      </div>
      <div class="top3-bar-track">
        <div class="top3-bar-fill" id="bar-fill-${i}" style="background:${color};"></div>
      </div>
    `;
    container.appendChild(el);

    // Animate bar fill
    setTimeout(() => {
      document.getElementById(`bar-fill-${i}`).style.width = `${Math.min(100, item.confidence)}%`;
    }, 100 + i * 80);
  });
}

/* ── Disease Info Panel ─────────────────────────────────────────────────────── */
function showDiseasePanel(data) {
  const panel = document.getElementById("disease-panel");
  panel.style.display = "block";

  const info = data.disease_info;
  const color = info.color || "#22c55e";

  document.getElementById("panel-icon").textContent = data.is_healthy ? "🌱" : "🔬";
  document.getElementById("panel-title").textContent = info.display_name;
  document.getElementById("panel-description").textContent = info.description;
  document.getElementById("panel-title").style.color = color;

  // Reset to symptoms tab
  switchTab("symptoms");
  document.querySelectorAll(".tab-btn").forEach((btn, i) => {
    btn.classList.toggle("active", i === 0);
  });
}

function switchTab(tabName) {
  currentTab = tabName;

  // Update tab buttons
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.classList.toggle("active", btn.textContent.toLowerCase() === tabName);
  });

  // Render content
  const content = document.getElementById("tab-content");
  if (!currentDiseaseData) return;

  const items = currentDiseaseData[tabName] || [];

  if (items.length === 0) {
    const labels = {
      symptoms: "No symptoms listed.",
      causes: "No specific causes listed.",
      treatment: "No treatment required — plant is healthy!",
      prevention: "Continue standard care practices."
    };
    content.innerHTML = `<p class="tab-empty">${labels[tabName] || "No information available."}</p>`;
    return;
  }

  const listItems = items.map((item, i) => `
    <div class="tab-list-item" style="animation-delay:${i * 0.06}s">
      <div class="tab-list-bullet"></div>
      <span>${item}</span>
    </div>
  `).join("");

  content.innerHTML = `<div class="tab-list">${listItems}</div>`;
}

/* ════════════════════════════════════════════════════════════════════════════
   SAMPLE IMAGES
   ════════════════════════════════════════════════════════════════════════════ */
async function loadSamples() {
  const grid = document.getElementById("samples-grid");

  try {
    const response = await fetch("/samples");
    const samples = await response.json();

    if (samples.length === 0) {
      grid.innerHTML = '<p style="color:var(--text-muted);text-align:center;grid-column:1/-1;">No sample images found.</p>';
      return;
    }

    grid.innerHTML = "";
    samples.forEach((filename, i) => {
      const label = filenameToLabel(filename);
      const card = document.createElement("div");
      card.className = "sample-card";
      card.style.animationDelay = `${i * 0.05}s`;
      card.innerHTML = `
        <img src="/sample-image/${encodeURIComponent(filename)}" alt="${label}" loading="lazy" />
        <div class="sample-overlay">🔍</div>
        <div class="sample-card-label">${label}</div>
      `;
      card.addEventListener("click", () => loadSampleForPrediction(filename, label));
      grid.appendChild(card);
    });

  } catch (err) {
    grid.innerHTML = '<p style="color:var(--text-muted);text-align:center;grid-column:1/-1;">Could not load sample images.</p>';
    console.error("Samples error:", err);
  }
}

function filenameToLabel(filename) {
  // Map filenames like "AppleScab1.JPG" to readable labels
  const map = {
    "AppleCedarRust": "Cedar Apple Rust",
    "AppleScab": "Apple Scab",
    "CornCommonRust": "Common Rust",
    "PotatoEarlyBlight": "Early Blight",
    "PotatoHealthy": "Healthy Potato",
  };
  for (const [key, val] of Object.entries(map)) {
    if (filename.includes(key)) return val;
  }
  return filename.replace(/\.[^.]+$/, "").replace(/([A-Z])/g, " $1").trim();
}

async function loadSampleForPrediction(filename, label) {
  try {
    const response = await fetch(`/sample-image/${encodeURIComponent(filename)}`);
    const blob = await response.blob();
    const file = new File([blob], filename, { type: blob.type });
    handleFile(file);

    // Scroll to detect section and run prediction
    document.getElementById("detect").scrollIntoView({ behavior: "smooth" });
    setTimeout(() => {
      document.getElementById("btn-analyze").disabled = false;
      runPrediction();
    }, 600);

  } catch (err) {
    console.error("Could not load sample:", err);
  }
}

/* ════════════════════════════════════════════════════════════════════════════
   PREDICTION HISTORY
   ════════════════════════════════════════════════════════════════════════════ */
async function updateHistory() {
  try {
    const response = await fetch("/history");
    const history = await response.json();
    renderHistory(history);
  } catch (err) {
    console.error("History error:", err);
  }
}

function renderHistory(history) {
  const empty = document.getElementById("history-empty");
  const grid = document.getElementById("history-grid");
  const clearBtn = document.getElementById("btn-clear-history");

  if (history.length === 0) {
    empty.style.display = "block";
    grid.style.display = "none";
    clearBtn.style.display = "none";
    return;
  }

  empty.style.display = "none";
  grid.style.display = "grid";
  clearBtn.style.display = "inline-block";

  grid.innerHTML = history.map((item, i) => `
    <div class="history-card" style="animation-delay:${i * 0.05}s">
      <img class="history-thumb" src="${item.thumbnail}" alt="${item.display_name}" />
      <div class="history-body">
        <div class="history-disease">${item.display_name}</div>
        <div class="history-plant">${item.plant}</div>
        <div class="history-footer">
          <span class="history-conf" style="color:${item.color}">${item.confidence.toFixed(1)}%</span>
          <span class="history-time">${item.timestamp}</span>
        </div>
      </div>
    </div>
  `).join("");
}

function clearHistory() {
  document.getElementById("history-grid").innerHTML = "";
  document.getElementById("history-grid").style.display = "none";
  document.getElementById("history-empty").style.display = "block";
  document.getElementById("btn-clear-history").style.display = "none";
}

/* ════════════════════════════════════════════════════════════════════════════
   MODEL INFO
   ════════════════════════════════════════════════════════════════════════════ */
async function loadModelInfo() {
  const grid = document.getElementById("model-grid");

  try {
    const response = await fetch("/model-info");
    const info = await response.json();

    const items = [
      { label: "Architecture",     value: info.architecture,          accent: true },
      { label: "Framework",        value: info.framework,             accent: false },
      { label: "Number of Classes",value: info.num_classes,           accent: false },
      { label: "Input Image Size", value: info.input_size,            accent: false },
      { label: "Patch Size",       value: `${info.patch_size}×${info.patch_size} px`, accent: false },
      { label: "Hidden Size",      value: info.hidden_size,           accent: false },
      { label: "Attention Heads",  value: info.num_attention_heads,   accent: false },
      { label: "Transformer Layers", value: info.num_hidden_layers,   accent: false },
      { label: "Normalization",    value: info.normalization,         accent: false },
      { label: "Model Size",       value: `${info.model_size_mb} MB`, accent: false },
      { label: "Device",           value: info.device.toUpperCase(), accent: false },
      { label: "Validation Accuracy", value: info.validation_accuracy, accent: true },
      { label: "Training Epochs",  value: info.training_epochs,       accent: false },
      { label: "Dataset",          value: info.dataset,               accent: false },
      { label: "Training Images",  value: info.train_images?.toLocaleString(), accent: false },
      { label: "Val Images",       value: info.val_images?.toLocaleString(),   accent: false },
    ];

    grid.innerHTML = items.map((item, i) => `
      <div class="model-info-card" style="animation-delay:${i * 0.04}s">
        <div class="model-info-label">${item.label}</div>
        <div class="model-info-value ${item.accent ? "accent" : ""}">${item.value ?? "—"}</div>
      </div>
    `).join("");

  } catch (err) {
    grid.innerHTML = `<p style="color:var(--text-muted);grid-column:1/-1;text-align:center;padding:2rem;">Could not load model info. Is the server running?</p>`;
    console.error("Model info error:", err);
  }
}

/* ── Training Chart (Chart.js) ──────────────────────────────────────────────── */
function buildTrainingChart() {
  const ctx = document.getElementById("training-chart").getContext("2d");

  trainingChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: TRAINING_EPOCHS.map(e => `Epoch ${e}`),
      datasets: [{
        label: "Validation Accuracy",
        data: TRAINING_EVAL_ACCURACY.map(v => (v * 100).toFixed(2)),
        borderColor: "#22c55e",
        backgroundColor: "rgba(34, 197, 94, 0.08)",
        borderWidth: 2.5,
        pointBackgroundColor: "#22c55e",
        pointBorderColor: "#22c55e",
        pointRadius: 5,
        pointHoverRadius: 7,
        fill: true,
        tension: 0.4,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#0d1521",
          borderColor: "rgba(34,197,94,0.3)",
          borderWidth: 1,
          titleColor: "#94a3b8",
          bodyColor: "#f0f6ff",
          callbacks: {
            label: (ctx) => ` Accuracy: ${ctx.raw}%`
          }
        }
      },
      scales: {
        x: {
          grid: { color: "rgba(255,255,255,0.04)" },
          ticks: { color: "#64748b", font: { size: 11 } },
        },
        y: {
          min: 99,
          max: 100.1,
          grid: { color: "rgba(255,255,255,0.04)" },
          ticks: {
            color: "#64748b",
            font: { size: 11 },
            callback: (v) => `${v}%`
          },
        }
      },
      animation: { duration: 1200, easing: "easeOutQuart" },
    }
  });
}

/* ── Classes Grid ──────────────────────────────────────────────────────────── */
function buildClassesGrid() {
  const grid = document.getElementById("classes-grid");
  grid.innerHTML = CLASSES.map((cls, i) => `
    <div class="class-item" style="animation-delay:${i * 0.04}s">
      <div class="class-dot" style="background:${cls.color};box-shadow:0 0 8px ${cls.color}60;"></div>
      <div>
        <div class="class-name">${cls.display}</div>
        <div class="class-plant">${cls.plant} · ${cls.status}</div>
      </div>
    </div>
  `).join("");
}

/* ════════════════════════════════════════════════════════════════════════════
   UTILITIES
   ════════════════════════════════════════════════════════════════════════════ */
function showValidationMsg(msg) {
  const el = document.getElementById("validation-msg");
  el.textContent = msg;
  el.style.display = "block";
}

function hideValidationMsg() {
  document.getElementById("validation-msg").style.display = "none";
}

function animateCounter(el, from, to, duration, formatter) {
  const start = performance.now();
  function step(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const value = from + (to - from) * eased;
    el.textContent = formatter(value);
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// Animate ViT diagram patches on load
(function animateViTPatches() {
  const patches = document.querySelectorAll(".patch");
  if (!patches.length) return;
  setInterval(() => {
    patches.forEach(p => {
      p.classList.toggle("active", Math.random() > 0.4);
    });
  }, 1500);
})();
