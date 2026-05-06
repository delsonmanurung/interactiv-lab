import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Computer Vision 101: Interactive Lab",
    page_icon="🔬",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #0f1117; }
h1 { background: linear-gradient(135deg, #6ee7f7, #a78bfa);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.stMetric label { color: #a78bfa !important; }
.info-box { background: #1e1f2e; border-left: 4px solid #6ee7f7;
            padding: 1rem 1.2rem; border-radius: 8px; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🔬 Computer Vision 101: Interactive Lab")
st.caption("Laboratorium edukasi interaktif untuk Image Processing, ML Klasik, dan Deep Learning.")

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Kontrol Lab")
    uploaded = st.file_uploader("Upload Gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])
    module = st.selectbox(
        "Pilih Modul",
        [
            "Dasar & Sinyal Citra",
            "Machine Learning Klasik",
            "CNN & Klasifikasi (ResNet-50)",
            "Deteksi Objek (YOLOv8)",
        ],
    )
    st.markdown("---")
    st.info("Upload gambar lalu pilih modul untuk memulai demonstrasi.")

# ── Helper: load image ────────────────────────────────────────────────────────
def load_image(file) -> np.ndarray:
    img = Image.open(file).convert("RGB")
    return np.array(img)

# ═════════════════════════════════════════════════════════════════════════════
# MODEL LOADERS  (cached agar tidak reload tiap interaksi)
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_resnet():
    import torch
    import torchvision.models as models
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
    return model

@st.cache_resource
def load_yolo():
    from ultralytics import YOLO
    # YOLOv8n (nano) — model paling ringan (~6MB), diunduh otomatis sekali lalu di-cache
    model = YOLO("yolov8n.pt")
    return model

@st.cache_resource
def load_imagenet_labels() -> list:
    """Ambil 1000 label ImageNet dari torchvision — tanpa perlu internet tambahan."""
    from torchvision.models import ResNet50_Weights
    meta = ResNet50_Weights.IMAGENET1K_V1.meta
    return meta["categories"]

# ═════════════════════════════════════════════════════════════════════════════
# MODUL 1 — DASAR & SINYAL CITRA
# ═════════════════════════════════════════════════════════════════════════════
def module_basic(img_rgb: np.ndarray):
    st.subheader("📐 Modul 1: Dasar & Sinyal Citra")

    h, w, c = img_rgb.shape

    # ── Metadata ──────────────────────────────────────────────────────────────
    st.markdown("### 🔎 Metadata Gambar")
    col1, col2, col3 = st.columns(3)
    col1.metric("Lebar (px)", w)
    col2.metric("Tinggi (px)", h)
    col3.metric("Mode Warna", f"RGB ({c} ch)")

    st.markdown("---")

    # ── Manipulasi ────────────────────────────────────────────────────────────
    st.markdown("### 🎛️ Manipulasi Gambar")
    c1, c2, c3 = st.columns(3)
    scale  = c1.slider("Resize (%)", 10, 200, 100, 5)
    angle  = c2.slider("Rotate (°)", -180, 180, 0, 5)
    crop_p = c3.slider("Center Crop (%)", 10, 100, 80, 5)

    # Resize
    new_w = max(1, int(w * scale / 100))
    new_h = max(1, int(h * scale / 100))
    img_m = cv2.resize(img_rgb, (new_w, new_h))

    # Rotate
    cx, cy = new_w // 2, new_h // 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    img_m = cv2.warpAffine(img_m, M, (new_w, new_h))

    # Center crop
    ch_px = max(1, int(new_h * crop_p / 100))
    cw_px = max(1, int(new_w * crop_p / 100))
    sy = (new_h - ch_px) // 2
    sx = (new_w - cw_px) // 2
    img_m = img_m[sy:sy + ch_px, sx:sx + cw_px]

    st.image(img_m, caption="Hasil Manipulasi", width="stretch")
    st.markdown("---")

    # ── Filter ────────────────────────────────────────────────────────────────
    st.markdown("### 🌊 Pipeline Filter")

    # 1. Gaussian Blur
    blurred = cv2.GaussianBlur(img_rgb, (15, 15), 0)

    # 2. Sobel Edge Detection
    gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel   = cv2.magnitude(sobel_x, sobel_y)
    s_max   = sobel.max()
    sobel   = np.uint8(np.clip(sobel / s_max * 255, 0, 255) if s_max > 0 else sobel)

    # 3. Fourier Transform — magnitude spectrum
    f_shift   = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = 20 * np.log(np.abs(f_shift) + 1)
    m_max     = magnitude.max()
    magnitude = np.uint8(np.clip(magnitude / m_max * 255, 0, 255) if m_max > 0 else magnitude)

    fc1, fc2, fc3 = st.columns(3)
    fc1.image(blurred,   caption="① Gaussian Blur",             width="stretch")
    fc2.image(sobel,     caption="② Sobel Edge Detection",      width="stretch", clamp=True)
    fc3.image(magnitude, caption="③ Fourier Magnitude Spectrum", width="stretch", clamp=True)

    st.markdown("""
<div class='info-box'>
<b>💡 Insight:</b> Gaussian Blur menghilangkan noise dengan rata-rata berbobot piksel tetangga.
Sobel mendeteksi tepi dengan menghitung gradien intensitas horizontal &amp; vertikal.
Fourier Transform mengubah gambar ke domain frekuensi — titik terang di tengah = komponen
frekuensi rendah (background), titik di pinggir = frekuensi tinggi (tepi &amp; detail).
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# MODUL 2 — MACHINE LEARNING KLASIK
# ═════════════════════════════════════════════════════════════════════════════
def module_ml_classic(img_rgb: np.ndarray):
    st.subheader("🤖 Modul 2: Machine Learning Klasik")
    st.markdown("### 🎨 Color Quantization dengan K-Means Clustering")

    k = st.select_slider("Jumlah Warna (k)", options=[3, 5, 8], value=5)

    with st.spinner(f"Menjalankan K-Means dengan k={k}..."):
        from sklearn.cluster import MiniBatchKMeans

        h, w, _ = img_rgb.shape
        pixels  = img_rgb.reshape(-1, 3).astype(np.float32)

        kmeans   = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3)
        kmeans.fit(pixels)
        labels   = kmeans.labels_
        centers  = np.uint8(kmeans.cluster_centers_)
        quantized = centers[labels].reshape(h, w, 3)

    col1, col2 = st.columns(2)
    col1.image(img_rgb,   caption="Gambar Asli",                    width="stretch")
    col2.image(quantized, caption=f"Setelah K-Means (k={k} warna)", width="stretch")

    st.markdown("**Palet Warna yang Ditemukan:**")
    palette_bar = np.zeros((60, k * 80, 3), dtype=np.uint8)
    for i, color in enumerate(centers):
        palette_bar[:, i * 80:(i + 1) * 80] = color
    st.image(palette_bar, width="content")

    st.markdown("""
<div class='info-box'>
<b>💡 Insight:</b> K-Means mengelompokkan piksel berdasarkan kemiripan warna di ruang 3D (R, G, B).
Setiap piksel diganti dengan warna centroid kluster terdekatnya.
Semakin kecil k, semakin terkompresi gambar — berguna untuk kompresi gambar lossy &amp; segmentasi sederhana.
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# MODUL 3 — CNN & KLASIFIKASI (ResNet-50 saja)
# ═════════════════════════════════════════════════════════════════════════════
def module_cnn(img_rgb: np.ndarray):
    import torch
    import torchvision.transforms as T

    st.subheader("🧠 Modul 3: CNN & Klasifikasi (ResNet-50)")
    st.markdown("### 🔍 Klasifikasi Gambar dengan ResNet-50 (ImageNet)")

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pil_img = Image.fromarray(img_rgb)
    tensor  = transform(pil_img).unsqueeze(0)

    with st.spinner("Memuat ResNet-50 & melakukan inferensi..."):
        model  = load_resnet()
        labels = load_imagenet_labels()
        with torch.no_grad():
            out   = model(tensor)
            probs = torch.softmax(out, dim=1)[0]

    top3_probs, top3_idx = torch.topk(probs, 3)

    st.markdown("**Top-3 Prediksi:**")
    for prob, idx in zip(top3_probs, top3_idx):
        i     = idx.item()
        label = labels[i] if i < len(labels) else f"Class {i}"
        pct   = prob.item() * 100
        st.progress(min(100, max(0, int(pct))), text=f"**{label}** — {pct:.2f}%")

    st.markdown("---")

    # ── Penjelasan arsitektur (teks saja, tanpa load model tambahan) ──────────
    st.markdown("### 📚 Perbandingan Arsitektur CNN")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### AlexNet (2012)")
        st.markdown("""
- **5 Conv + 3 FC layers**
- ReLU & Dropout
- Memenangkan ImageNet 2012
- **Masalah:** Vanishing gradient pada jaringan sangat dalam
""")
    with col2:
        st.markdown("#### GoogLeNet / Inception (2014)")
        st.markdown("""
- **Inception Module:** konvolusi paralel 1×1, 3×3, 5×5
- Komputasi efisien dengan bottleneck 1×1
- 22 layer, parameter lebih sedikit dari AlexNet
- **Masalah:** Training kompleks
""")
    with col3:
        st.markdown("#### ResNet-50 (2015) ← digunakan")
        st.markdown("""
- **Residual Block / Skip Connection:** `F(x) + x`
- Memungkinkan training 50–152+ layer
- **Solusi Vanishing Gradient:** gradien mengalir lewat skip connection
- State-of-the-art hingga saat ini
""")

    st.info("""
**🔬 Mengapa ResNet Memecahkan Vanishing Gradient?**

Pada jaringan dalam tanpa skip connection (seperti AlexNet), gradien yang di-backpropagate
melewati banyak layer perkalian (<1) sehingga nilainya mendekati nol di layer awal.

ResNet memperkenalkan **Residual Block**: alih-alih mempelajari `H(x)`, jaringan mempelajari
`F(x) = H(x) - x`, lalu outputnya adalah `F(x) + x`. Operasi penjumlahan langsung
meneruskan gradien ke layer sebelumnya tanpa perkalian berulang.

**vs GoogLeNet:** Inception module menggunakan multi-skala konvolusi paralel untuk efisiensi,
tapi tidak secara eksplisit mengatasi vanishing gradient pada kedalaman ekstrem.
""")

# ═════════════════════════════════════════════════════════════════════════════
# MODUL 4 — DETEKSI OBJEK (YOLOv8 saja, tanpa DeepLabV3)
# ═════════════════════════════════════════════════════════════════════════════
def module_yolo(img_rgb: np.ndarray):
    st.subheader("🚀 Modul 4: Deteksi Objek (YOLOv8)")
    st.markdown("### Object Detection dengan YOLOv8n")

    with st.spinner("Memuat YOLOv8n & mendeteksi objek..."):
        model_yolo = load_yolo()
        results    = model_yolo(img_rgb, verbose=False)

    # .plot() dari ultralytics mengembalikan numpy BGR
    result_0     = results[0]
    rendered_bgr = result_0.plot()
    rendered_rgb = cv2.cvtColor(rendered_bgr, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    col1.image(img_rgb,      caption="Input",             width="stretch")
    col2.image(rendered_rgb, caption="YOLOv8n Detection", width="stretch")

    # Tabel deteksi dengan filter & limit tampilan
    # YOLOv8 sudah menerapkan NMS secara internal saat inferensi
    boxes = result_0.boxes
    if boxes is not None and len(boxes) > 0:
        import pandas as pd

        # Slider filter
        col_s1, col_s2 = st.columns(2)
        conf_thresh = col_s1.slider(
            "Confidence minimum", min_value=0.0, max_value=1.0,
            value=0.25, step=0.05, key="yolo_conf_thresh"
        )
        max_show = col_s2.slider(
            "Tampilkan deteksi teratas", min_value=1, max_value=30,
            value=10, key="yolo_max_show"
        )

        # Bangun tabel menggunakan atribut Boxes sesuai dokumentasi resmi
        names = result_0.names
        data = []
        for i in range(len(boxes)):
            conf = float(boxes.conf[i].item())
            if conf < conf_thresh:
                continue
            cls_id = int(boxes.cls[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            data.append({
                "Label":      names[cls_id],
                "Confidence": conf,
                "xmin": int(x1),
                "ymin": int(y1),
                "xmax": int(x2),
                "ymax": int(y2),
            })

        data = sorted(data, key=lambda d: d["Confidence"], reverse=True)
        shown = data[:max_show]
        shown_display = [
            {**d, "Confidence": f"{d['Confidence']:.2%}"} for d in shown
        ]

        st.markdown(
            f"**Objek Terdeteksi:** total **{len(data)}** deteksi "
            f"(conf ≥ {conf_thresh:.0%}), menampilkan **{len(shown)}**"
        )
        if shown_display:
            st.dataframe(pd.DataFrame(shown_display), use_container_width=True)
        else:
            st.info("Tidak ada objek dengan confidence di atas threshold yang dipilih.")
    else:
        st.warning("Tidak ada objek yang terdeteksi pada gambar ini.")

    st.info("""
**🔬 YOLOv8 vs Faster R-CNN**

|  | **YOLO (Single-Stage)** | **Faster R-CNN (Two-Stage)** |
|---|---|---|
| **Cara Kerja** | Langsung prediksi bbox & class sekali pass | 1) RPN → 2) ROI Pooling & klasifikasi |
| **Kecepatan** | ⚡ Sangat cepat (~45–140 FPS) | 🐢 Lebih lambat (~5–17 FPS) |
| **Akurasi** | Baik untuk objek besar/umum | Lebih akurat untuk objek kecil |
| **Use Case** | Real-time detection, video | High-accuracy batch processing |

YOLO memformulasikan deteksi sebagai **single regression problem** dari piksel ke bbox & probabilitas.
""")

# ═════════════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ═════════════════════════════════════════════════════════════════════════════
if uploaded is None:
    st.markdown("""
<div style="text-align:center; padding:4rem 0;">
  <h2 style="color:#6ee7f7;">👈 Upload gambar di sidebar untuk memulai</h2>
  <p style="color:#8b8fa8;">Pilih salah satu dari 4 modul pembelajaran interaktif</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📖 Daftar Modul")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.markdown("**🖼️ Modul 1**\nDasar & Sinyal Citra\n\nResize · Rotate · Blur · Sobel · FFT")
    mc2.markdown("**🤖 Modul 2**\nML Klasik\n\nK-Means Color Quantization")
    mc3.markdown("**🧠 Modul 3**\nCNN Klasifikasi\n\nResNet-50 + Penjelasan Arsitektur")
    mc4.markdown("**🚀 Modul 4**\nDeteksi Objek\n\nYOLOv8n Real-time Detection")
else:
    try:
        img_rgb = load_image(uploaded)
        st.sidebar.image(img_rgb, caption="Gambar Terupload", width="stretch")

        if module == "Dasar & Sinyal Citra":
            module_basic(img_rgb)
        elif module == "Machine Learning Klasik":
            module_ml_classic(img_rgb)
        elif module == "CNN & Klasifikasi (ResNet-50)":
            module_cnn(img_rgb)
        elif module == "Deteksi Objek (YOLOv8)":
            module_yolo(img_rgb)

    except Exception as e:
        st.error(f"❌ Terjadi error: {e}")
        st.exception(e)
