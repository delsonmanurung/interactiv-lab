import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd

st.set_page_config(
    page_title="Computer Vision Lab",
    page_icon="vision",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'Plus Jakarta Sans', sans-serif; 
    }

    h1, .gradient-text { 
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    .info-box { 
        background-color: rgba(128, 128, 128, 0.05); 
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-left: 4px solid #38bdf8;
        padding: 1.5rem; 
        border-radius: 12px; 
        margin: 1.5rem 0; 
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        
        font-size: 0.95rem;
        line-height: 1.6;
    }

    .feature-card {
        background: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        height: 100%;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
        border-color: rgba(56, 189, 248, 0.3);
    }
    
    .feature-title {
        
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        padding-bottom: 0.5rem;
    }
    
    .feature-desc {
        opacity: 0.8;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    hr {
        border-color: rgba(128, 128, 128, 0.2);
        margin: 2rem 0;
    }
    
    .stMetric label { 
         !important; 
        font-weight: 500;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.title("Computer Vision 101: Interactive Lab")
st.markdown("<p style='opacity: 0.8; font-size: 1.1rem; margin-top: -10px; margin-bottom: 2rem;'>Laboratorium edukasi interaktif untuk Image Processing, ML Klasik, dan Deep Learning.</p>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h3 style=' font-weight: 600;'>Kontrol Lab</h3>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload Gambar (JPG/PNG)", type=["jpg", "jpeg", "png"], help="Format yang didukung: JPG, JPEG, PNG")
    
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
    st.markdown("<p style='opacity: 0.8; font-size: 0.9rem;'>Upload gambar lalu pilih modul untuk memulai demonstrasi.</p>", unsafe_allow_html=True)

def load_image(file) -> np.ndarray:
    img = Image.open(file).convert("RGB")
    return np.array(img)

@st.cache_resource(show_spinner=False)
def load_resnet():
    import torch
    import torchvision.models as models
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
    return model

@st.cache_resource(show_spinner=False)
def load_yolo():
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    return model

@st.cache_resource(show_spinner=False)
def load_imagenet_labels() -> list:
    from torchvision.models import ResNet50_Weights
    meta = ResNet50_Weights.IMAGENET1K_V1.meta
    return meta["categories"]

def module_basic(img_rgb: np.ndarray):
    st.markdown("<h3 class='gradient-text'>Modul 1: Dasar & Sinyal Citra</h3>", unsafe_allow_html=True)
    h, w, c = img_rgb.shape

    st.markdown("<h4 style=' margin-top: 1.5rem;'>Metadata Gambar</h4>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Lebar", f"{w} px")
    m2.metric("Tinggi", f"{h} px")
    m3.metric("Mode Warna", f"RGB ({c} ch)")

    st.markdown("---")

    st.markdown("<h4 style=''>Manipulasi Gambar</h4>", unsafe_allow_html=True)
    
    with st.container():
        c1, c2, c3 = st.columns(3)
        scale  = c1.slider("Resize (%)", 10, 200, 100, 5)
        angle  = c2.slider("Rotate (Derajat)", -180, 180, 0, 5)
        crop_p = c3.slider("Center Crop (%)", 10, 100, 80, 5)

        new_w = max(1, int(w * scale / 100))
        new_h = max(1, int(h * scale / 100))
        img_m = cv2.resize(img_rgb, (new_w, new_h))

        cx, cy = new_w // 2, new_h // 2
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        img_m = cv2.warpAffine(img_m, M, (new_w, new_h))

        ch_px = max(1, int(new_h * crop_p / 100))
        cw_px = max(1, int(new_w * crop_p / 100))
        sy = (new_h - ch_px) // 2
        sx = (new_w - cw_px) // 2
        img_m = img_m[sy:sy + ch_px, sx:sx + cw_px]

        st.image(img_m, caption="Pratinjau Manipulasi", width="stretch")

    st.markdown("---")

    st.markdown("<h4 style=''>Pipeline Filter</h4>", unsafe_allow_html=True)

    with st.spinner("Memproses filter..."):
        blurred = cv2.GaussianBlur(img_rgb, (15, 15), 0)

        gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel   = cv2.magnitude(sobel_x, sobel_y)
        s_max   = sobel.max()
        sobel   = np.uint8(np.clip(sobel / s_max * 255, 0, 255) if s_max > 0 else sobel)

        f_shift   = np.fft.fftshift(np.fft.fft2(gray))
        magnitude = 20 * np.log(np.abs(f_shift) + 1)
        m_max     = magnitude.max()
        magnitude = np.uint8(np.clip(magnitude / m_max * 255, 0, 255) if m_max > 0 else magnitude)

    fc1, fc2, fc3 = st.columns(3)
    fc1.image(blurred,   caption="Gaussian Blur", width="stretch")
    fc2.image(sobel,     caption="Sobel Edge Detection", width="stretch", clamp=True)
    fc3.image(magnitude, caption="Fourier Magnitude Spectrum", width="stretch", clamp=True)

    st.markdown("""
    <div class='info-box'>
        <span style=' font-weight: 600;'>Insight Teori</span><br/><br/>
        <b>Gaussian Blur:</b> Menghilangkan noise dengan menerapkan rata-rata berbobot pada piksel tetangga.<br/>
        <b>Sobel Edge:</b> Mendeteksi tepi dengan menghitung gradien intensitas horizontal dan vertikal.<br/>
        <b>Fourier Transform:</b> Mengubah gambar ke domain frekuensi. Titik terang di tengah merepresentasikan komponen frekuensi rendah (background), sedangkan titik di pinggir adalah frekuensi tinggi (tepi dan detail).
    </div>
    """, unsafe_allow_html=True)

def module_ml_classic(img_rgb: np.ndarray):
    st.markdown("<h3 class='gradient-text'>Modul 2: Machine Learning Klasik</h3>", unsafe_allow_html=True)
    st.markdown("<h4 style=' margin-top: 1.5rem;'>Color Quantization dengan K-Means Clustering</h4>", unsafe_allow_html=True)

    k = st.select_slider("Tentukan Jumlah Warna (K)", options=[3, 5, 8], value=5)

    with st.spinner(f"Menjalankan algoritma K-Means (K={k})..."):
        from sklearn.cluster import MiniBatchKMeans

        h, w, _ = img_rgb.shape
        pixels  = img_rgb.reshape(-1, 3).astype(np.float32)

        kmeans   = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3)
        kmeans.fit(pixels)
        labels   = kmeans.labels_
        centers  = np.uint8(kmeans.cluster_centers_)
        quantized = centers[labels].reshape(h, w, 3)

    col1, col2 = st.columns(2)
    col1.image(img_rgb,   caption="Input Original", width="stretch")
    col2.image(quantized, caption=f"Output K-Means (K={k})", width="stretch")

    st.markdown("<h5 style=' margin-top: 1rem;'>Ekstraksi Palet Warna</h5>", unsafe_allow_html=True)
    palette_bar = np.zeros((60, k * 80, 3), dtype=np.uint8)
    for i, color in enumerate(centers):
        palette_bar[:, i * 80:(i + 1) * 80] = color
    st.image(palette_bar, width="content")

    st.markdown("""
    <div class='info-box'>
        <span style=' font-weight: 600;'>Insight Teori</span><br/><br/>
        Algoritma <b>K-Means</b> mengelompokkan piksel berdasarkan kemiripan nilai warna di ruang 3 dimensi (R, G, B). 
        Setiap piksel kemudian diganti dengan warna centroid kluster terdekatnya.<br/><br/>
        Semakin kecil nilai <i>K</i>, semakin sedikit variasi warna yang dihasilkan, membuat gambar tampak lebih terkompresi. Teknik ini sangat berguna untuk kompresi gambar lossy dan segmentasi objek sederhana berbasis warna.
    </div>
    """, unsafe_allow_html=True)

def module_cnn(img_rgb: np.ndarray):
    import torch
    import torchvision.transforms as T

    st.markdown("<h3 class='gradient-text'>Modul 3: CNN & Klasifikasi (ResNet-50)</h3>", unsafe_allow_html=True)
    st.markdown("<h4 style=' margin-top: 1.5rem;'>Klasifikasi Gambar dengan ResNet-50 (Pre-trained ImageNet)</h4>", unsafe_allow_html=True)

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pil_img = Image.fromarray(img_rgb)
    tensor  = transform(pil_img).unsqueeze(0)

    with st.spinner("Memuat model ResNet-50 dan memproses inferensi..."):
        model  = load_resnet()
        labels = load_imagenet_labels()
        with torch.no_grad():
            out   = model(tensor)
            probs = torch.softmax(out, dim=1)[0]

    top3_probs, top3_idx = torch.topk(probs, 3)

    st.markdown("<h5 style=' margin-top: 1rem;'>Top-3 Prediksi Model</h5>", unsafe_allow_html=True)
    for prob, idx in zip(top3_probs, top3_idx):
        i     = idx.item()
        label = labels[i] if i < len(labels) else f"Class {i}"
        pct   = prob.item() * 100
        st.progress(min(100, max(0, int(pct))), text=f"{label.capitalize()} — {pct:.2f}%")

    st.markdown("---")

    st.markdown("<h4 style=''>Perbandingan Arsitektur Konvolusi</h4>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='feature-card'>
            <div class='feature-title'>AlexNet (2012)</div>
            <div class='feature-desc'>
            • Terdiri dari 5 layer Konvolusi dan 3 layer Fully Connected.<br/>
            • Mempopulerkan penggunaan aktivasi ReLU dan teknik Dropout.<br/>
            • <b>Kelemahan:</b> Rentan terhadap isu <i>Vanishing Gradient</i> jika jaringan dibuat terlalu dalam.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='feature-card'>
            <div class='feature-title'>GoogLeNet / Inception (2014)</div>
            <div class='feature-desc'>
            • Menggunakan <i>Inception Module</i> (konvolusi paralel 1x1, 3x3, 5x5).<br/>
            • Komputasi sangat efisien dengan teknik bottleneck 1x1.<br/>
            • <b>Kelemahan:</b> Proses pelatihan dan arsitektur yang cukup kompleks.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='feature-card' style='border-color: rgba(56, 189, 248, 0.5); box-shadow: 0 0 15px rgba(56, 189, 248, 0.1);'>
            <div class='feature-title' style=''>ResNet-50 (2015)</div>
            <div class='feature-desc'>
            • Memperkenalkan inovasi <b>Residual Block / Skip Connection</b>.<br/>
            • Memungkinkan model untuk dilatih hingga kedalaman 152+ layer.<br/>
            • <b>Kelebihan:</b> Sepenuhnya menyelesaikan masalah <i>Vanishing Gradient</i>.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
        <span style=' font-weight: 600;'>Bagaimana ResNet Mengatasi Vanishing Gradient?</span><br/><br/>
        Pada arsitektur jaringan dalam standar (seperti AlexNet), nilai gradien error yang merambat mundur (backpropagation) 
        sering kali mengecil mendekati nol setelah melewati banyak layer, membuat awal jaringan berhenti belajar.<br/><br/>
        <b>ResNet</b> memodifikasi fungsi objektif dengan <i>Residual Block</i>. Alih-alih mempelajari fungsi pemetaan utuh H(x), 
        model mempelajari selisihnya F(x) = H(x) - x, sehingga output layer adalah F(x) + x. Operasi penjumlahan ini memberikan rute "jalan pintas" (skip connection) 
        bagi gradien untuk mengalir lurus melewati layer tanpa perkalian yang mereduksi nilainya.
    </div>
    """, unsafe_allow_html=True)

def module_yolo(img_rgb: np.ndarray):
    st.markdown("<h3 class='gradient-text'>Modul 4: Deteksi Objek (YOLOv8)</h3>", unsafe_allow_html=True)
    st.markdown("<h4 style=' margin-top: 1.5rem;'>Deteksi Multi-Objek secara Real-time</h4>", unsafe_allow_html=True)

    with st.spinner("Memuat model YOLOv8n dan menjalankan deteksi..."):
        model_yolo = load_yolo()
        results    = model_yolo(img_rgb, verbose=False)

    result_0     = results[0]
    rendered_bgr = result_0.plot()
    rendered_rgb = cv2.cvtColor(rendered_bgr, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    col1.image(img_rgb,      caption="Input Original", width="stretch")
    col2.image(rendered_rgb, caption="Output Deteksi YOLOv8n", width="stretch")

    st.markdown("<h4 style=' margin-top: 2rem;'>Filter Hasil Deteksi</h4>", unsafe_allow_html=True)
    boxes = result_0.boxes
    if boxes is not None and len(boxes) > 0:
        
        with st.container():
            c_s1, c_s2 = st.columns(2)
            conf_thresh = c_s1.slider(
                "Minimal Confidence", min_value=0.0, max_value=1.0,
                value=0.25, step=0.05, key="yolo_conf_thresh"
            )
            max_show = c_s2.slider(
                "Batas Jumlah Tampilan", min_value=1, max_value=30,
                value=10, key="yolo_max_show"
            )

        names = result_0.names
        data = []
        for i in range(len(boxes)):
            conf = float(boxes.conf[i].item())
            if conf < conf_thresh:
                continue
            cls_id = int(boxes.cls[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            data.append({
                "Label":      names[cls_id].capitalize(),
                "Confidence": conf,
                "Koordinat": f"[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]"
            })

        data = sorted(data, key=lambda d: d["Confidence"], reverse=True)
        shown = data[:max_show]
        shown_display = [
            {**d, "Confidence": f"{d['Confidence']:.1%}"} for d in shown
        ]

        st.markdown(f"<p style='opacity: 0.8; font-size: 0.95rem;'>Ditemukan <b>{len(data)}</b> objek valid (Confidence ≥ {conf_thresh:.0%}). Menampilkan {len(shown)} teratas:</p>", unsafe_allow_html=True)
        
        if shown_display:
            st.dataframe(
                pd.DataFrame(shown_display), 
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Tidak ada objek yang memenuhi syarat minimal confidence.")
    else:
        st.warning("Tidak ada objek yang terdeteksi pada gambar input.")

    st.markdown("""
    <div class='info-box'>
        <span style=' font-weight: 600;'>YOLO (Single-Stage) vs Faster R-CNN (Two-Stage)</span><br/><br/>
        <b>Faster R-CNN:</b> Memiliki alur kerja dua tahap. Pertama, menggunakan <i>Region Proposal Network (RPN)</i> untuk menebak area yang kemungkinan berisi objek. Kedua, melakukan klasifikasi dan perbaikan kotak batas (ROI Pooling). Akurasinya sangat tinggi, tetapi relatif lambat (5-17 FPS).<br/><br/>
        <b>YOLO (You Only Look Once):</b> Memformulasikan deteksi objek murni sebagai masalah regresi tunggal. Gambar dievaluasi hanya dengan satu kali pass melalui neural network untuk langsung memprediksi probabilitas kelas dan koordinat bounding box secara simultan. Sangat cepat (45-140+ FPS) dan optimal untuk sistem real-time.
    </div>
    """, unsafe_allow_html=True)

if uploaded is None:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 1rem; margin-top: 1rem; background: rgba(128, 128, 128, 0.05); border-radius: 24px; border: 1px dashed rgba(128, 128, 128, 0.3);">
      <h2 style=" font-weight: 600; margin-bottom: 0.5rem;">Selamat Datang di Lab Interaktif</h2>
      <p style=" font-size: 1.1rem; max-width: 600px; margin: 0 auto;">Pilih file gambar pada panel sidebar di sebelah kiri untuk mulai mengeksplorasi teknik-teknik Computer Vision modern.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3 style=' margin-top: 4rem; margin-bottom: 1.5rem; font-size: 1.4rem;'>Kurikulum Pembelajaran</h3>", unsafe_allow_html=True)
    
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown("""
        <div class='feature-card'>
            <div class='feature-title' style=''>Modul 1</div>
            <div style=' font-weight:600; margin-bottom: 0.5rem;'>Dasar & Sinyal Citra</div>
            <div class='feature-desc'>Manipulasi spasial (Resize, Rotate, Crop) dan transformasi domain frekuensi (Gaussian Blur, Deteksi Tepi Sobel, FFT).</div>
        </div>
        """, unsafe_allow_html=True)
    with mc2:
        st.markdown("""
        <div class='feature-card'>
            <div class='feature-title' style=''>Modul 2</div>
            <div style=' font-weight:600; margin-bottom: 0.5rem;'>Machine Learning Klasik</div>
            <div class='feature-desc'>Penerapan algoritma Unsupervised Learning untuk Color Quantization berbasis K-Means Clustering.</div>
        </div>
        """, unsafe_allow_html=True)
    with mc3:
        st.markdown("""
        <div class='feature-card'>
            <div class='feature-title' style=''>Modul 3</div>
            <div style=' font-weight:600; margin-bottom: 0.5rem;'>CNN Klasifikasi</div>
            <div class='feature-desc'>Inferensi klasifikasi objek 1000 kelas menggunakan arsitektur ResNet-50 dengan konsep Skip Connection.</div>
        </div>
        """, unsafe_allow_html=True)
    with mc4:
        st.markdown("""
        <div class='feature-card'>
            <div class='feature-title' style=''>Modul 4</div>
            <div style=' font-weight:600; margin-bottom: 0.5rem;'>Deteksi Multi-Objek</div>
            <div class='feature-desc'>Deteksi spasial dan lokalisasi objek secara real-time memanfaatkan arsitektur Single-Stage YOLOv8.</div>
        </div>
        """, unsafe_allow_html=True)

else:
    try:
        img_rgb = load_image(uploaded)
        st.sidebar.image(img_rgb, caption="Pratinjau Input", width="stretch")

        if module == "Dasar & Sinyal Citra":
            module_basic(img_rgb)
        elif module == "Machine Learning Klasik":
            module_ml_classic(img_rgb)
        elif module == "CNN & Klasifikasi (ResNet-50)":
            module_cnn(img_rgb)
        elif module == "Deteksi Objek (YOLOv8)":
            module_yolo(img_rgb)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
        st.exception(e)
