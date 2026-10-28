"""
Streamlit Web Interface - Face Masker
Minimalist terminalesque UI with authentic camcorder touches:
- Discreet REC indicator & camcorder top bar
- Real dynamic timecode (TC HH:MM:SS:FF) synced to timeline scrubbing
- Viewfinder corner reticles on video previews
- Clean camcorder matte palette & Space Mono typography
"""

import os
import tempfile
import time
import cv2
import streamlit as st
from engine import FaceMasker


# Page configuration
st.set_page_config(
    page_title="Face Masker",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Camcorder Minimalist Stylesheet
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&display=swap');

    /* Global reset */
    header[data-testid="stHeader"] { display: none !important; }
    footer { display: none !important; }
    #MainMenu { display: none !important; }
    div[data-testid="stToolbar"] { display: none !important; }
    div[data-testid="stDecoration"] { display: none !important; }
    .stDeployButton { display: none !important; }

    /* Palette Camcorder Épurée */
    :root {
        --bg-main: #0c0e12;
        --bg-panel: #13161c;
        --border-color: #202530;
        --border-active: #323b4b;
        --text-primary: #e6e9f0;
        --text-muted: #6f7887;
        --cam-red: #d32f2f;
        --cam-amber: #d4952b;
        --font-mono: 'Space Mono', monospace;
    }

    html, body, .stApp {
        background-color: var(--bg-main) !important;
        background-image: 
            radial-gradient(ellipse 90% 55% at 50% -5%, #1e2533 0%, #12161f 50%, var(--bg-main) 100%) !important;
        background-attachment: fixed !important;
        color: var(--text-primary) !important;
        font-family: var(--font-mono) !important;
        -webkit-font-smoothing: antialiased;
    }

    .main .block-container {
        max-width: 780px !important;
        padding-top: 1.5rem !important;
        padding-bottom: 4rem !important;
    }

    /* Petite barre de statut camcorder sobre */
    .cam-status-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 8px;
        margin-bottom: 1.5rem;
        font-size: 0.72rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-muted);
    }

    .cam-rec-badge {
        color: var(--cam-red);
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    .cam-rec-dot {
        width: 8px;
        height: 8px;
        background-color: var(--cam-red);
        border-radius: 50%;
        display: inline-block;
    }

    /* En-tête */
    .app-header {
        margin-bottom: 1.75rem;
    }

    .app-title {
        font-family: var(--font-mono) !important;
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--text-primary);
        margin: 0 0 0.3rem 0;
    }

    .app-subtitle {
        font-family: var(--font-mono) !important;
        font-size: 0.85rem;
        color: var(--text-muted);
        margin: 0;
    }

    /* Grille de métadonnées */
    .meta-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        background: var(--bg-panel);
        border: 1px solid var(--border-color);
        margin-bottom: 1.75rem;
    }

    .meta-cell {
        padding: 10px 14px;
        border-left: 1px solid var(--border-color);
    }

    .meta-cell:first-child {
        border-left: none;
    }

    .meta-label {
        font-size: 0.68rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 3px;
    }

    .meta-val {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-primary);
    }

    /* Titres de sections avec timecode réel */
    .section-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.75rem;
        font-weight: 700;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 1.75rem;
        margin-bottom: 0.85rem;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 6px;
    }

    .section-tc {
        color: var(--cam-red);
        font-weight: 700;
        letter-spacing: 0.05em;
    }

    /* File Uploader épuré */
    div[data-testid="stFileUploaderDropzone"] {
        background-color: var(--bg-panel) !important;
        border: 1px dashed var(--border-color) !important;
        border-radius: 0px !important;
        padding: 24px 16px !important;
        transition: border-color 0.15s ease !important;
    }

    div[data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--border-active) !important;
    }

    div[data-testid="stFileUploaderDropzone"] button {
        background: var(--bg-main) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 0px !important;
        font-family: var(--font-mono) !important;
        font-size: 0.82rem !important;
        padding: 6px 14px !important;
    }

    div[data-testid="stFileUploaderDropzone"] span {
        font-family: var(--font-mono) !important;
        color: var(--text-muted) !important;
        font-size: 0.78rem !important;
    }

    /* Sliders */
    div[data-testid="stSlider"] label {
        color: var(--text-muted) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.72rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
        background-color: var(--cam-red) !important;
        box-shadow: 0 0 0 2px var(--bg-main) !important;
        border-radius: 0px !important;
        width: 12px !important;
        height: 18px !important;
    }

    div[data-testid="stSlider"] div[data-testid="stTickBar"] {
        display: none !important;
    }

    /* Cadres de prévisualisation nets */
    div[data-testid="stImage"] {
        border: 1px solid var(--border-color) !important;
        background: #000000 !important;
        padding: 2px !important;
    }

    div[data-testid="stImage"] img {
        border-radius: 0px !important;
    }

    div[data-testid="stVideo"] {
        border: 1px solid var(--border-color) !important;
        background: #000000 !important;
        padding: 2px !important;
    }

    .preview-tag {
        font-size: 0.7rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 5px;
    }

    .preview-tag.active {
        color: var(--cam-red);
    }

    /* Boutons Camcorder */
    .stButton > button, div[data-testid="stDownloadButton"] > button {
        background: var(--bg-panel) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 0px !important;
        font-family: var(--font-mono) !important;
        font-size: 0.88rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        padding: 12px 20px !important;
        cursor: pointer !important;
        width: 100% !important;
        transition: all 0.15s ease !important;
        margin-top: 6px !important;
    }

    .stButton > button:hover {
        background: #181d26 !important;
        border-color: var(--cam-red) !important;
        color: #ffffff !important;
    }

    div[data-testid="stDownloadButton"] > button {
        border-color: var(--border-color) !important;
        margin-top: 10px !important;
    }

    div[data-testid="stDownloadButton"] > button:hover {
        background: #181d26 !important;
        border-color: var(--cam-amber) !important;
        color: #ffffff !important;
    }

    /* Barre de progression */
    div[data-testid="stProgress"] {
        background: var(--bg-panel) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 0px !important;
        height: 14px !important;
        margin: 12px 0 !important;
    }

    div[data-testid="stProgress"] > div > div > div {
        background: var(--cam-red) !important;
        border-radius: 0px !important;
    }

    /* Alertes */
    div[data-testid="stAlert"] {
        background: var(--bg-panel) !important;
        border-left: 3px solid var(--cam-amber) !important;
        border-top: 1px solid var(--border-color) !important;
        border-right: 1px solid var(--border-color) !important;
        border-bottom: 1px solid var(--border-color) !important;
        border-radius: 0px !important;
        color: var(--text-primary) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.8rem !important;
    }

    @media (max-width: 600px) {
        .meta-grid {
            grid-template-columns: repeat(2, 1fr);
        }
        .meta-cell:nth-child(2n+1) {
            border-left: none;
        }
        .meta-cell:nth-child(n+3) {
            border-top: 1px solid var(--border-color);
        }
    }
</style>
""", unsafe_allow_html=True)


def format_timecode(frame_num: int, frame_rate: float) -> str:
    """Format actual frame into an authentic camcorder timecode: HH:MM:SS:FF."""
    if frame_rate <= 0:
        return "00:00:00:00"
    total_sec = frame_num / frame_rate
    h = int(total_sec // 3600)
    m = int((total_sec % 3600) // 60)
    s = int(total_sec % 60)
    f = int(frame_num % int(frame_rate if frame_rate > 0 else 25))
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


@st.cache_resource
def load_masker():
    return FaceMasker()


masker = load_masker()

# Petite barre camcorder épurée
st.markdown("""
<div class="cam-status-bar">
    <div class="cam-rec-badge">
        <span class="cam-rec-dot"></span>
        <span>REC / STANDBY</span>
    </div>
    <div>YOLOV8-FACE // ONNX</div>
</div>
""", unsafe_allow_html=True)

# En-tête
st.markdown("""
<div class="app-header">
    <div class="app-title">Face Masker</div>
    <div class="app-subtitle">Video face anonymization with audio preservation</div>
</div>
""", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader(
    "Select video file",
    type=["mp4", "mov", "avi", "mkv"],
    label_visibility="collapsed"
)

def _remove_quietly(path):
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def get_input_video_path(uploaded) -> str:
    """Write the upload to disk once per file, not on every Streamlit rerun."""
    if st.session_state.get("upload_id") != uploaded.file_id:
        # New upload: drop the previous temp files before writing the new one
        _remove_quietly(st.session_state.get("input_path"))
        _remove_quietly(st.session_state.get("output_path"))
        suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
        fd, path = tempfile.mkstemp(prefix="facemask_in_", suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(uploaded.getbuffer())
        st.session_state["upload_id"] = uploaded.file_id
        st.session_state["input_path"] = path
        st.session_state["output_path"] = None
    return st.session_state["input_path"]


if uploaded_file is not None:
    temp_dir = tempfile.gettempdir()
    input_video_path = get_input_video_path(uploaded_file)

    # Metadata extraction
    cap = cv2.VideoCapture(input_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0
    cap.release()

    # Grille de métadonnées
    st.markdown(f"""
    <div class="meta-grid">
        <div class="meta-cell">
            <div class="meta-label">Resolution</div>
            <div class="meta-val">{width}x{height}</div>
        </div>
        <div class="meta-cell">
            <div class="meta-label">Framerate</div>
            <div class="meta-val">{fps:.1f} fps</div>
        </div>
        <div class="meta-cell">
            <div class="meta-label">Duration</div>
            <div class="meta-val">{duration_sec:.1f}s</div>
        </div>
        <div class="meta-cell">
            <div class="meta-label">Total Frames</div>
            <div class="meta-val">{total_frames}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Section 01 : Calibration & Timeline (avec vrai timecode camcorder dynamique)
    col_slide1, col_slide2 = st.columns([3, 2])
    with col_slide1:
        if total_frames > 1:
            frame_idx = st.slider(
                "Timeline position",
                min_value=0,
                max_value=max(0, total_frames - 1),
                value=min(total_frames // 4, total_frames - 1),
                step=1
            )
        else:
            frame_idx = 0

    with col_slide2:
        conf_threshold = st.slider(
            "Sensitivity",
            min_value=0.10,
            max_value=0.60,
            value=0.30,
            step=0.02,
            help="Lower values increase sensitivity on side angles and profile faces."
        )

    current_tc = format_timecode(frame_idx, fps)
    st.markdown(f"""
    <div class="section-bar">
        <span>Preview & Calibration</span>
        <span class="section-tc">TC {current_tc}</span>
    </div>
    """, unsafe_allow_html=True)

    # Preview frame extraction
    cap = cv2.VideoCapture(input_video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame_bgr = cap.read()
    cap.release()

    if ret and frame_bgr is not None:
        processed_bgr, face_count = masker.process_frame(
            frame_bgr,
            conf_threshold=conf_threshold
        )

        orig_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        proc_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)

        col_orig, col_proc = st.columns(2)
        with col_orig:
            st.markdown('<div class="preview-tag">Source Feed</div>', unsafe_allow_html=True)
            st.image(orig_rgb, width="stretch")

        with col_proc:
            face_plural = "faces" if face_count != 1 else "face"
            st.markdown(f'<div class="preview-tag active">Masked ({face_count} {face_plural})</div>', unsafe_allow_html=True)
            st.image(proc_rgb, width="stretch")

    # Section 02 : Render & Export
    st.markdown('<div class="section-bar"><span>Render & Export</span></div>', unsafe_allow_html=True)

    output_filename = f"masked_{uploaded_file.name}"
    if not output_filename.endswith(".mp4"):
        output_filename = os.path.splitext(output_filename)[0] + ".mp4"
    output_video_path = os.path.join(temp_dir, output_filename)
    st.session_state["output_path"] = output_video_path

    if st.button("● Record & Process Video", use_container_width=True):
        progress_bar = st.progress(0, text="Initializing...")
        start_time = time.time()

        def update_progress(current: int, total: int):
            pct = min(1.0, current / total) if total > 0 else 0.0
            elapsed = time.time() - start_time
            fps_proc = current / elapsed if elapsed > 0 else 0
            remaining = (total - current) / fps_proc if fps_proc > 0 else 0
            progress_bar.progress(
                pct,
                text=f"REC {current}/{total} frames ({int(pct * 100)}%) - {fps_proc:.1f} fps - {int(remaining)}s remaining"
            )

        try:
            with st.spinner("Processing video frames..."):
                masker.process_video(
                    input_path=input_video_path,
                    output_path=output_video_path,
                    conf_threshold=conf_threshold,
                    progress_callback=update_progress
                )

            total_duration = time.time() - start_time
            progress_bar.progress(1.0, text="Recording complete")
            st.success(f"Video processed in {total_duration:.1f}s ({total_frames / total_duration:.1f} fps)")

            with open(output_video_path, "rb") as vf:
                video_bytes = vf.read()

                st.video(video_bytes)

                st.download_button(
                    label="Download Masked Video",
                    data=video_bytes,
                    file_name=output_filename,
                    mime="video/mp4",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Processing failed: {e}")
