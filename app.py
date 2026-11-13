"""
Streamlit Web Interface - Face Masker
Minimalist terminalesque UI with authentic camcorder touches:
- Discreet REC indicator & camcorder top bar
- Real dynamic timecode (TC HH:MM:SS:FF) synced to timeline scrubbing
- Viewfinder corner reticles on video previews
- Clean camcorder matte palette & Space Mono typography
"""

import base64
import inspect
import os
import re
import tempfile
import time
import cv2
import pandas as pd
import streamlit as st
from engine import FaceMasker, check_input_container


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
    f = int(frame_num % max(1, round(frame_rate)))
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

# Uploaded (non-anonymized) videos and outputs live in a dedicated folder and are
# purged after a while: Streamlit has no reliable session-end hook to delete them.
TEMP_DIR = os.path.join(tempfile.gettempdir(), "face-masker")
STALE_FILE_AGE_SEC = 6 * 3600
# Height of the rows of the ID table, so the face thumbnails are large enough to recognize
THUMB_ROW_HEIGHT = 110


def purge_stale_files() -> None:
    now = time.time()
    try:
        names = os.listdir(TEMP_DIR)
    except OSError:
        return
    for name in names:
        path = os.path.join(TEMP_DIR, name)
        try:
            if name.startswith("facemask_") and now - os.path.getmtime(path) > STALE_FILE_AGE_SEC:
                os.remove(path)
        except OSError:
            pass


def _remove_quietly(path):
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def safe_stem(filename: str) -> str:
    """Client-supplied upload name -> bare file stem with no path or special characters."""
    base = os.path.basename(filename.replace("\\", "/"))
    stem = os.path.splitext(base)[0]
    stem = re.sub(r"[^\w\-. ]", "_", stem).strip(" .")
    return stem or "video"


def get_session_paths(uploaded) -> tuple:
    """
    Write the upload to disk once per file, not on every Streamlit rerun.
    Input and output paths are unique per upload, so concurrent sessions never
    read, overwrite or delete each other's files.
    """
    if st.session_state.get("upload_id") != uploaded.file_id:
        # New upload: drop the previous temp files before writing the new one
        _remove_quietly(st.session_state.get("input_path"))
        _remove_quietly(st.session_state.get("output_path"))
        st.session_state.pop("result", None)  # analysis and thumbnails of the previous video
        os.makedirs(TEMP_DIR, exist_ok=True)
        purge_stale_files()
        # Client-controlled name: only reuse the extension if it is a known one
        ext = os.path.splitext(uploaded.name)[1].lower()
        suffix = ext if ext in (".mp4", ".mov", ".avi", ".mkv") else ".mp4"
        fd, path = tempfile.mkstemp(prefix="facemask_in_", suffix=suffix, dir=TEMP_DIR)
        with os.fdopen(fd, "wb") as f:
            f.write(uploaded.getbuffer())
        out_fd, out_path = tempfile.mkstemp(prefix="facemask_out_", suffix=".mp4", dir=TEMP_DIR)
        os.close(out_fd)
        st.session_state["upload_id"] = uploaded.file_id
        st.session_state["input_path"] = path
        st.session_state["output_path"] = out_path
    return st.session_state["input_path"], st.session_state["output_path"]


if uploaded_file is not None:
    input_video_path, output_video_path = get_session_paths(uploaded_file)

    # Validate the real container (by content) before OpenCV / ffmpeg parse it
    try:
        check_input_container(input_video_path)
    except (ValueError, OSError) as e:
        st.error(f"Rejected file: {e}")
        st.stop()

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

    small_faces = st.checkbox(
        "Detect small faces (crowds, wide shots)",
        value=False,
        help="Also analyzes two halves of each frame to catch tiny faces. About 3x slower."
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
            conf_threshold=conf_threshold,
            small_faces=small_faces
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

    # Download name only: uploaded_file.name is client-controlled, never use it in a path
    output_filename = f"masked_{safe_stem(uploaded_file.name)}.mp4"

    mask_range = None
    if duration_sec > 0.2 and st.checkbox(
        "Mask only a portion of the video",
        value=False,
        help="Faces outside the selected portion stay visible in the exported video."
    ):
        range_max = round(duration_sec + 0.05, 1)
        range_start, range_end = st.slider(
            "Portion to mask (seconds)",
            min_value=0.0,
            max_value=range_max,
            value=(0.0, range_max),
            step=0.1
        )
        if range_end <= range_start:
            st.warning("Select a portion with a non-zero duration.")
            st.stop()
        mask_range = (range_start, None if range_end >= range_max else range_end)
        st.caption(
            f"Masking from TC {format_timecode(int(range_start * fps), fps)} "
            f"to TC {format_timecode(min(total_frames, int(range_end * fps)), fps)} — "
            "faces stay visible elsewhere."
        )

    reid_ok = FaceMasker.reid_available()
    reidentify = st.checkbox(
        "Keep the same ID when a person re-enters the frame",
        value=False,
        disabled=not reid_ok,
        help=(
            "Compares faces within this video to recognize someone who left the frame and came back. "
            "Slower. Face signatures stay in memory and are discarded after processing."
            if reid_ok else
            "Recognition model missing: see the README to download face_recognition_sface_2021dec.onnx."
        )
    )

    def run_with_progress(work) -> float:
        """Run work(progress_callback) under a progress bar; returns the elapsed seconds."""
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

        with st.spinner("Processing video frames..."):
            work(update_progress)
        progress_bar.progress(1.0, text="Recording complete")
        return time.time() - start_time

    if st.button("● Record & Process Video", use_container_width=True):
        try:
            box = {}

            def first_run(update_progress):
                analysis = masker.analyze_video(
                    input_video_path,
                    conf_threshold=conf_threshold,
                    reidentify=reidentify,
                    small_faces=small_faces,
                    mask_range=mask_range,
                    progress_callback=update_progress
                )
                masker.render_video(
                    analysis,
                    output_video_path,
                    progress_callback=update_progress,
                    progress_offset=analysis.total_frames,
                    progress_total=analysis.total_frames * 2
                )
                box["analysis"] = analysis
                box["thumbs"] = masker.track_thumbnails(analysis)

            total_duration = run_with_progress(first_run)
            st.session_state["result"] = {
                "upload_id": uploaded_file.file_id,
                "analysis": box["analysis"],
                "thumbs": box["thumbs"],
                "excluded": set(),      # IDs left unmasked in the video currently shown
                "merges": {},           # merges applied to the video currently shown
                "unticked": set(),      # IDs unticked when the table was last rebuilt
                "table": st.session_state.get("result", {}).get("table", 0) + 1,
                "message": f"Video processed in {total_duration:.1f}s ({total_frames / total_duration:.1f} fps)",
            }
        except Exception as e:
            st.session_state.pop("result", None)
            st.error(f"Processing failed: {e}")

    # Kept in the session: the result must survive the reruns triggered by the ID table
    result = st.session_state.get("result")
    if result is not None and result["upload_id"] == uploaded_file.file_id:
        analysis = result["analysis"]
        st.success(result["message"])

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

        # Section 03 : Detected IDs
        st.markdown(
            f'<div class="section-bar"><span>Detected IDs</span>'
            f'<span class="section-tc">{len(analysis.tracks)} ID{"s" if len(analysis.tracks) != 1 else ""}</span></div>',
            unsafe_allow_html=True
        )

        if not analysis.tracks:
            st.caption("No face was tracked in this video.")
        else:
            def thumb_uri(track_id: int) -> str:
                img = result["thumbs"].get(track_id)
                if img is None:
                    return ""
                ok, png = cv2.imencode(".png", img)
                return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode("ascii") if ok else ""

            table = pd.DataFrame([
                {
                    "Mask": t.track_id not in result["unticked"],
                    "ID": t.track_id,
                    "Face": thumb_uri(t.track_id),
                    "First seen": format_timecode(t.first_frame, analysis.fps),
                    "Last seen": format_timecode(t.last_frame, analysis.fps),
                    "On screen (s)": round(t.n_frames / analysis.fps, 1),
                }
                for t in analysis.tracks
            ])
            st.caption(
                "Untick the IDs to leave unmasked, then process again. "
                "Thumbnails show the unmasked source so people can be told apart; they are not saved."
            )
            editor_options = {}
            if "row_height" in inspect.signature(st.data_editor).parameters:
                editor_options["row_height"] = THUMB_ROW_HEIGHT
            edited = st.data_editor(
                table,
                key=f"id_table_{result['table']}",
                hide_index=True,
                use_container_width=True,
                disabled=["ID", "Face", "First seen", "Last seen", "On screen (s)"],
                column_config={
                    "Mask": st.column_config.CheckboxColumn("Mask", help="Untick to leave this face visible"),
                    "Face": st.column_config.ImageColumn("Face", width="medium"),
                },
                **editor_options
            )
            wanted_excluded = {int(i) for i in edited.loc[~edited["Mask"].astype(bool), "ID"]}

            # Merging: the same person tracked under several IDs
            all_ids = [t.track_id for t in analysis.tracks]
            to_merge = st.multiselect(
                "Same person under several IDs? Select them to merge",
                options=all_ids,
                format_func=lambda i: f"ID {i}",
                key=f"merge_select_{result['table']}",
                help="The selected IDs become a single one (the lowest). The video is updated at the next processing."
            )
            col_merge, col_undo = st.columns(2)
            with col_merge:
                do_merge = st.button(
                    f"Merge {len(to_merge)} IDs" if len(to_merge) >= 2 else "Merge IDs",
                    use_container_width=True,
                    disabled=len(to_merge) < 2
                )
            with col_undo:
                do_undo = st.button("Undo all merges", use_container_width=True, disabled=not analysis.id_map)

            if do_merge:
                kept, together = masker.merge_ids(analysis, to_merge)
                # The merged ID stays masked unless every merged ID had been unticked
                merged = set(to_merge)
                unticked = wanted_excluded - merged
                if merged <= wanted_excluded:
                    unticked.add(kept)
                result["unticked"] = unticked
                result["table"] += 1  # rows changed: rebuild the table and the selector
                result["notice"] = (
                    f"IDs {', '.join(map(str, sorted(merged)))} merged into ID {kept}."
                    + (
                        f" Warning: they are on screen together on {together} frame"
                        f"{'s' if together != 1 else ''}, so they may be different people."
                        if together else ""
                    )
                )
                st.rerun()
            if do_undo:
                masker.reset_merges(analysis)
                result["unticked"] = {i for i in wanted_excluded if i in {t.track_id for t in analysis.tracks}}
                result["table"] += 1
                result["notice"] = "Merges undone."
                st.rerun()

            if result.get("notice"):
                (st.warning if "Warning" in result["notice"] else st.info)(result["notice"])

            if result["excluded"]:
                st.caption("Currently left unmasked: " + ", ".join(f"ID {i}" for i in sorted(result["excluded"])))

            n = len(wanted_excluded)
            label = (
                f"● Process Again — leave {n} ID{'s' if n != 1 else ''} unmasked" if n
                else "● Process Again — mask every ID"
            )
            up_to_date = wanted_excluded == result["excluded"] and analysis.id_map == result["merges"]
            if st.button(label, use_container_width=True, disabled=up_to_date):
                try:
                    rerender_s = run_with_progress(lambda update_progress: masker.render_video(
                        analysis,
                        output_video_path,
                        exclude_ids=wanted_excluded,
                        progress_callback=update_progress
                    ))
                    result["excluded"] = set(wanted_excluded)
                    result["merges"] = dict(analysis.id_map)
                    result["notice"] = None
                    result["message"] = (
                        f"Video processed again in {rerender_s:.1f}s, "
                        f"{n} ID{'s' if n != 1 else ''} left unmasked"
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Processing failed: {e}")
