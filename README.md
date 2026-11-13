# Face Masker

Video face anonymization tool with clean mosaic pixelation and tracking box:
- Crisp mosaic pixelation covering the full head region (face, forehead, chin, hairline).
- Green tracking box.
- Temporal stabilization to keep bounding boxes smooth across frames.
- Original audio track merged seamlessly via FFmpeg.

---

## Requirements

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) available on your `PATH` (used to re-encode the output and restore the audio track).
  - Windows: `winget install Gyan.FFmpeg`
  - macOS: `brew install ffmpeg`
  - Debian/Ubuntu: `sudo apt install ffmpeg`

## Installation

```bash
git clone https://github.com/jaimegomesparente/face-mask.git
cd face-mask
pip install -r requirements.txt
```

---

## Quick Start

### Option 1: Web Interface (Streamlit)
Run `launch_streamlit.bat` on Windows or:

```powershell
streamlit run app.py
```

The application opens in your web browser:
1. Select your video file (MP4, MOV, AVI, MKV).
2. Preview timeline and inspect detection.
3. Click "Process Video" and download the final MP4.

---

### Option 2: Command Line (CLI)
For batch pipelines or headless processing:

```powershell
python cli.py my_video.mp4 --output masked_output.mp4
```

Options:
- `-o, --output`: output path (default: `<name>_masked.mp4`).
- `--conf`: detection confidence threshold (default: 0.30).
- `--no-tracking`: disable temporal tracking and smoothing (per-frame detection only).
- `--reid`: keep the same ID when a person leaves the frame and comes back (off by default, see below).
- `--start`, `--end`: only mask faces between these times (seconds, `MM:SS` or `HH:MM:SS`). Either can be omitted. Faces outside this portion stay visible.

```powershell
python cli.py my_video.mp4 --start 0:12 --end 0:45.5
```

### Persistent IDs (optional)

By default a person who leaves the frame and comes back gets a new ID. With `--reid` (or the
matching checkbox in the web interface), faces are compared within the video so that the same
person keeps the same ID.

- Only the ID badges change: the masked areas are exactly the same with or without the option.
- Two cues are used: position continuity when a face is lost for a short time (up to 30 frames,
  whatever its size), and face comparison when someone leaves the frame and comes back.
- Face comparison needs faces of at least 24 px, and works best from 40 px and frontal. Below
  24 px (crowds, wide shots) different people cannot be told apart, so only short losses are bridged.
  When recognition fails the person simply gets a new ID, as without the option.
- The first pass is slower.
- Face signatures are computed in memory for the video being processed and are never written to disk.
- Needs `face_recognition_sface_2021dec.onnx` next to `engine.py`
  ([download](https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx)).

---

## Project Structure

- `app.py`: Streamlit web application.
- `engine.py`: Core detection engine (YOLOv8-Face), targeted face pixelation, persistent ID tracking, and FFmpeg audio remuxing.
- `cli.py`: Command-line interface.
- `yolov8n-face.onnx`: Deep-learning face detector (~11.7 MB).
- `face_detection_yunet_2023mar.onnx`: Fallback lightweight detector (~230 KB).
- `face_recognition_sface_2021dec.onnx`: Face recognition model for persistent IDs (~37 MB, optional).
- `launch_streamlit.bat`: Windows launcher.

---

## Models & Credits

- **YOLOv8n-face** (`yolov8n-face.onnx`): [lindevs/yolov8-face](https://github.com/lindevs/yolov8-face), trained on WIDER FACE with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics). Licensed under **AGPL-3.0**.
- **YuNet** (`face_detection_yunet_2023mar.onnx`): from [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet). Licensed under **MIT**.
- **SFace** (`face_recognition_sface_2021dec.onnx`): from [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface). Licensed under **Apache-2.0**.

## License

This project is licensed under the **GNU Affero General Public License v3.0** — see [LICENSE](LICENSE).
It bundles an Ultralytics YOLOv8-derived model, which is itself distributed under AGPL-3.0.
