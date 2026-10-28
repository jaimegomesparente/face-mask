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

---

## Project Structure

- `app.py`: Streamlit web application.
- `engine.py`: Core detection engine (YOLOv8-Face), targeted face pixelation, persistent ID tracking, and FFmpeg audio remuxing.
- `cli.py`: Command-line interface.
- `yolov8n-face.onnx`: Deep-learning face detector (~11.7 MB).
- `face_detection_yunet_2023mar.onnx`: Fallback lightweight detector (~230 KB).
- `launch_streamlit.bat`: Windows launcher.

---

## Models & Credits

- **YOLOv8n-face** (`yolov8n-face.onnx`): [lindevs/yolov8-face](https://github.com/lindevs/yolov8-face), trained on WIDER FACE with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics). Licensed under **AGPL-3.0**.
- **YuNet** (`face_detection_yunet_2023mar.onnx`): from [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet). Licensed under **MIT**.

## License

This project is licensed under the **GNU Affero General Public License v3.0** — see [LICENSE](LICENSE).
It bundles an Ultralytics YOLOv8-derived model, which is itself distributed under AGPL-3.0.
