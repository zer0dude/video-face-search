# AI Video Face Search 🔍

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://video-face-search-wxubbcrf3tufe2xczy6zlb.streamlit.app/)
*Live Demo: [video-face-search-wxubbcrf3tufe2xczy6zlb.streamlit.app](https://video-face-search-wxubbcrf3tufe2xczy6zlb.streamlit.app/)*

**AI Video Face Search** is a professional computer vision application designed to automate the process of finding specific people in video footage. Originally built to assist media production teams in sifting through hours of raw footage, this tool allows users to "enroll" a person using a few reference photos and instantly generate a timeline of their appearances in a target video.

![Demo Screenshot](assets/demo_screenshot.png)

## 🚀 Features

*   **One-Shot Enrollment**: Learn a person's facial identity from just a few reference images using Deep Learning.
*   **High-Performance Scanning**: Scans video files (MP4, MOV) to detect and recognize faces frame-by-frame.
*   **Smart Timeline**: Generates a detailed log of time segments (e.g., "00:12 - 00:15") where the person appears.
*   **Visual Analytics**: Provides an interactive bar chart timeline to visualize presence density.
*   **Configurable Performance**: Adjustable sampling rates and resolution scaling to balance speed vs. accuracy.

## 🛠️ Technology Stack

This project leverages state-of-the-art open-source libraries:

*   **Frontend**: [Streamlit](https://streamlit.io/) for a responsive, interactive UI.
*   **Face Recognition**: [Facenet-PyTorch](https://github.com/timesler/facenet-pytorch) (Inception Resnet V1) for generating 512-dimensional face embeddings.
*   **Face Detection**: MTCNN (Multi-task Cascaded Convolutional Networks).
*   **Video Processing**: [OpenCV](https://opencv.org/) for efficient frame extraction and manipulation.
*   **Data Analysis**: [Pandas](https://pandas.pydata.org/) for result aggregation and visualization.

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/zer0dude/video-face-search.git
    cd video-face-search
    ```

2.  **Create a virtual environment** (Recommended):
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Model Weights** (Optional but recommended):
    The app will attempt to download weights automatically, but you can pre-download `vggface2.pt` to the `assets/` folder to speed up the first run.

## 🎮 Usage

Run the application locally:

```bash
streamlit run app.py
```

### Workflow
1.  **Select Data**: Upload a video and reference photos, or use the built-in "Demo Mode" (Obama/Merkel).
2.  **Enroll**: Click "Enroll on Images" to teach the AI the target identity.
3.  **Configure**: Adjust "Sampling Rate" (check every N frames) and "Resolution" based on your needs.
4.  **Analyze**: Start the analysis and view the generated timeline and report.

## 👨‍💻 Author

**Brian Jin** - AI Applications Engineer

*   🌐 **Website**: [brianjin.eu](https://brianjin.eu/)
*   💻 **GitHub**: [zer0dude](https://github.com/zer0dude)

---
*Built as a technical portfolio demonstration of applied Computer Vision.*
