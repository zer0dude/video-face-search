# Technical Architecture

This document outlines the technical design and implementation details of the **AI Video Face Search** application.

## System Overview

The application is built as a linear pipeline that processes visual data in two stages: **Enrollment** and **Inference**.

### 1. Enrollment Pipeline (One-Shot Learning)
The goal of enrollment is to create a robust vector representation (embedding) of the target person's face.

1.  **Input**: List of reference images (uploaded by user).
2.  **Preprocessing**: Images are converted to RGB.
3.  **Detection (MTCNN)**: The Multi-task Cascaded Convolutional Network detects faces in each image. If multiple faces are found, the largest/most prominent one is selected.
4.  **Embedding (Inception Resnet V1)**: The detected face crop is passed through a pre-trained Inception Resnet V1 model (trained on VGGFace2). This outputs a **512-dimensional vector**.
5.  **Aggregation**: Embeddings from all reference images are averaged (mean pooling) to create a single "canonical" embedding for the person. This reduces the impact of outliers (e.g., bad lighting in one photo).

### 2. Inference Pipeline (Video Analysis)
The inference stage scans the video to find matches against the enrolled embedding.

1.  **Frame Extraction**: OpenCV reads the video file.
2.  **Optimization**:
    *   **Sampling**: Only every $N$-th frame is processed (user configurable).
    *   **Resizing**: Frames are downscaled (e.g., 0.5x) to speed up detection.
3.  **Detection & Embedding**: Similar to enrollment, MTCNN finds faces in the frame, and Inception Resnet V1 generates embeddings for them.
4.  **Matching (Cosine Similarity)**:
    *   We calculate the cosine distance between the *frame face embedding* and the *enrolled target embedding*.
    *   **Threshold**: If the distance is < `0.6` (configurable), it is considered a match.
5.  **Result Storage**: Matches are stored with timestamp and frame index.

## Component Design

### `core/identifier.py`
*   **`FaceIdentifier` Class**: Encapsulates the PyTorch models.
    *   Loads `vggface2.pt` weights (local or download).
    *   Handles the complexity of tensor conversions and device management (CPU/CUDA).
    *   Implements `strict=False` loading to bypass unnecessary classification layers in the pre-trained model.

### `core/processor.py`
*   **`VideoProcessor` Class**: Manages the video I/O.
    *   Uses `cv2.VideoCapture` for efficient decoding.
    *   Implements a generator/callback pattern to update the UI progress bar without blocking the main thread entirely.

### `utils/ui_helpers.py`
*   **Aggregation Logic**: The raw frame matches are often "noisy" (e.g., match, match, miss, match). The `aggregate_segments` function smooths this data by grouping consecutive matches into continuous time segments, allowing for small gaps (dropout).

## Performance Considerations

*   **Model Choice**: `facenet-pytorch` was chosen over `dlib` for better Windows compatibility and easier installation (pure Python/PyTorch vs. C++ compilation).
*   **Inference Speed**: On a standard CPU, processing every frame is slow (~5-10 FPS). By default, we sample every 10 frames and downscale by 50%, achieving >30 FPS processing speed (faster than real-time).
