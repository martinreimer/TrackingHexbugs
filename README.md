# Tracking Hexbugs Project

This project, developed for the *Tracking Olympiad* seminar, focuses on tracking Hexbugs—small, vibrating, bug-like devices that move randomly. The primary goal was to explore methods for accurately detecting and tracking the head of these objects in videos, especially given challenges such as low frame rates and erratic movements.

<div align="center">
    <img src="/media/tracking_gifs/run_214_head_b_ed_001o.gif" width="200" alt="Description of GIF">
</div>

---

## Table of Contents

1. [Experimental Setup](#experimental-setup)
2. [Setup Instructions](#setup-instructions)
3. [Project Structure](#project-structure)
4. [Methodology](#methodology)
5. [Results and Key Findings](#results-and-key-findings)

---

## Experimental Setup

### Dataset
- The dataset comprises **105 videos**:
  - **100 videos** were used for training and evaluation.
  - **5 videos** were reserved exclusively for testing.
- The videos feature a **static camera view** and multiple Hexbugs with the following challenges:
  - Hexbugs lying on their sides or backs.
  - Occlusions and collisions between objects.
  - Color variations, including underrepresented cases like black Hexbugs.
  - Variations in lighting conditions.

### Annotation Process
- Hexbug **head positions** were given as center points.
- To train YOLO, fixed-size bounding boxes were generated around the labeled head positions.

### Body Label Generation with SAM
- To enhance detection and tracking performance, an idea was to make the YOLO not only learn to detect the head but also to detect the body of the hexbugs. **Body labels** were generated using Meta's **Segment Anything Model (SAM)**.
- The process involved:
  - Segmenting the entire frame into potential objects using SAM.
  - Matching YOLO-detected head positions with SAM segments using Intersection over Union (IoU).
  - Selecting the best-matched segment for each head detection, refining it by expanding its size by 20%.
- These generated labels helped in training a more robust YOLO model with both head (class 0) and body (class 1) annotations.


### Project Flow
1. **Created Head Annotations**: Converted head center points into head bounding boxes.
2. **Generating Body Labels**: Used the **Segment Anything Model (SAM)** to create bounding boxes for Hexbug bodies.
3. **Detection Models**:
   - **Head-only Detection**: YOLO model trained with head labels only.
   - **Head and Body Detection**: YOLO model trained with both head (class 0) and SAM-generated body (class 1) labels.
4. **Tracking Algorithms**:
   - **Minimal Euclidean Distance Tracking**: Tracks Hexbugs across frames by matching detections based on proximity.
   - **Lucas-Kanade Optical Flow**: Uses optical flow to predict Hexbug motion between frames and refines tracking.
5. **Evaluation**: Models and algorithms were tested on **5 exclusive test videos** to analyze performance.

The project focused on exploring combinations of these detection and tracking methods to determine the most accurate and robust approach.

---

## Setup Instructions

### Environment Setup
1. **Python Environment:** Create Python environments using `requirements.txt` in the respective directories (e.g., GUI, tracking).
2. **Dependencies:** Install required libraries:
   ```bash
   pip install -r requirements.txt
3. **Ultralytics library:**
  - Use the custom forked version of the Ultralytics library provided in this project which includes more augmentation techniques.
  - Install it using: pip install path/to/ultralytics.zip


## Data and Models
- Dataset: Place videos or extracted frames in the data/ directory.
- YOLO Models:
  - head_black_best.pt: Head-only detection model.
  - head_body_model_best.pt: Head and body detection model.
- SAM Weights: Download and set up SAM weights for body label generation.


## Usage
1. Generate Body Labels: Run SAM4Labels/generate_body_labels.ipynb to generate body labels from head labels using SAM.
2. Training: Train YOLO detection models using src/training/train.ipynb.
3. Tracking Inference: Run tracking using src/tracking/tracking_inference.py.


## Project Structure
```md
Hexbug_Tracking_Project/
├── data/                      # Datasets (not included in repository)
├── gui/                       # GUI for labeling
│   ├── gui.py                 # Main labeling tool script
│   ├── requirements.txt       # GUI environment dependencies
│   └── validate_gui_output.ipynb  # Validation and visualization
├── models/                    # our-trained YOLO models
├── SAM4Labels/                # Body label generation using SAM
├── src/                       # Source code for detection, tracking, and visualization
│   ├── data_preparation/      # Preprocessing scripts
│   ├── tracking/              # Tracking inference
│   ├── training/              # YOLO training scripts
│   └── viz/                   # Visualization scripts
├── media/                     # Media (images, gifs, results)
│   └── trackinggifs/          # Demo gifs
└── results/                   # Results (e.g., tracking outputs, videos)
```


### Visuals

#### SAM Body Label Generation
This image shows the steps in using SAM for generating body labels:
- Top: Original frame.
- Middle: SAM-generated segments.
- Bottom: Resulting bounding boxes for heads (YOLO detections) and bodies (SAM refinement).
<div align="center">
    <img src="/media/sam.png" width="500" alt="Description of GIF">
</div>

#### Good and Bad Examples of SAM
This image shows two good examples where SAM worked well and two cases where it failed:

<div align="center">
    <img src="/media/sam_output.png" width="500" alt="Description of GIF">
</div>

#### Detection and Tracking Failures
This image shows common detection and tracking issues, such as:
- Missed detections.
- ID switches during collisions or occlusions.

<div align="center">
    <img src="/media/failings.png" width="500" alt="Description of GIF">
</div>

---

## Methodology

### Detection
Two detection approaches were implemented:
1. **Head-only Detection:**
   - YOLO was trained with head labels only (`class 0`).
2. **Head and Body Detection:**
   - YOLO was trained with both head (`class 0`) and body (`class 1`) labels.
   - Body labels were generated using SAM for better performance in tracking Hexbugs.

### Tracking
Two tracking methods were evaluated:
1. **Minimal Euclidean Distance Tracking:**
   - Tracks detected heads across frames based on the smallest Euclidean distance between detections in consecutive frames.
   - Assumes a fixed number of Hexbugs and handles missing detections by retaining their last known positions.
2. **Lucas-Kanade Optical Flow Tracking:**
   - Uses optical flow to predict object movement between frames.
   - Matches tracked objects with YOLO detections based on proximity.




---

## Results and Key Findings

### Detection
1. **Body-head detection** showed superior performance in scenarios with occlusions or collisions.
2. **Head-only detection** struggled in complex cases but performed well in simpler scenarios.

### Tracking
1. **Minimal Euclidean Distance Tracking:**
   - Simpler and faster but prone to ID switching in challenging situations (e.g., collisions).
2. **Lucas-Kanade Optical Flow Tracking:**
   - More robust to challenging scenarios but computationally more expensive.

### Challenges
- Frequent **ID switching** during collisions.
- Difficulty in **tracking Hexbugs** that lay flat or became partially obscured.
