"""
This script performs test-time inference for detection and tracking algorithms on a single video. 
It combines YOLOv8-based object detection with two tracking methods: Euclidean Distance Tracking 
and Lucas-Kanade Optical Flow Tracking. The pipeline processes one video at a time, evaluates detections, 
performs tracking, visualizes results, and exports tracking data as a CSV file.

Key Features:
1. **Detection Approaches:**
   - YOLO with Head Labels: Detections are made using a YOLO model trained on head labels (class=0).
   - YOLO with Body and Head Labels: Detections include both body (class=1) and head (class=0) labels.

2. **Tracking Algorithms:**
   - Euclidean Distance Tracking: Matches detections to tracked objects based on minimal Euclidean distance.
   - Lucas-Kanade Optical Flow Tracking: Predicts object movements between frames using optical flow.

3. **Pipeline:**
   - Determines optimal detection confidence based on stability across frames.
   - Filters detections to ensure head detections are accurate and within body bounding boxes (if applicable).
   - Tracks objects over frames, maintaining consistent IDs.
   - Saves annotated frames and exports tracking results to a CSV file.

Outputs:
- Annotated frames with bounding boxes and object IDs.
- CSV file with tracking results, including object coordinates per frame.

"""

import ultralytics 
import sys
import cv2
import numpy as np
import os
import math
import shutil
from ultralytics import YOLO
import copy
import torch
import csv

EXIT_FRAMES_CRITERIA = 101

# Configuration
videoname = 'test005'
project_dir = 'C:/Users/marti/Documents/courses/Traco'
input_videos_folder_path = os.path.join(project_dir, 'data/Leaderboarddata')
cur_dir_path = os.path.join(project_dir, 'src/tracking')
output_general_folder_path = os.path.join(cur_dir_path, 'tracking_test_runs')

# Set up output folder
os.makedirs(output_general_folder_path, exist_ok=True)
existing_folders = os.listdir(output_general_folder_path)
highest_number = 0
for folder in existing_folders:
    if folder.startswith('run_'):
        number = int(folder.split('_')[1])
        if number > highest_number:
            highest_number = number
new_folder_name = f'run_{highest_number+1:03d}'
output_folder_path = os.path.join(output_general_folder_path, new_folder_name)
os.makedirs(output_folder_path, exist_ok=True)

# Paths
video_input_path = os.path.join(input_videos_folder_path, f'{videoname}.mp4')
csv_tracking_path = os.path.join(output_folder_path, f'{videoname}.csv')
video_output_path = os.path.join(output_folder_path, f'{videoname}_original.mp4')
frames_output_folder = os.path.join(output_folder_path, 'frames')
os.makedirs(frames_output_folder, exist_ok=True)
shutil.copyfile(video_input_path, video_output_path)

# Model loading
print("Loading models...")
model_path_body = os.path.join(cur_dir_path, 'models', 'head_body_model_best.pt')
model_path_head = os.path.join(cur_dir_path, 'models', 'head_black_best.pt')
model_body_abs_path = os.path.abspath(model_path_body)
model_head_abs_path = os.path.abspath(model_path_head)
yolo_model_body = YOLO(model_body_abs_path)
yolo_model_head = YOLO(model_head_abs_path)

# Utility Functions

def euclidean(x1, x2, y1, y2):
    """
    Calculates the Euclidean distance between two points.
    - Inputs:
        x1, y1: Coordinates of the first point.
        x2, y2: Coordinates of the second point.
    - Output:
        Distance between the two points.
    """
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def analyze_detections(model, video, base_conf=0.2, conf_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], iou=0.4, class_id=0):
    """
    Determines the optimal confidence threshold for YOLO detections by evaluating detection stability across frames.
    - Inputs:
        model: YOLO model used for inference.
        video: Path to the input video.
        base_conf: Base confidence level for initial inference.
        conf_levels: List of confidence levels to evaluate.
        iou: Intersection over Union threshold.
        class_id: Class ID for the detections to analyze.
    - Outputs:
        Optimal confidence level and median object count per frame.
    """
    print("Analyzing body detections...")
    cap = cv2.VideoCapture(video)
    detection_counts = {conf: [] for conf in conf_levels}
    for conf in conf_levels:
        detection_counts[conf] = [0] * int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    success = True
    frame_idx = 0
    while success:
        print(f"Frame {frame_idx}")
        success, frame = cap.read()
        if not success:
            break

        # Perform inference with a very low confidence threshold
        result = model(frame, conf=base_conf, iou=iou, classes=[class_id])
        
        for res in result:
            for r in res.boxes.data.tolist():
                _, _, _, _, score, class_id = r
                for conf in conf_levels:
                    if score >= conf:
                        detection_counts[conf][frame_idx] += 1
        frame_idx += 1
    cap.release()

    # Compute stability scores (variance) for each confidence level
    stability_scores = {conf: np.std(detection_counts[conf]) for conf in conf_levels}
    optimal_conf = min(stability_scores, key=stability_scores.get)
    # get median of counts for optimal confidence
    median_count = int(np.median(detection_counts[optimal_conf]))
    print(f"Optimal confidence level: {optimal_conf}")
    print(f"Stability scores: {stability_scores}")
    print(f"Median count per frame: {median_count}")
    print(f"Detection counts: {detection_counts[optimal_conf]}")
    print(detection_counts)
    return optimal_conf, median_count

def point_in_box(x, y, box):
    """
    Checks if a point lies within a bounding box.
    - Inputs:
        x, y: Coordinates of the point.
        box: Bounding box as (x1, y1, x2, y2).
    - Output:
        True if the point is inside the box; False otherwise.
    """
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

def filter_heads_within_bodies(yolo_model_body, yolo_model_head, video, body_conf, head_conf=0.1):
    """
    Filters head detections that lie within body bounding boxes.
    
    - Inputs:
        yolo_model_body: YOLO model trained to detect body.
        yolo_model_head: YOLO model trained to detect head.
        video: Path to the video file.
        body_conf: Confidence threshold for body detections.
        head_conf: Confidence threshold for head detections.
    
    - Returns:
        A list of tuples (body results, filtered head results, frames) for all frames in the video.
    """
    cap = cv2.VideoCapture(video)
    all_frame_results = []
    success = True
    frame_idx = 0
    while success:
        print(f"Frame {frame_idx}")
        success, frame = cap.read()
        if not success:
            break

        body_results = yolo_model_body(frame, conf=body_conf, iou=0.4, classes=[1])
        head_results = yolo_model_body(frame, conf=head_conf, iou=0.4, classes=[0])

        filtered_heads = []

        for body in body_results[0].boxes:
            best_head = None
            best_head_conf = 0
            for head in head_results[0].boxes:
                head_center = [(head.xyxy[0][0] + head.xyxy[0][2]) / 2, (head.xyxy[0][1] + head.xyxy[0][3]) / 2]
                if point_in_box(head_center[0], head_center[1], body.xyxy[0].tolist()):
                    if head.conf > best_head_conf:
                        best_head = head
                        best_head_conf = head.conf
            if best_head is not None:
                filtered_heads.append(best_head)

        all_frame_results.append((body_results[0].boxes, filtered_heads, frame))
        frame_idx += 1
        if frame_idx == EXIT_FRAMES_CRITERIA:
            break

    cap.release()
    return all_frame_results



def filter_heads(yolo_model_head, video, body_conf, head_conf=0.1):
    """
    Filters head detections directly without relying on body detections.
    
    - Inputs:
        yolo_model_head: YOLO model trained to detect head.
        video: Path to the video file.
        body_conf: Placeholder (not used here).
        head_conf: Confidence threshold for head detections.
    
    - Returns:
        A list of tuples (None, filtered head results, frames) for all frames in the video.
    """
    cap = cv2.VideoCapture(video)
    all_frame_results = []
    success = True
    frame_idx = 0
    while success:
        print(f"Frame {frame_idx}")
        success, frame = cap.read()
        if not success:
            break

        head_results = yolo_model_head(frame, conf=head_conf, iou=0.4, classes=[0])

        filtered_heads = []

        
        for head in head_results[0].boxes:
            filtered_heads.append(head)

        all_frame_results.append((None, filtered_heads, frame))
        frame_idx += 1
        if frame_idx == EXIT_FRAMES_CRITERIA:
            break

    cap.release()
    return all_frame_results

class Head:
    """
    A class representing a detected head object.
    
    - Attributes:
        xyxy: Bounding box coordinates as [x1, y1, x2, y2].
        conf: Confidence score of the detection.
    """
    def __init__(self, xyxy, conf):
        self.xyxy = xyxy
        self.conf = conf



def tracking_via_minimal_euclidean_distance(filtered_head_detections, init_heads=4, threshold_distance=300, max_disappearance=100):
    """
    Tracks detected objects using minimal Euclidean distance.
    
    - Inputs:
        filtered_head_detections: Filtered detections for all frames.
        init_heads: Initial number of heads to track.
        threshold_distance: Maximum distance for matching detections.
        max_disappearance: Maximum allowed frames for an object to be missing before removing it.
    
    - Returns:
        A list of tuples (frame index, tracked objects) with tracked object positions and IDs.
    """
    print("Tracking heads...")
    tracked_objects = {}  # Keeps track of currently active objects and their IDs
    next_object_id = 0  # ID counter for new objects if needed

    tracking_results = []  # Store tracking results per frame

    # Initialize the IDs in the first frame
    for frame_idx, (body_results, head_results, _) in enumerate(filtered_head_detections):
        current_frame_objects = []  # Objects tracked in the current frame
        previous_ids = set(tracked_objects.keys())
        used_ids = set()  # Track IDs that have been used in this frame

        if frame_idx == 0:
            print("First frame")
            # Initialize a fixed number of IDs even if some are dummy heads
            for i in range(init_heads):
                if i < len(head_results):
                    head = head_results[i]
                    head_center = [(head.xyxy[0][0] + head.xyxy[0][2]) / 2, (head.xyxy[0][1] + head.xyxy[0][3]) / 2]
                    tracked_objects[next_object_id] = {'position': head_center, 'last_frame': frame_idx}
                    current_frame_objects.append((next_object_id, head))
                else:
                    # Initialize dummy heads with placeholder positions
                    tracked_objects[next_object_id] = {'position': [i * 10, i * 10], 'last_frame': frame_idx}
                    new_head = Head([torch.tensor([i * 10, i * 10, i * 10 + 10, i * 10 + 10])], 0.5)
                    current_frame_objects.append((next_object_id, new_head))
                
                next_object_id += 1

        else:
            # Create a list of tracked IDs
            tracked_ids = list(tracked_objects.keys())
            n_tracked = len(tracked_ids)
            n_detected = len(head_results)

            # Create distance matrix between tracked heads and new detections
            distance_matrix = np.full((n_tracked, n_detected), np.inf)  # Initialize with large values

            # Calculate distances only for unassigned IDs
            for i, obj_id in enumerate(tracked_ids):
                obj = tracked_objects[obj_id]
                for j, head in enumerate(head_results):
                    head_center = [(head.xyxy[0][0] + head.xyxy[0][2]) / 2, (head.xyxy[0][1] + head.xyxy[0][3]) / 2]
                    distance_matrix[i, j] = euclidean(head_center[0], obj['position'][0], head_center[1], obj['position'][1])

            # List to keep track of used IDs in the current frame
            used_ids = set()

            # Assign IDs based on the minimum distance
            for j, head in enumerate(head_results):
                min_dist = float('inf')
                best_match_id = None

                for i, obj_id in enumerate(tracked_ids):
                    if obj_id not in used_ids and distance_matrix[i, j] < min_dist and distance_matrix[i, j] < threshold_distance:
                        min_dist = distance_matrix[i, j]
                        best_match_id = obj_id

                if best_match_id is not None:
                    used_ids.add(best_match_id)
                    head_center = [(head.xyxy[0][0] + head.xyxy[0][2]) / 2, (head.xyxy[0][1] + head.xyxy[0][3]) / 2]
                    tracked_objects[best_match_id]['position'] = head_center
                    tracked_objects[best_match_id]['last_frame'] = frame_idx
                    current_frame_objects.append((best_match_id, head))

            # Handle missing detections
            missing_ids = previous_ids - used_ids
            for obj_id in missing_ids:
                # Use the last known position for missing IDs
                last_position = tracked_objects[obj_id]['position']
                last_head = Head([torch.tensor([last_position[0] - 5, last_position[1] - 5, last_position[0] + 5, last_position[1] + 5])], 0.5)
                tracked_objects[obj_id]['position'] = last_position  # Reuse last known position
                tracked_objects[obj_id]['last_frame'] = frame_idx
                current_frame_objects.append((obj_id, last_head))

        # Mark objects for removal if they have disappeared for too long
        for obj_id in list(tracked_objects.keys()):
            if frame_idx - tracked_objects[obj_id]['last_frame'] > max_disappearance:
                del tracked_objects[obj_id]  # Remove old objects

        tracking_results.append((frame_idx, current_frame_objects))

    return tracking_results



def tracking_via_lucas_kanade(filtered_head_detections, init_heads=4, max_disappearance=100):
    """
    Tracks detected objects using the Lucas-Kanade optical flow method.

    - Inputs:
        filtered_head_detections: Filtered detections for all frames.
        init_heads: Initial number of heads to track.
        max_disappearance: Maximum allowed frames for an object to be missing before removing it.

    - Returns:
        A list of tuples (frame index, tracked objects) with tracked object positions and IDs.
    """
    print("Tracking heads with Lucas-Kanade optical flow...")
    tracked_objects = {}  # Keeps track of currently active objects and their IDs
    next_object_id = 0  # ID counter for new objects if needed
    previous_gray = None  # For storing the previous frame in grayscale
    # get width and height of the frame
    width = filtered_head_detections[0][2].shape[1]
    height = filtered_head_detections[0][2].shape[0]
    print(f"Width: {width}, Height: {height}")
    tracking_results = []  # Store tracking results per frame

    for frame_idx, (body_results, head_results, frame) in enumerate(filtered_head_detections):
        current_frame_objects = []  # Objects tracked in the current frame
        previous_ids = set(tracked_objects.keys())
        used_ids = set()  # Track IDs that have been used in this frame
        used_head_ids = set()  # Track head IDs that have been used in this frame
        # Convert the current frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_idx == 0:
            print("First frame")
            # Initialize a fixed number of IDs
            for i in range(init_heads):
                if i < len(head_results):
                    head = head_results[i]
                    head_center = [(head.xyxy[0][0] + head.xyxy[0][2]) / 2, (head.xyxy[0][1] + head.xyxy[0][3]) / 2]
                    tracked_objects[next_object_id] = {'position': head_center, 'last_frame': frame_idx}
                    current_frame_objects.append((next_object_id, head))
                else:
                    # Initialize dummy heads with placeholder positions
                    tracked_objects[next_object_id] = {'position': [i * 10, i * 10], 'last_frame': frame_idx}
                    new_head = Head([torch.tensor([i * 10, i * 10, i * 10 + 10, i * 10 + 10])], 0.5)
                    current_frame_objects.append((next_object_id, new_head))
                
                next_object_id += 1
            previous_gray = gray

        else:
            if previous_gray is not None:
                # Prepare points for optical flow estimation
                old_points = np.array([tracked_objects[obj_id]['position'] for obj_id in tracked_objects.keys()], dtype=np.float32).reshape(-1, 1, 2)
                
                if len(old_points) > 0:
                    # Calculate optical flow to predict new positions
                    new_points, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, gray, old_points, None, winSize=(100,100), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.01))
                    # check if new points are within the frame, if not use old points
                    for i in range(len(new_points)):
                        if new_points[i][0][0] < 0 or new_points[i][0][0] > width or new_points[i][0][1] < 0 or new_points[i][0][1] > height:
                            print(f"detected point outside of frame {frame_idx}: {new_points[i]}")
                            new_points[i] = old_points[i]
                    # plot and save the optical flow predictions and the old points
                    
                    for i in range(len(old_points)):
                        cv2.circle(frame, (int(old_points[i][0][0]), int(old_points[i][0][1])), 5, (0, 0, 255), 5)
                        cv2.circle(frame, (int(new_points[i][0][0]), int(new_points[i][0][1])), 5, (0, 255, 0), 5)
                    # use output path to write the images
                    cv2.imwrite(output_folder_path + f'/frame_{frame_idx:04d}.jpg', frame)
                    
                    #new_points = new_points[status.flatten() == 1]  # Filter out points that did not track
                    tracked_ids = list(tracked_objects.keys())
                    
                    # Create a list to keep track of the closest detected heads
                    distances = np.full(len(new_points), np.inf)
                    
                    for i, new_point in enumerate(new_points):
                        min_dist = float('inf')
                        best_match_id = None
                        best_head_center = None
                        for j, head in enumerate(head_results):
                            if j in used_head_ids:
                                continue
                            head_center = [(head.xyxy[0][0] + head.xyxy[0][2]) / 2, (head.xyxy[0][1] + head.xyxy[0][3]) / 2]
                            dist = np.linalg.norm(np.array(head_center) - new_point.flatten())
                            if dist < min_dist:
                                min_dist = dist
                                best_match_id = j
                                best_head_center = head_center
                        if best_match_id is not None:
                            used_ids.add(tracked_ids[i])
                            used_head_ids.add(best_match_id)
                            head = head_results[best_match_id]
                            current_frame_objects.append((tracked_ids[i], head))
                            tracked_objects[tracked_ids[i]]['position'] = best_head_center
                            distances[i] = min_dist

                    # Handle missing detections
                    
                    missing_ids = set(tracked_ids) - used_ids
                    for obj_id in missing_ids:
                        # if there are ids missing, we will use the ids from the previous frame and as points use the lucas kanade estimation
                        # get index of the missing id in the tracked_ids list
                        obj_id_idx = tracked_ids.index(obj_id)
                        # get the point from the new_points list
                        new_point = old_points[obj_id_idx]
                        last_head = Head([torch.tensor([new_point.flatten()[0] - 5, new_point.flatten()[1] - 5, new_point.flatten()[0] + 5, new_point.flatten()[1] + 5])], 0.5)
                        tracked_objects[obj_id]['position'] = [torch.tensor(new_point.flatten()[0]), torch.tensor(new_point.flatten()[1])]  # Reuse last known position
                        tracked_objects[obj_id]['last_frame'] = frame_idx
                        current_frame_objects.append((obj_id, last_head))
                        used_ids.add(obj_id)
                    print(f"Still missing ids: {set(tracked_ids) - used_ids}")
                    
            # Update the previous frame for the next iteration
            previous_gray = gray


        tracking_results.append((frame_idx, current_frame_objects))

    return tracking_results


def draw_bounding_boxes(image, bounding_boxes, labels, confidences, object_ids):
    """
    Draw bounding boxes with labels, confidence scores, and IDs on the image.

    - Inputs:
        image: The image on which bounding boxes will be drawn.
        bounding_boxes: List of bounding box coordinates [(x1, y1, x2, y2), ...].
        labels: List of labels corresponding to the bounding boxes.
        confidences: List of confidence scores for each bounding box.
        object_ids: List of IDs for tracked objects.

    - Returns:
        Annotated image with bounding boxes and labels.
    """
    for bbox, label, confidence, obj_id in zip(bounding_boxes, labels, confidences, object_ids):
        x1, y1, x2, y2 = map(int, bbox)
        color = (0, 0, 255)  # Red for head boxes
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        text = f"{label}: {confidence:.2f}, ID: {obj_id}"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, cv2.FILLED)
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return image


def save_frames(tracking_results, output_folder, video_path):
    """
    Save annotated frames with bounding boxes for each frame in the video.

    - Inputs:
        tracking_results: List of tracking results for all frames.
        output_folder: Directory where annotated frames will be saved.
        video_path: Path to the input video file.
    """
    print("Saving frames...")
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    frame_idx = 0

    while success:
        print(f"Frame {frame_idx}")
        frame_data = tracking_results[frame_idx]
        current_frame_objects = frame_data[1]
        bounding_boxes = []
        labels = []
        confidences = []
        object_ids = []

        for obj_id, head in current_frame_objects:
            bounding_boxes.append(head.xyxy[0].tolist())
            labels.append("Head")
            confidences.append(float(head.conf))
            object_ids.append(obj_id)

        image_with_boxes = draw_bounding_boxes(frame, bounding_boxes, labels, confidences, object_ids)
        output_path = os.path.join(output_folder, f'frame_{frame_idx:04d}.jpg')
        cv2.imwrite(output_path, image_with_boxes)

        success, frame = cap.read()
        frame_idx += 1
        if frame_idx == EXIT_FRAMES_CRITERIA:
            break
    cap.release()


def export_tracking_to_csv(tracking_results, csv_filename, frame_height, frame_width):
    """
    Export tracking results to a CSV file with object positions.

    - Inputs:
        tracking_results: List of tuples with frame index and tracked objects.
        csv_filename: Output CSV file path.
        frame_height: Height of the video frames.
        frame_width: Width of the video frames.
    """
    with open(csv_filename, mode='w', newline='') as csv_file:
        fieldnames = ['', 't', 'hexbug', 'x', 'y']
        writer = csv.writer(csv_file)
        writer.writerow(fieldnames)

        idx = 0
        for t, frame_objects in tracking_results:
            for hexbug_id, head in frame_objects:
                # Convert the head position to the appropriate coordinates
                head_center = [(head.xyxy[0][0] + head.xyxy[0][2]) / 2, (head.xyxy[0][1] + head.xyxy[0][3]) / 2]

                # Convert tensors to floats
                head_center_x = head_center[0].item() if isinstance(head_center[0], torch.Tensor) else head_center[0]
                head_center_y = head_center[1].item() if isinstance(head_center[1], torch.Tensor) else head_center[1]


                writer.writerow([idx, t, hexbug_id, head_center_x, head_center_y])
                idx += 1

    print(f"CSV file '{csv_filename}' has been created successfully!")


def get_video_dimensions(video_path):
    """
    Get the dimensions (width and height) of the video.

    - Inputs:
        video_path: Path to the video file.

    - Returns:
        Width and height of the video.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {video_path}")

    # Get video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Release the video capture object
    cap.release()

    return width, height


def main_pipeline(video_path, detection_mode='head', tracking_method='euclidean'):
    """
    Main pipeline for detection and tracking in a video.

    Parameters:
    - video_path: Path to the input video.
    - detection_mode: Detection mode to use ('head' or 'body-head').
        - 'head': Use only head detection.
        - 'body-head': Use both body and head detection.
    - tracking_method: Tracking method to use ('euclidean' or 'lucas-kanade').
        - 'euclidean': Minimal Euclidean distance tracking.
        - 'lucas-kanade': Lucas-Kanade optical flow tracking.
    """
    # Analyze detections to determine optimal confidence level and initial object count
    if detection_mode == 'head':
        print("Using head-only detection...")
        optimal_conf, init_heads = analyze_detections(yolo_model_head, video_path, class_id=0)
        filtered_results = filter_heads(yolo_model_head, video_path, optimal_conf)
    elif detection_mode == 'body-head':
        print("Using body-head detection...")
        optimal_conf, init_heads = analyze_detections(yolo_model_body, video_path, class_id=1)
        filtered_results = filter_heads_within_bodies(yolo_model_body, yolo_model_head, video_path, optimal_conf)
    else:
        raise ValueError(f"Invalid detection mode: {detection_mode}. Use 'head' or 'body-head'.")

    # Get video dimensions
    width, height = get_video_dimensions(video_path)
    print(f"Video dimensions: {width}x{height}")

    # Perform tracking based on the chosen method
    if tracking_method == 'euclidean':
        print("Using Euclidean distance tracking...")
        tracking_results = tracking_via_minimal_euclidean_distance(filtered_results, init_heads=init_heads)
    elif tracking_method == 'lucas-kanade':
        print("Using Lucas-Kanade optical flow tracking...")
        tracking_results = tracking_via_lucas_kanade(filtered_results, init_heads=init_heads)
    else:
        raise ValueError(f"Invalid tracking method: {tracking_method}. Use 'euclidean' or 'lucas-kanade'.")

    # Export tracking results to CSV
    export_tracking_to_csv(tracking_results, csv_tracking_path, height, width)

    # Save annotated frames
    save_frames(tracking_results, frames_output_folder, video_path)

    print("Pipeline execution completed.")


if __name__ == "__main__":
    # Example usage: Set the desired detection mode and tracking method
    detection_mode = 'head'  # Options: 'head' or 'body-head'
    tracking_method = 'euclidean'  # Options: 'euclidean' or 'lucas-kanade'
    main_pipeline(video_input_path, detection_mode=detection_mode, tracking_method=tracking_method)
