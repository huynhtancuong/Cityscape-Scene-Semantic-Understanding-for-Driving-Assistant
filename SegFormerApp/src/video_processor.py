# Video Processing for Autonomous Vehicle Decision Making
import cv2
import os
import time
from collections import deque
from typing import List, Tuple, Optional
import numpy as np
from IPython.display import Video, HTML, display
import tempfile
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import io
import base64
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import ndimage
from src.vision_system import vision_to_movement_pipeline

def annotate_frame(frame: np.ndarray, parsed_response: dict) -> np.ndarray:
    """
    Annotate a frame with parsed LLM movement decision results.
    """
    # Support both PIL Image and numpy array
    if hasattr(frame, 'copy') and hasattr(frame, 'shape'):
        annotated = frame.copy()
    else:
        # Assume PIL Image
        import numpy as np
        annotated = np.array(frame.convert('RGB'))
    height, width = annotated.shape[:2]

    # Convert to BGR for OpenCV
    if annotated.shape[2] == 3:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

    # Create info panel with fixed height
    panel_height = 200
    panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)  # Dark gray background

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    color = (255, 255, 255)
    thickness = 1

    # Safety color (green/yellow/red)
    def get_safety_color(score):
        if score >= 8:
            return (0, 200, 0)
        elif score >= 5:
            return (0, 200, 200)
        else:
            return (0, 0, 255)

    # Safety and action info
    safety_score = parsed_response.get('safety_score', 5)
    primary_action = parsed_response.get('primary_action', '').upper()
    speed = parsed_response.get('speed', '').upper()
    hazards = parsed_response.get('hazards', [])
    reasoning = parsed_response.get('reasoning', '')
    # monitor_priorities = parsed_response.get('monitor_priorities', [])

    y = 30
    # Display all three on the same line, spaced apart
    cv2.putText(panel, f"Safety: {safety_score}/10", (10, y), font, font_scale, get_safety_color(safety_score), thickness)
    cv2.putText(panel, f"Action: {primary_action}", (220, y), font, font_scale, color, thickness)
    cv2.putText(panel, f"Speed: {speed}", (400, y), font, font_scale, color, thickness)
    y += 30
    if hazards:
        hazard_str = ", ".join(hazards)
        # Use orange for hazards label, yellow for hazard text
        cv2.putText(panel, "Hazards:", (10, y), font, font_scale, (0, 140, 255), 1)  # orange
        y += 20
        cv2.putText(panel, f"{hazard_str}", (10, y), font, font_scale, (0, 255, 255), 1)  # yellow
        y += 20
    # if monitor_priorities:
    #     for mp in monitor_priorities:
    #         cv2.putText(panel, mp, (10, y), font, font_scale, (180, 255, 180), 1)
    #         y += 18
    # Reasoning (wrap if long)
    if reasoning:
        max_len = 80
        lines = [reasoning[i:i+max_len] for i in range(0, len(reasoning), max_len)]
        # Use blue for reasoning label, light blue for text
        cv2.putText(panel, "Reasoning:", (10, y), font, font_scale, (255, 128, 0), 1)  # blue-ish
        y += 18
        for line in lines:
            cv2.putText(panel, line, (10, y), font, font_scale, (255, 220, 180), 1)  # light blue
            y += 16

    # Combine frame with panel
    result_frame = np.vstack([annotated, panel])
    expected_height = height + panel_height
    if result_frame.shape[0] != expected_height or result_frame.shape[1] != width:
        result_frame = cv2.resize(result_frame, (width, expected_height))
    return result_frame


def process_video_file(video_path: str, output_path: str = None, 
                      movement_goal: str = "safe navigation",
                      max_frames: Optional[int] = None,
                      skip_frames: int = 1, run_segmentation_on_skip: bool = False) -> dict:
    """
    Process a video file frame by frame.
    
    Args:
        video_path: Path to input video
        output_path: Path for output video (if None, generates temp file)
        movement_goal: Navigation objective
        max_frames: Maximum frames to process (None for all)
        skip_frames: Process every nth frame (1 = all frames)
        
    Returns:
        Dict with processing results and statistics
    """
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video Info: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Setup output video with corrected dimensions (including annotation panel)
    if output_path is None:
        output_path = tempfile.mktemp(suffix='.avi')
    
    # Calculate output dimensions (original height + annotation panel)
    panel_height = 200
    output_height = height + panel_height
    
    # Use more compatible codec and settings
    try:
        # Try H.264 codec first (more compatible)
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, output_height))
        
        # Test if the writer opened successfully
        if not out.isOpened():
            print("H.264 codec failed, trying XVID...")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output_path = output_path.replace('.mp4', '.avi')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, output_height))
            
            if not out.isOpened():
                print("XVID failed, trying mp4v...")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                output_path = output_path.replace('.avi', '.mp4')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, output_height))
                
                if not out.isOpened():
                    raise ValueError("Cannot create video writer with any codec")
    except Exception as e:
        print(f"Error setting up video writer: {e}")
        print("Falling back to default codec...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, output_height))
    
    print(f"📝 Output video will be: {width}x{output_height} (original + {panel_height}px panel)")
    
    # Processing variables
    processed_frames = 0
    decisions_log = []
    frame_count = 0
    last_annotated_frame = None
    processing_times = []
    
    from PIL import Image
    import time
    last_parsed_response = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Skip frames if specified
        if (frame_count - 1) % skip_frames != 0:
            if run_segmentation_on_skip and last_parsed_response is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb_frame)
                seg_result = vision_to_movement_pipeline(pil_frame, movement_goal, show_visualization=False, run_llm=False)
                segmented_frame = seg_result.get('segmented_image', frame)
                annotated_frame = annotate_frame(segmented_frame, last_parsed_response)
                out.write(annotated_frame)
                last_annotated_frame = annotated_frame
            elif last_annotated_frame is not None:
                out.write(last_annotated_frame)
            else:
                out.write(frame)
            continue

        if max_frames and processed_frames >= max_frames:
            if last_annotated_frame is not None:
                out.write(last_annotated_frame)
            else:
                out.write(frame)
            continue

        start_time = time.time()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(rgb_frame)

        # Use vision_to_movement_pipeline for decision
        result = vision_to_movement_pipeline(pil_frame, movement_goal, show_visualization=False)
        # print(f"Result keys: {list(result.keys())}")
        processing_time = time.time() - start_time
        processing_times.append(processing_time)

        parsed_response = result.get('parsed_response', {})
        segmented_frame = result.get('segmented_image', frame)
        annotated_frame = annotate_frame(segmented_frame, parsed_response)
        last_annotated_frame = annotated_frame
        last_parsed_response = parsed_response

        # Print log
        print(f"Frame {frame_count}/{total_frames} | Decision: {parsed_response.get('primary_action', 'N/A')} {parsed_response.get('speed', 'N/A')} | Segmentation Time: {result.get('segmentation_time', 0):.3f}s | LLM Time: {result.get('llm_response_time', 0):.3f}s")

        decisions_log.append({
            'frame': frame_count,
            'timestamp': frame_count / fps,
            'decision': parsed_response,
            'processing_time': processing_time
        })

        if annotated_frame.shape[:2] == (output_height, width):
            out.write(annotated_frame)
        else:
            print(f"Warning: Frame size mismatch. Expected {(output_height, width)}, got {annotated_frame.shape[:2]}")
            annotated_frame_resized = cv2.resize(annotated_frame, (width, output_height))
            out.write(annotated_frame_resized)

        processed_frames += 1

        if processed_frames % 30 == 0:
            progress = (frame_count / total_frames) * 100
            avg_time = np.mean(processing_times[-30:])
            print(f"Progress: {progress:.1f}% | Avg processing: {avg_time:.3f}s/frame")

    cap.release()
    out.release()
    
    # Calculate statistics
    avg_processing_time = np.mean(processing_times) if processing_times else 0
    total_time = sum(processing_times)
    
    results = {
        'output_video_path': output_path,
        'total_frames_processed': processed_frames,
        'avg_processing_time': avg_processing_time,
        'total_processing_time': total_time,
        'decisions_log': decisions_log,
        'video_info': {
            'fps': fps,
            'width': width,
            'height': height,
            'output_height': output_height,
            'duration': total_frames / fps
        }
    }
    
    print(f"✅ Video processing completed!")
    print(f"📊 Processed {processed_frames} frames in {total_time:.2f}s")
    print(f"⏱️  Average: {avg_processing_time:.3f}s per frame")
    
    return results
