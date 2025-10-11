from src.segmentation import *
from src.llm_prompt import *
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

def vision_to_movement_pipeline(input_image, movement_goal="safe navigation", show_visualization=True, run_llm=True):
    """
    Complete pipeline from image to movement decision text for LLM.
    
    Args:
        input_image: Image input (path, URL, or PIL Image)
        movement_goal: Navigation objective
        show_visualization: Whether to display the segmentation result
    
    Returns:
        dict: Complete analysis including LLM prompt and all intermediate results
    """
    
    import time
    # Step 1: Perform segmentation with scene understanding (timed)
    t0 = time.time()
    result_img, detected_classes, spatial_analysis, scene_description = \
        segment_image_with_scene_understanding(input_image, alpha=0.6)
    segmentation_time = time.time() - t0

    # Step 4: Optional visualization
    if show_visualization:
        plt.figure(figsize=(12, 6))
        plt.imshow(result_img)
        plt.title(f'Scene Analysis for: {movement_goal}')
        plt.axis('off')
        plt.show()

    if run_llm is False:
        return {
            'segmented_image': result_img,
            'detected_classes': detected_classes,
            'spatial_analysis': spatial_analysis,
            'scene_description': scene_description,
            'segmentation_time': segmentation_time
        }
    
    # Step 2: Create LLM prompt
    llm_prompt = create_llm_prompt_for_movement(scene_description, movement_goal)

    # Step 3: Get LLM response (timed)
    t1 = time.time()
    llm_response = get_response_from_llm(llm_prompt)
    llm_time = time.time() - t1

    parsed_response = parse_llm_response(llm_response)

    # Step 5: Return comprehensive results
    return {
        'segmented_image': result_img,
        'detected_classes': detected_classes,
        'spatial_analysis': spatial_analysis,
        'scene_description': scene_description,
        'llm_prompt': llm_prompt,
        'movement_goal': movement_goal,
        'llm_response': llm_response,
        'parsed_response': parsed_response,
        'segmentation_time': segmentation_time,
        'llm_response_time': llm_time
    }
