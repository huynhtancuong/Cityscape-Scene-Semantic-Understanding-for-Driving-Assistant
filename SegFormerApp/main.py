from src.segmentation import *
from src.llm_prompt import *
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.video_processor import annotate_frame, process_video_file

def test_segmentation():
    # Example usage with labels:
    # url = "https://www.shutterstock.com/shutterstock/videos/1106252821/thumb/1.jpg?ip=x480"
    url = "images/city2.png"

    # Use the new function that displays segmentation with legend and class information
    result_img, detected_classes = display_segmentation_with_legend(url, alpha=0.6)

    # You can also see just the detected classes info
    print("\nSummary of detected classes:")
    for class_id, info in detected_classes.items():
        print(f"• {info['label']}: {info['percentage']}%")

def test_scene_description():
    # Test vision-to-text pipeline for movement decision making
    # url = "https://www.shutterstock.com/shutterstock/videos/1106252821/thumb/1.jpg?ip=x480"
    url = "images/city2.png"

    # Use the enhanced function that includes scene understanding
    result_img, detected_classes, spatial_analysis, scene_description = segment_image_with_scene_understanding(url, alpha=0.6)

    # Display the segmentation result
    plt.figure(figsize=(12, 6))
    plt.imshow(result_img)
    plt.title('Segmentation Result for Movement Analysis')
    plt.axis('off')
    plt.show()

    # Print the scene description that can be sent to LLM
    print("=" * 80)
    print("SCENE DESCRIPTION FOR LLM (Movement Decision Making)")
    print("=" * 80)
    print(scene_description)
    print("=" * 80)

def test_llm_prompt():
    
    url = "images/city2.png"

    # Use the enhanced function that includes scene understanding
    result_img, detected_classes, spatial_analysis, scene_description = segment_image_with_scene_understanding(url, alpha=0.6)

    # Example usage with the scene we just analyzed
    llm_prompt = create_llm_prompt_for_movement(scene_description, "safe forward navigation")

    print("=" * 80)
    print("COMPLETE LLM PROMPT FOR MOVEMENT DECISION")
    print("=" * 80)
    print(llm_prompt)
    print("=" * 80)



def test_vision_to_movement_pipeline():
    # Example: Complete pipeline test
    print("Testing complete Vision-to-Movement pipeline...")
    print("=" * 60)

    # Test with different movement goals
    test_url = "images/city2.png"

    # Test 1: Safe navigation
    result = vision_to_movement_pipeline(test_url, "safe forward navigation", show_visualization=False)

    # Print raw LLM response
    print("\nRaw LLM Response:")
    print("=" * 80)
    print(result['llm_response'])
    print("=" * 80)

    # Print parsed LLM response
    print("\nParsed LLM Response:")
    print("=" * 80)
    print(result['parsed_response'])
    print("=" * 80)

    # Annotate the frame with the parsed response
    annotated_frame = annotate_frame(result['segmented_image'], result['parsed_response'])

    # Display time taken for each step
    print(f"Segmentation time: {result['segmentation_time']:.3f}s")
    print(f"LLM response time: {result['llm_response_time']:.3f}s")

    # Display the annotated frame
    cv2.imshow("Annotated Movement Decision", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # print("Movement Decision Output (Safe Navigation):")
    # print("=" * 80)
    # print(result['parsed_response'])
    # print("=" * 80)


if __name__ == "__main__":
    # test_scene_description()
    # test_llm_prompt()
    # test_vision_to_movement_pipeline()
    # get_movement_from_llm("")
    # test_llm()
    process_video_file("videos/00067cfb-e535423e.mov", "videos/00067cfb-e535423e_out_1.avi", "Safe navigation",
                       max_frames=None, skip_frames=30, run_segmentation_on_skip=True)
