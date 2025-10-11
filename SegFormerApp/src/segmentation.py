import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import ndimage

# Initialize model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"
processor = SegformerImageProcessor(do_resize=False)
model = SegformerForSemanticSegmentation.from_pretrained(model_name)
model.to(device)

def cityscapes_palette():
    """Cityscapes palette that maps each class to RGB values."""
    return [
        [128, 64, 128],   # road
        [244, 35, 232],   # sidewalk
        [70, 70, 70],     # building
        [102, 102, 156],  # wall
        [190, 153, 153],  # fence
        [153, 153, 153],  # pole
        [250, 170, 30],   # traffic light
        [220, 220, 0],    # traffic sign
        [107, 142, 35],   # vegetation
        [152, 251, 152],  # terrain
        [70, 130, 180],   # sky
        [220, 20, 60],    # person
        [255, 0, 0],      # rider
        [0, 0, 142],      # car
        [0, 0, 70],       # truck
        [0, 60, 100],     # bus
        [0, 80, 100],     # train
        [0, 0, 230],      # motorcycle
        [119, 11, 32],    # bicycle
        [0, 0, 0]         # ignore/unlabeled
    ]

def cityscapes_labels():
    """Cityscapes class labels."""
    return [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle',
        'ignore'
    ]

def get_object_position(mask, class_id):
    """Get spatial information about an object in the segmentation mask."""
    object_mask = (mask == class_id)
    if not np.any(object_mask):
        return None
    
    # Find bounding box
    rows, cols = np.where(object_mask)
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()
    
    # Calculate center of mass
    center_y, center_x = ndimage.center_of_mass(object_mask)
    
    # Get relative positions
    height, width = mask.shape
    center_x_rel = center_x / width
    center_y_rel = center_y / height
    
    # Determine position descriptors
    horizontal_pos = "left" if center_x_rel < 0.33 else "center" if center_x_rel < 0.67 else "right"
    vertical_pos = "top" if center_y_rel < 0.33 else "middle" if center_y_rel < 0.67 else "bottom"
    
    # Calculate object size relative to image
    object_area = np.sum(object_mask)
    relative_size = object_area / (height * width)
    
    size_descriptor = "large" if relative_size > 0.1 else "medium" if relative_size > 0.01 else "small"
    
    return {
        'center': (center_x, center_y),
        'center_relative': (center_x_rel, center_y_rel),
        'bounding_box': (min_col, min_row, max_col, max_row),
        'area': object_area,
        'relative_size': relative_size,
        'horizontal_position': horizontal_pos,
        'vertical_position': vertical_pos,
        'size_descriptor': size_descriptor
    }

def analyze_spatial_relationships(seg_map, detected_classes):
    """Analyze spatial relationships between objects in the scene."""
    analysis = {}
    
    for class_id, info in detected_classes.items():
        position_info = get_object_position(seg_map, class_id)
        if position_info:
            analysis[class_id] = {
                **info,
                'spatial_info': position_info
            }
    
    return analysis

def generate_scene_description(spatial_analysis, prioritize_movement=True):
    """Convert spatial analysis to natural language description for LLM."""
    
    # Define movement-relevant categories
    movement_critical = ['road', 'sidewalk', 'person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle']
    obstacles = ['building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign']
    navigation_aids = ['road', 'sidewalk']
    dynamic_objects = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
    
    # Sort objects by importance for movement decisions
    sorted_objects = []
    
    if prioritize_movement:
        # First: Navigation surfaces
        nav_objects = [obj for obj in spatial_analysis.values() 
                      if obj['label'] in navigation_aids]
        nav_objects.sort(key=lambda x: x['spatial_info']['relative_size'], reverse=True)
        sorted_objects.extend(nav_objects)
        
        # Second: Dynamic objects (potential moving obstacles)
        dynamic_objs = [obj for obj in spatial_analysis.values() 
                       if obj['label'] in dynamic_objects and obj['label'] not in navigation_aids]
        dynamic_objs.sort(key=lambda x: x['spatial_info']['relative_size'], reverse=True)
        sorted_objects.extend(dynamic_objs)
        
        # Third: Static obstacles
        static_obstacles = [obj for obj in spatial_analysis.values() 
                           if obj['label'] in obstacles]
        static_obstacles.sort(key=lambda x: x['spatial_info']['relative_size'], reverse=True)
        sorted_objects.extend(static_obstacles)
        
        # Fourth: Environment elements
        env_objects = [obj for obj in spatial_analysis.values() 
                      if obj['label'] not in movement_critical and obj['label'] not in obstacles]
        env_objects.sort(key=lambda x: x['spatial_info']['relative_size'], reverse=True)
        sorted_objects.extend(env_objects)
    else:
        # Sort by size if not prioritizing movement
        sorted_objects = sorted(spatial_analysis.values(), 
                               key=lambda x: x['spatial_info']['relative_size'], reverse=True)
    
    # Generate description
    description_parts = []
    
    # Scene overview
    total_objects = len(spatial_analysis)
    description_parts.append(f"Scene Analysis ({total_objects} object types detected):")
    
    # Navigation surfaces
    nav_surfaces = [obj for obj in sorted_objects if obj['label'] in navigation_aids]
    if nav_surfaces:
        description_parts.append("\nNAVIGATION SURFACES:")
        for obj in nav_surfaces:
            pos_desc = f"{obj['spatial_info']['vertical_position']}-{obj['spatial_info']['horizontal_position']}"
            size_desc = obj['spatial_info']['size_descriptor']
            description_parts.append(
                f"- {obj['label'].title()}: {size_desc} area, located at {pos_desc} of scene ({obj['percentage']:.1f}% coverage)"
            )
    
    # Dynamic objects (potential threats/obstacles)
    dynamic_objs = [obj for obj in sorted_objects if obj['label'] in dynamic_objects and obj['label'] not in navigation_aids]
    if dynamic_objs:
        description_parts.append("\nDYNAMIC OBJECTS (Potential Moving Obstacles):")
        for obj in dynamic_objs:
            pos_desc = f"{obj['spatial_info']['vertical_position']}-{obj['spatial_info']['horizontal_position']}"
            size_desc = obj['spatial_info']['size_descriptor']
            description_parts.append(
                f"- {obj['label'].title()}: {size_desc} size, positioned at {pos_desc} ({obj['percentage']:.1f}% of view)"
            )
    
    # Static obstacles
    static_obs = [obj for obj in sorted_objects if obj['label'] in obstacles]
    if static_obs:
        description_parts.append("\nSTATIC OBSTACLES:")
        for obj in static_obs:
            pos_desc = f"{obj['spatial_info']['vertical_position']}-{obj['spatial_info']['horizontal_position']}"
            size_desc = obj['spatial_info']['size_descriptor']
            description_parts.append(
                f"- {obj['label'].title()}: {size_desc}, {pos_desc} area ({obj['percentage']:.1f}% coverage)"
            )
    
    # Environmental context
    env_objs = [obj for obj in sorted_objects 
               if obj['label'] not in movement_critical and obj['label'] not in obstacles]
    if env_objs:
        description_parts.append("\nENVIRONMENT CONTEXT:")
        for obj in env_objs:
            pos_desc = f"{obj['spatial_info']['vertical_position']}-{obj['spatial_info']['horizontal_position']}"
            description_parts.append(
                f"- {obj['label'].title()}: {pos_desc} ({obj['percentage']:.1f}% coverage)"
            )
    
    # Movement recommendations
    description_parts.append("\nMOVEMENT CONSIDERATIONS:")
    
    # Check for clear paths
    road_present = any(obj['label'] == 'road' for obj in spatial_analysis.values())
    sidewalk_present = any(obj['label'] == 'sidewalk' for obj in spatial_analysis.values())
    
    if road_present:
        road_info = next(obj for obj in spatial_analysis.values() if obj['label'] == 'road')
        description_parts.append(f"- Road available: {road_info['spatial_info']['size_descriptor']} coverage, {road_info['spatial_info']['horizontal_position']} side")
    
    if sidewalk_present:
        sidewalk_info = next(obj for obj in spatial_analysis.values() if obj['label'] == 'sidewalk')
        description_parts.append(f"- Sidewalk available: {sidewalk_info['spatial_info']['size_descriptor']} coverage, {sidewalk_info['spatial_info']['horizontal_position']} side")
    
    # Warn about dynamic objects
    if dynamic_objs:
        description_parts.append(f"- CAUTION: {len(dynamic_objs)} types of moving objects detected")
        for obj in dynamic_objs[:3]:  # Top 3 most significant
            description_parts.append(f"  * {obj['label'].title()} at {obj['spatial_info']['horizontal_position']}")
    
    return "\n".join(description_parts)

def segment_image(input_image, alpha: float = 0.5, include_scene_understanding: bool = False, filter_bottom_cars: bool = True, bottom_threshold: float = 0.9):
    """
    Unified function for semantic segmentation with optional scene understanding.

    Args:
        input_image (str or PIL.Image): Path to an image file, URL, or a PIL Image instance.
        alpha (float): Blending factor for overlay. 0 = only original, 1 = only mask.
        include_scene_understanding (bool): Whether to include spatial analysis and scene description.
        filter_bottom_cars (bool): Whether to filter out car masks at the bottom of the image.
        bottom_threshold (float): Threshold for bottom region (0.6 = bottom 40% of image).

    Returns:
        tuple: If include_scene_understanding=False: (overlay_image, detected_classes)
               If include_scene_understanding=True: (overlay_image, detected_classes, spatial_analysis, scene_description)
    """
    # Load image
    if isinstance(input_image, str):
        if input_image.startswith(('http://', 'https://')):
            resp = requests.get(input_image)
            img = Image.open(BytesIO(resp.content)).convert("RGB")
        else:
            img = Image.open(input_image).convert("RGB")
    elif isinstance(input_image, Image.Image):
        img = input_image.convert("RGB")
    else:
        raise ValueError("Unsupported input type. Provide a file path, URL, or PIL.Image.")

    # Preprocess and forward pass
    pixel_values = processor(img, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        outputs = model(pixel_values)

    # Post-process to get mask
    seg_map = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[img.size[::-1]]
    )[0].cpu().numpy()

    # Filter out car and truck masks located at the bottom of the image (if enabled)
    labels = cityscapes_labels()
    if filter_bottom_cars:
        car_class_id = 13  # 'car' class ID in Cityscapes
        truck_class_id = 14  # 'truck' class ID in Cityscapes
        vehicle_class_ids = [car_class_id, truck_class_id]
        
        unique_classes = np.unique(seg_map)
        height, width = seg_map.shape
        bottom_region_start = int(height * bottom_threshold)
        
        # Create a mask for the bottom region
        bottom_region_mask = np.zeros((height, width), dtype=bool)
        bottom_region_mask[bottom_region_start:, :] = True
        
        total_filtered_pixels = 0
        
        for vehicle_id in vehicle_class_ids:
            if vehicle_id in unique_classes:
                # Create a mask for vehicle pixels
                vehicle_mask = (seg_map == vehicle_id)
                
                # Find vehicle pixels that are in the bottom region
                bottom_vehicle_mask = vehicle_mask & bottom_region_mask
                
                # Filter out (set to background/road) vehicle pixels in bottom region
                if np.any(bottom_vehicle_mask):
                    # Set bottom vehicle pixels to road class (class 0) instead of vehicle
                    seg_map[bottom_vehicle_mask] = 0  # road class
                    filtered_count = np.sum(bottom_vehicle_mask)
                    total_filtered_pixels += filtered_count
                    vehicle_label = labels[vehicle_id]
                    # print(f"Filtered out {filtered_count} {vehicle_label} pixels from bottom region of image")
        
        if total_filtered_pixels > 0:
            # print(f"Total filtered pixels: {total_filtered_pixels}")
            pass

    # Use Cityscapes palette and labels
    palette = np.array(cityscapes_palette(), dtype=np.uint8)
    
    # Create color mask
    color_mask = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    for label_id, color in enumerate(palette):
        color_mask[seg_map == label_id] = color

    # Get unique classes in the image
    unique_classes = np.unique(seg_map)
    detected_classes = {}
    for class_id in unique_classes:
        if class_id < len(labels):
            pixel_count = np.sum(seg_map == class_id)
            percentage = (pixel_count / seg_map.size) * 100
            detected_classes[class_id] = {
                'label': labels[class_id],
                'color': tuple(palette[class_id]),
                'pixel_count': int(pixel_count),
                'percentage': round(percentage, 2)
            }

    # Blend original and mask
    orig_arr = np.array(img)
    overlay = (orig_arr * (1 - alpha) + color_mask * alpha).astype(np.uint8)
    result_image = Image.fromarray(overlay)
    
    # Return basic result or enhanced result with scene understanding
    if include_scene_understanding:
        # Analyze spatial relationships
        spatial_analysis = analyze_spatial_relationships(seg_map, detected_classes)
        
        # Generate scene description for LLM
        scene_description = generate_scene_description(spatial_analysis)
        
        return result_image, detected_classes, spatial_analysis, scene_description
    else:
        return result_image, detected_classes

# Legacy function aliases for backward compatibility
def segment_image_with_scene_understanding(input_image, alpha: float = 0.5):
    """
    Legacy function for enhanced segmentation with scene understanding.
    Now calls the unified segment_image function with include_scene_understanding=True.
    """
    return segment_image(input_image, alpha=alpha, include_scene_understanding=True)

def display_segmentation_with_legend(input_image, alpha: float = 0.5):
    """
    Display segmentation result with a legend showing detected classes.
    """
    result_img, detected_classes = segment_image(input_image, alpha)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Display segmented image
    ax1.imshow(result_img)
    ax1.set_title('Segmentation Result')
    ax1.axis('off')
    
    # Create legend
    legend_elements = []
    class_info_text = []
    
    for class_id, info in detected_classes.items():
        color_normalized = [c/255.0 for c in info['color']]
        legend_elements.append(mpatches.Patch(color=color_normalized, label=info['label']))
        class_info_text.append(f"{info['label']}: {info['percentage']}%")
    
    ax2.legend(handles=legend_elements, loc='center', fontsize=10)
    ax2.set_title('Detected Classes')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed class information
    print("Detected Classes:")
    print("-" * 50)
    for class_id, info in detected_classes.items():
        print(f"Class {class_id}: {info['label']}")
        print(f"  Color (RGB): {info['color']}")
        print(f"  Pixels: {info['pixel_count']:,}")
        print(f"  Coverage: {info['percentage']}%")
        print()
    
    return result_img, detected_classes