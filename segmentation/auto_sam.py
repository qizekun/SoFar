import cv2
import torch
import numpy as np
import supervision as sv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "checkpoints/sam_vit_h_4b8939.pth"
MIN_AREA_PERCENTAGE = 0.005
MAX_AREA_PERCENTAGE = 0.05


def get_model():
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator


def get_mask(image, mask_generator):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # segment image
    sam_result = mask_generator.generate(image)
    detections = sv.Detections.from_sam(sam_result=sam_result)

    # filter masks
    height, width, channels = image.shape
    image_area = height * width

    min_area_mask = (detections.area / image_area) > MIN_AREA_PERCENTAGE
    max_area_mask = (detections.area / image_area) < MAX_AREA_PERCENTAGE
    detections = detections[min_area_mask & max_area_mask]

    # annotate image with detections
    mask_annotator = sv.MaskAnnotator(
        color_lookup=sv.ColorLookup.INDEX,
        opacity=0.3
    )
    box_annotator = sv.BoundingBoxAnnotator(
        color_lookup=sv.ColorLookup.INDEX,
        color=sv.Color.BLACK,
        thickness=1
    )
    label_annotator = sv.LabelAnnotator(
        color_lookup=sv.ColorLookup.INDEX,
        text_position=sv.Position.CENTER,
        text_color=sv.Color.WHITE,
        color=sv.Color.BLACK,
        text_padding=2,
        text_thickness=1,
        text_scale=0.5
    )

    index_labels = [str(i + 1) for i in range(len(detections))]

    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=index_labels)

    # save the annotated grounded-sam image
    cv2.imwrite("sam_annotated_image.jpg", annotated_image)

    return detections
