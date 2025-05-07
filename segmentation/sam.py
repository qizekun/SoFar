import os.path
import numpy as np
from PIL import Image
import supervision as sv

import torch
from segment_anything import sam_model_registry, SamPredictor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "checkpoints/sam_vit_h_4b8939.pth"


def get_model():
    # Building SAM Model and SAM Predictor
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    return sam


# Prompting SAM with detected boxes
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks, dtype=bool)


def get_mask(image, object_list, sam, detections, output_folder="output"):
    sam_predictor = SamPredictor(sam)
    image = np.array(image)

    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=image,
        xyxy=detections.xyxy
    )

    # annotate image with detections
    mask_annotator = sv.MaskAnnotator(
        opacity=0.15
    )
    box_annotator = sv.BoxAnnotator(
        color=sv.Color.BLACK,
        thickness=1
    )
    label_annotator = sv.LabelAnnotator(
        text_position=sv.Position.CENTER,
        text_color=sv.Color.WHITE,
        color=sv.Color.BLACK,
        text_padding=2,
        text_thickness=1,
        text_scale=0.5
    )

    index_labels = [str(i + 1) for i in range(len(detections))]
    object_labels = [
        f"{object_list[class_id]}"
        for _, _, confidence, class_id, _, _
        in detections]
    confidence_labels = [
        f"{confidence:0.2f}"
        for _, _, confidence, class_id, _, _
        in detections]

    label_image = label_annotator.annotate(scene=image.copy(), detections=detections, labels=index_labels)
    label_image = Image.fromarray(label_image)

    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=index_labels)

    # save the annotated grounded-sam image
    annotated_image = Image.fromarray(annotated_image)
    annotated_image.save(os.path.join(output_folder, "sam_annotated_image.jpg"))

    return detections.mask, label_image, object_labels
