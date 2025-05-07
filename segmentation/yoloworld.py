import os.path
import numpy as np
from PIL import Image
import supervision as sv

import torch
import torchvision
from inference.models import YOLOWorld

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model():
    # Building YOLO-World inference model
    yolo_world_model = YOLOWorld(model_id="yolo_world/l")

    return yolo_world_model


def get_detections(image, object_list, yolo_world_model, output_folder="output",
                   confidence_threshold=0.05, nms_threshold=0.5, single=False):

    image = np.array(image)
    yolo_world_model.set_classes(object_list)

    # detect objects
    results = yolo_world_model.infer(image, confidence=confidence_threshold)
    detections = sv.Detections.from_inference(results).with_nms(
        class_agnostic=True, threshold=nms_threshold
    )

    # annotate image with detections
    box_annotator = sv.BoxAnnotator(
        thickness=1
    )
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)

    # save the annotated yolo world image
    annotated_frame = Image.fromarray(annotated_frame)
    annotated_frame.save(os.path.join(output_folder, "yoloworld_annotated_image.jpg"))

    # NMS post process
    print(f"Before NMS: {len(detections.xyxy)} boxes")
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy),
        torch.from_numpy(detections.confidence),
        nms_threshold
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    print(f"After NMS: {len(detections.xyxy)} boxes")

    if single:
        # Filter detections to only include highest confidence detections per object
        print(f"Before Selection: {len(detections.xyxy)} boxes")

        # Select highest confidence detection for each object in object_list
        highest_confidence_detections = {}
        for i, class_id in enumerate(detections.class_id):
            confidence = detections.confidence[i]
            if class_id not in highest_confidence_detections or confidence > highest_confidence_detections[class_id][1]:
                highest_confidence_detections[class_id] = (i, confidence)

        # Filter detections to only include highest confidence detections per object
        selected_indices = [idx for idx, _ in highest_confidence_detections.values()]
        detections.xyxy = detections.xyxy[selected_indices]
        detections.confidence = detections.confidence[selected_indices]
        detections.class_id = detections.class_id[selected_indices]

        print(f"After Selection: {len(detections.xyxy)} boxes")

    return detections
