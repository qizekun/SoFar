import os
import numpy as np
from typing import Union, Any, Tuple, Dict
from unittest.mock import patch

import torch
from PIL import Image
import supervision as sv
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import get_imports


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# florance-2 checkpoint
FLORENCE_CHECKPOINT = "microsoft/Florence-2-base"
FLORENCE_OPEN_VOCABULARY_DETECTION_TASK = '<OPEN_VOCABULARY_DETECTION>'


def fixed_get_imports(filename: Union[str, os.PathLike]) -> list[str]:
    """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
    if not str(filename).endswith("/modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    # imports.remove("flash_attn")
    return imports


def get_model():
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained(
            FLORENCE_CHECKPOINT, trust_remote_code=True).to(DEVICE).eval()
        processor = AutoProcessor.from_pretrained(
            FLORENCE_CHECKPOINT, trust_remote_code=True)
        return model, processor


def run_florence_inference(
    model: Any,
    processor: Any,
    device: torch.device,
    image: Image,
    task: str,
    text: str = ""
) -> Tuple[str, Dict]:
    prompt = task + text
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=False)[0]
    response = processor.post_process_generation(
        generated_text, task=task, image_size=image.size)
    return generated_text, response


def get_detections(image, obj_list, florence_model, output_folder="output", single=False):

    model, processor = florence_model

    detections_list = []
    for i in range(len(obj_list)):
        obj = obj_list[i]
        _, result = run_florence_inference(
            model=model,
            processor=processor,
            device=DEVICE,
            image=image,
            task=FLORENCE_OPEN_VOCABULARY_DETECTION_TASK,
            text=obj
        )
        detections = sv.Detections.from_lmm(
            lmm=sv.LMM.FLORENCE_2,
            result=result,
            resolution_wh=image.size
        )
        detections.class_id = np.full(len(detections.xyxy), i)
        detections.confidence = np.full(len(detections.xyxy), 1.0)
        detections_list.append(detections)

    detections = sv.Detections.merge(detections_list)

    image = np.array(image)
    # annotate image with detections
    box_annotator = sv.BoxAnnotator(
        thickness=1
    )
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)

    # save the annotated florence image
    annotated_frame = Image.fromarray(annotated_frame)
    annotated_frame.save(os.path.join(output_folder, "florence_image.jpg"))

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
