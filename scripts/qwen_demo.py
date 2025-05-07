import os
import json
import torch
import warnings
import numpy as np
from PIL import Image

from serve import pointso as orientation
from serve.scene_graph import open6dor_scene_graph
from segmentation import sam, florence as detection
from serve.utils import generate_rotation_matrix, get_point_cloud_from_rgbd

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from serve.qwen_inference import open6dor_parsing, open6dor_spatial_reasoning

warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)


if __name__ == "__main__":
    image_path = "assets/open6dor.png"
    depth_path = "assets/open6dor.npy"
    prompt = "Place the knife behind the clipboard on the table. And rotate the handle of the knife to left."
    output_folder = "output"

    image = Image.open(image_path).convert("RGB")
    depth = np.load(depth_path)
    vinvs = np.array([[0., 1., 0., 0.],
                      [-0.9028605, -0., 0.42993355, -0.],
                      [0.42993355, -0., 0.9028605, -0.],
                      [1., 0., 1.2, 1.]])
    projs = [[1.7320507, 0., 0., 0.],
             [0., 2.5980759, 0., 0.],
             [0., 0., 0., -1.],
             [0., 0., 0.05, 0.]]
    pcd = get_point_cloud_from_rgbd(depth, np.array(image), vinvs, projs).cpu().numpy().astype(np.float64)
    np.save(os.path.join(output_folder, "scene.npy"), pcd)
    pcd = pcd.reshape(depth.shape[0], depth.shape[1], 6)[:, :, :3]

    print("Load models...")
    detection_model = detection.get_model()
    sam_model = sam.get_model()
    orientation_model = orientation.get_model()
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    ).to('cuda')
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    print("Start object parsing...")
    info = open6dor_parsing(qwen_model, processor, image, prompt)
    print(json.dumps(info, indent=2))
    object_list = [info['picked_object']] + info['related_objects']

    print("Start Segment Anything...")
    detections = detection.get_detections(image, object_list, detection_model, output_folder=output_folder, single=True)
    mask, ann_img, object_names = sam.get_mask(image, object_list, sam_model, detections, output_folder=output_folder)
    if set(object_list) != set(object_names):
        raise ValueError("Grounded SAM Error: object list does not match the detected object names.")

    print("Generate scene graph...")
    picked_object_info, other_objects_info, picked_object_dict \
        = open6dor_scene_graph(image, pcd, mask, info, object_names, orientation_model, output_folder=output_folder)
    print("picked info:", picked_object_info)
    print("other info:")
    for node in other_objects_info:
        print(node)

    print("Start spatial reasoning...")
    response = open6dor_spatial_reasoning(qwen_model, processor, image, prompt, picked_object_info, other_objects_info)
    print(response)

    init_position = picked_object_dict["center"]
    target_position = response["target_position"]
    init_orientation = picked_object_dict["orientation"]
    target_orientation = info["target_orientation"]

    if len(target_orientation) > 0 and target_orientation.keys() == init_orientation.keys():
        direction_attributes = target_orientation.keys()
        init_directions = [init_orientation[direction] for direction in direction_attributes]
        target_directions = [target_orientation[direction] for direction in direction_attributes]
        transform_matrix = generate_rotation_matrix(np.array(init_directions), np.array(target_directions)).tolist()
    else:
        transform_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    print("Result:")
    result = {
        'init_position': init_position,
        'target_position': target_position,
        'delta_position': [round(target_position[i] - init_position[i], 2) for i in range(3)],
        'init_orientation': init_orientation,
        'target_orientation': target_orientation,
        'transform_matrix': transform_matrix
    }
    print(result)
    open(os.path.join(output_folder, "result.json"), 'w').write(json.dumps(result, indent=4))
