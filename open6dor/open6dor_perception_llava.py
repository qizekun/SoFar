import os
import glob
import json
import warnings
import numpy as np
from PIL import Image
from tqdm import tqdm

from serve import sofar_llava_inference
from serve import pointso as orientation
from utils import preprocess_open6dor_image
from serve.scene_graph import open6dor_scene_graph
from segmentation import sam, florence as detection
from serve.utils import generate_rotation_matrix, get_point_cloud_from_rgbd

warnings.filterwarnings("ignore")


def process_dataset(p):
    try:
        image_path = p + "isaac_render-rgb-0-1.png"
        depth_path = p + "isaac_render-depth-0-1.npy"
        info = json.load(open(p + "task_config_new5.json"))
        prompt = f"Pick the {info['target_obj_name']}." + info["instruction"] if "rot" in p else info["instruction"]

        output_path = p + "output_llava"
        os.makedirs(output_path, exist_ok=True)

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
        pcd = pcd.reshape(depth.shape[0], depth.shape[1], 6)[:, :, :3]

        info = sofar_llava_inference.inference(
            obj_parsing_model, obj_parsing_tokenizer, obj_parsing_image_processor, image, prompt)
        info['related_objects'] = [] if "rot" in p else info['related_objects']
        object_list = [info['picked_object']] + info['related_objects']

        image = preprocess_open6dor_image(image)
        detections = detection.get_detections(image, object_list, detection_model, output_folder=output_path, single=True)
        mask, ann_img, object_names = sam.get_mask(image, object_list, sam_model, detections, output_folder=output_path)

        picked_object_info, other_objects_info, picked_object_dict \
            = open6dor_scene_graph(image, pcd, mask, info, object_names, orientation_model, output_folder=output_path)

        if "rot" not in p:
            prompt = f"Command: {prompt}picked_object_info: {picked_object_info}other_objects_info: {other_objects_info}"
            response = sofar_llava_inference.inference(vlm_model, vlm_tokenizer, vlm_image_processor, image, prompt)
            init_position = picked_object_dict["center"]
            target_position = response["target_position"]
        else:
            init_position = picked_object_dict["center"]
            target_position = picked_object_dict["center"]

        init_orientation = picked_object_dict["orientation"]
        target_orientation = info["target_orientation"]
        del target_orientation["calculation_process"]

        if len(target_orientation) > 0 and target_orientation.keys() == init_orientation.keys():
            direction_attributes = target_orientation.keys()
            init_directions = [init_orientation[direction] for direction in direction_attributes]
            target_directions = [target_orientation[direction] for direction in direction_attributes]
            transform_matrix = generate_rotation_matrix(np.array(init_directions), np.array(target_directions)).tolist()
        else:
            transform_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        result = {
            'init_position': init_position,
            'target_position': target_position,
            'delta_position': [round(target_position[i] - init_position[i], 2) for i in range(3)],
            'init_orientation': init_orientation,
            'target_orientation': target_orientation,
            'transform_matrix': transform_matrix
        }
        with open(os.path.join(output_path, "result.json"), 'w') as f:
            json.dump(result, f, indent=4)
        print("Successfully saved result for", output_path)
    except Exception as e:
        print("Error in", p, e)


if __name__ == "__main__":
    dataset_paths = glob.glob("/data/datasets/open6dor_v2/*/*/*/*/")

    obj_parsing_model, obj_parsing_tokenizer, obj_parsing_image_processor = sofar_llava_inference.get_model(
        model_path="checkpoints/llava-v1.5-7b-obj-parsing-lora/", model_base="lmsys/vicuna-7b-v1.5", device="cuda"
    )
    vlm_model, vlm_tokenizer, vlm_image_processor = sofar_llava_inference.get_model(
        model_path="checkpoints/llava-v1.5-7b-vlm-lora/", model_base="lmsys/vicuna-7b-v1.5", device="cuda"
    )

    detection_model = detection.get_model()
    sam_model = sam.get_model()
    orientation_model = orientation.get_model()

    for p in tqdm(dataset_paths):
        process_dataset(p)
