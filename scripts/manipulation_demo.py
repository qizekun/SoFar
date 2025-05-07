import os
import json
import warnings
import numpy as np
from PIL import Image

from depth.utils import depth2pcd
from serve import pointso as orientation
from serve.scene_graph import get_scene_graph
from segmentation import sam, florence as detection
from serve.chatgpt import manip_parsing, manip_spatial_reasoning
from serve.utils import generate_rotation_matrix, remove_outliers

warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)

if __name__ == "__main__":
    image_path = "assets/drawer.png"
    depth_path = "assets/drawer.npy"
    prompt = "Open top drawer."
    output_folder = "output"

    image = Image.open(image_path).convert("RGB")
    depth = np.load(depth_path)
    intrinsic = [425, 425, 320, 256]
    extrinsic = np.array([[-2.85973249e-03, -7.06918473e-01, 7.07288915e-01, 1.45990583e-01],
                          [-9.99995619e-01, 2.02166464e-03, -2.02260415e-03, -4.17460955e-04],
                          [-8.70080406e-08, -7.07292162e-01, -7.06921387e-01, 1.21335849e+00],
                          [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    pcd_camera, pcd = depth2pcd(depth, intrinsic, extrinsic)
    scene_pcd = np.concatenate((pcd.reshape(-1, 3), np.array(image).reshape(-1, 3)), axis=-1)
    np.save(os.path.join(output_folder, "scene.npy"), scene_pcd)

    print("Load models...")
    detection_model = detection.get_model()
    sam_model = sam.get_model()
    orientation_model = orientation.get_model()

    print("Start object parsing...")
    info = manip_parsing(prompt, image)
    print(json.dumps(info, indent=2))
    object_list = list(info.keys())

    print("Start Segment Anything...")
    detections = detection.get_detections(image, object_list, detection_model, output_folder=output_folder)
    mask, ann_img, object_names = sam.get_mask(
        image, object_list, sam_model, detections, output_folder=output_folder)

    print("Generate scene graph...")
    objects_info, objects_dict = get_scene_graph(image, pcd, mask, info, object_names, orientation_model,
                                                 output_folder=output_folder)
    print("objects info:")
    for node in objects_info:
        print(node)

    print("Start spatial reasoning...")
    response = manip_spatial_reasoning(image, prompt, objects_info)
    print(response)

    interact_object_id = response["interact_object_id"]
    interact_object_dict = objects_dict[interact_object_id - 1]
    init_position = interact_object_dict["center"]
    target_position = response["target_position"]
    init_orientation = interact_object_dict["orientation"]
    target_orientation = response["target_orientation"]

    image = np.array(image)
    object_mask = mask[interact_object_id - 1]
    segmented_object = pcd[object_mask]
    segmented_image = image[object_mask]
    colored_object_pcd = np.concatenate((segmented_object.reshape(-1, 3), segmented_image.reshape(-1, 3)), axis=-1)
    colored_object_pcd = remove_outliers(colored_object_pcd)
    np.save(os.path.join(output_folder, f"picked_obj.npy"), colored_object_pcd)

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
