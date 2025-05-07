import os
import cv2
import json
import warnings
from PIL import Image
from tqdm import tqdm

from serve.scene_graph import get_scene_graph_3d
from depth.utils import get_intrinsic, reconstruct_pcd
from segmentation import sam, grounding_dino as detection
from serve.chatgpt import vqa_parsing, vqa_spatial_reasoning
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)


def process_info(info):
    output_folder = "output"

    task = info["image"].split("/")[0]
    sample_id = info["image"].split("/")[1].split(".")[0]
    prompt = info["question"]
    answer = info["answer"]

    if task == "existence":
        prompt += " Response with Yes or No."

    image_path = f"datasets/SpatialBot/{info['image']}"
    image = Image.open(image_path).convert("RGB")
    depth_path = f"datasets/SpatialBot/{task}_d/{sample_id}.png"
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) / 1000.0

    fx, fy = get_intrinsic(image)
    intrinsic = [fx, fy, image.size[0] / 2, image.size[1] / 2]
    pcd = reconstruct_pcd(depth_img, intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3])

    info = vqa_parsing(prompt, image)
    print(json.dumps(info, indent=2))
    object_list = list(info.keys())

    detections = detection.get_detections(image, object_list, detection_model, output_folder=output_folder)
    mask, ann_img, object_names = sam.get_mask(
        image, object_list, sam_model, detections, output_folder=output_folder)

    scene_graph, _ = get_scene_graph_3d(image, pcd, mask, info, object_names, output_folder=output_folder)
    text = vqa_spatial_reasoning(image, prompt, scene_graph, eval=True)

    print(text)

    if text[:len(str(answer))] == str(answer):
        return True
    else:
        return False


if __name__ == "__main__":

    detection_model = detection.get_model()
    sam_model = sam.get_model()

    task_type = ['counting', 'existence', 'positional', 'reach', 'size']
    result = {
        "total": [],
        "counting": [],
        "existence": [],
        "positional": [],
        "reach": [],
        "size": []
    }

    for task in task_type:
        print(f"Task: {task}")
        info_list = json.load(open(f"datasets/SpatialBot/{task}.json"))
        total = len(info_list)

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(process_info, info): info for info in info_list}
            for future in tqdm(as_completed(futures), total=total):
                flag = future.result()
                result["total"].append(flag)
                result[task].append(flag)

    print("Total Accuracy: ", sum(result["total"]) / len(result["total"]))
    for task in task_type:
        print(f"{task} Accuracy: ", sum(result[task]) / len(result[task]))
