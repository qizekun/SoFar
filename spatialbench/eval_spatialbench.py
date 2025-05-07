import os
import json
import warnings
from PIL import Image
from tqdm import tqdm

from serve import pointso as orientation
from serve.scene_graph import get_scene_graph
from depth import metric3dv2 as depth_esti_model
from segmentation import sam, florence as detection
from serve.chatgpt import vqa_parsing, vqa_spatial_reasoning
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)


def process_info(info):
    id = info["id"]
    output_folder = "output"

    question = info["question"]
    options = info["options"]
    answer = info["answer"]
    task_type = info["task_type"]
    question_type = info["question_type"]

    prompt = question + "\n" + "A. " + options[0] + "\n" + "B. " + options[1] + "\n" + "C. " + options[
        2] + "\n" + "D. " + options[3]

    image_path = f"datasets/6dof_spatialbench/images/{id}.png"
    image = Image.open(image_path).convert("RGB")

    info = vqa_parsing(prompt, image)
    print(json.dumps(info, indent=2))
    object_list = list(info.keys())

    detections = detection.get_detections(image, object_list, detection_model, output_folder=output_folder)
    mask, ann_img, object_names = sam.get_mask(
        image, object_list, sam_model, detections, output_folder=output_folder)

    depth, _, pcd = depth_esti_model.depth_estimation(image, depth_model, output_folder=output_folder)
    scene_graph, _ = get_scene_graph(image, pcd, mask, info, object_names, orientation_model, output_folder=output_folder)
    text = vqa_spatial_reasoning(ann_img, prompt, scene_graph, eval=True)

    print(text)
    if ("A" == text[0] and answer == 0) or ("B" == text[0] and answer == 1) or ("C" == text[0] and answer == 2) or (
            "D" == text[0] and answer == 3):
        return True, task_type, question_type
    else:
        return False, task_type, question_type


if __name__ == "__main__":

    detection_model = detection.get_model()
    sam_model = sam.get_model()
    depth_model = depth_esti_model.get_model()
    orientation_model = orientation.get_model()

    info_list = json.load(open('datasets/6dof_spatialbench/spatial_data.json'))
    total = len(info_list)
    print("total: ", total)
    result = {
        "position": {
            "absolute": [],
            "relative": []
        },
        "orientation": {
            "absolute": [],
            "relative": []
        },
        "total": []
    }

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_info, info): info for info in info_list}
        for future in tqdm(as_completed(futures), total=total):
            flag, task_type, question_type = future.result()
            result["total"].append(flag)
            if task_type == "position":
                if question_type == "absolute":
                    result["position"]["absolute"].append(flag)
                else:
                    result["position"]["relative"].append(flag)
            else:
                if question_type == "absolute":
                    result["orientation"]["absolute"].append(flag)
                else:
                    result["orientation"]["relative"].append(flag)

    print("Position relative accuracy: ", sum(result["position"]["relative"]) / len(result["position"]["relative"]))
    print("Position absolute accuracy: ", sum(result["position"]["absolute"]) / len(result["position"]["absolute"]))
    print("Orientation relative accuracy: ",
          sum(result["orientation"]["relative"]) / len(result["orientation"]["relative"]))
    print("Orientation absolute accuracy: ",
          sum(result["orientation"]["absolute"]) / len(result["orientation"]["absolute"]))
    print("Total accuracy: ", sum(result["total"]) / len(result["total"]))
