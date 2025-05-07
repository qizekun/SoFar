import os
import json
import base64
import warnings
from PIL import Image
from tqdm import tqdm
from io import BytesIO

from serve import pointso as orientation
from serve.scene_graph import get_scene_graph
from depth import metric3dv2 as depth_esti_model
from segmentation import sam, florence as detection
from serve.chatgpt import vqa_parsing, vqa_spatial_reasoning
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)


def process_info(info):
    output_folder = "output"

    question = info["question"]
    options = info["answer_options"]
    answer = info["answer"]
    relation = info["relation"]
    base64_string = info["image"]

    prompt = question + "\n" + "A. " + options[0] + "\n" + "B. " + options[1] + "\n" + "C. " + options[
        2] + "\n" + "D. " + options[3]

    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data)).convert("RGB")

    info = vqa_parsing(prompt, image)
    print(json.dumps(info, indent=2))
    object_list = list(info.keys())

    detections = detection.get_detections(image, object_list, detection_model, output_folder=output_folder)
    mask, ann_img, object_names = sam.get_mask(
        image, object_list, sam_model, detections, output_folder=output_folder)

    depth, img, pcd = depth_esti_model.depth_estimation(image, metriced_model, output_folder=output_folder)
    scene_graph, _ = get_scene_graph(image, pcd, mask, info, object_names, orientation_model, output_folder=output_folder)
    text = vqa_spatial_reasoning(ann_img, prompt, scene_graph, eval=True)

    print(text)
    if ("A" == text[0] and answer == 0) or ("B" == text[0] and answer == 1) or ("C" == text[0] and answer == 2) or (
            "D" == text[0] and answer == 3):
        return True, relation
    else:
        return False, relation


if __name__ == "__main__":

    detection_model = detection.get_model()
    sam_model = sam.get_model()
    metriced_model = depth_esti_model.get_model()
    orientation_model = orientation.get_model()

    info_list = json.load(open('datasets/embspatial_bench.json'))
    total = len(info_list)
    print("total: ", total)
    result = {
        "close": [],
        "far": [],
        "left": [],
        "right": [],
        "above": [],
        "under": [],
        "total": []
    }

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_info, info): info for info in info_list}
        for future in tqdm(as_completed(futures), total=total):
            flag, relation = future.result()
            result["total"].append(flag)
            result[relation].append(flag)

    print("Success Rate:")
    for key in result:
        print(f"{key}: {sum(result[key]) / len(result[key])}")
