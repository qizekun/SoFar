import os
import json
import warnings
import numpy as np
from PIL import Image

from serve import pointso as orientation
from depth import metric3dv2 as depth_model
from serve.scene_graph import get_scene_graph
from segmentation import sam, florence as detection
from serve.chatgpt import vqa_parsing, vqa_spatial_reasoning

warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)

if __name__ == "__main__":
    image_path = "assets/table.jpg"
    prompt = "How far between the left bottle and the right bottle?"
    output_folder = "output"

    image = Image.open(image_path).convert("RGB")

    print("Load models...")
    detection_model = detection.get_model()
    sam_model = sam.get_model()
    orientation_model = orientation.get_model()
    metriced_model = depth_model.get_model()

    print("Start object parsing...")
    info = vqa_parsing(prompt, image)
    print(json.dumps(info, indent=2))
    object_list = list(info.keys())

    print("Start Segment Anything...")
    detections = detection.get_detections(image, object_list, detection_model, output_folder=output_folder)
    mask, ann_img, object_names = sam.get_mask(image, object_list, sam_model, detections, output_folder=output_folder)

    print("Predict depth map...")
    depth, _, pcd = depth_model.depth_estimation(image, metriced_model, output_folder=output_folder)
    np.save(os.path.join(output_folder, "scene.npy"), np.concatenate([pcd, np.array(image)], axis=-1).reshape(-1, 6))

    print("Generate scene graph...")
    scene_graph, _ = get_scene_graph(image, pcd, mask, info, object_names, orientation_model, output_folder=output_folder)

    print("objects info:")
    for node in scene_graph:
        print(node)

    print("Start spatial reasoning...")
    response = vqa_spatial_reasoning(ann_img, prompt, scene_graph)
    print(response)
