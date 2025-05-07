import os
import warnings
import numpy as np
from PIL import Image
from depth.utils import depth2pcd
from segmentation import sam, florence as detection

warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)

if __name__ == "__main__":
    image_path = "assets/simpler_env.png"
    depth_path = "assets/simpler_env.npy"
    picked_obj = ["coke can"]
    output_folder = "output"

    image = Image.open(image_path).convert("RGB")
    depth = np.load(depth_path)
    intrinsic = [425, 425, 320, 256]
    extrinsic = np.array([[2.8595643e-03, 9.9999607e-01, -8.6962245e-08, -2.0099998e-01],
                          [7.0691872e-01, -2.0215493e-03, -7.0729196e-01, 7.7029693e-01],
                          [-7.0728922e-01, 2.0224855e-03, -7.0692158e-01, 1.0575197e+00],
                          [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]])
    pcd_camera, pcd = depth2pcd(depth, intrinsic, extrinsic)
    scene_pcd = np.concatenate((pcd.reshape(-1, 3), np.array(image).reshape(-1, 3)), axis=-1)
    np.save(os.path.join(output_folder, "scene.npy"), scene_pcd)

    print("Load models...")
    detection_model = detection.get_model()
    sam_model = sam.get_model()

    print("Start Segment Anything...")
    detections = detection.get_detections(image, picked_obj, detection_model, output_folder=output_folder, single=True)
    mask, ann_img, object_names = sam.get_mask(image, picked_obj, sam_model, detections, output_folder=output_folder)

    image = np.array(image)
    picked_object_name = picked_obj[0]
    index = object_names.index(picked_object_name)
    object_mask = mask[index]
    segmented_object = pcd[object_mask]
    segmented_image = image[object_mask]
    colored_object_pcd = np.concatenate((segmented_object.reshape(-1, 3), segmented_image.reshape(-1, 3)), axis=-1)
    np.save(os.path.join(output_folder, f"picked_obj.npy"), colored_object_pcd)
