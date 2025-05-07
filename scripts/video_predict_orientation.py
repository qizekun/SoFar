import os
import glob
import warnings
import numpy as np
from PIL import Image
from tqdm import tqdm
from depth.utils import depth2pcd
from serve import pointso as orientation
from segmentation import sam, grounding_dino as detection
from serve.visualization import draw_vector_with_text, save_images_as_video

warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)

if __name__ == "__main__":
    video_image_paths = sorted(glob.glob("2025_01_20_20_27_10_1/images/*.jpg"),
                               key=lambda x: int(x.split("/")[-1].split(".")[0]))
    navigate_obj = ["microwave"]
    instruction = ["front"]
    output_folder = "output"
    intrinsic = [385.327, 384.862, 322.151, 246.106]
    ema = 0.9
    camera_matrix = np.array([[intrinsic[0], 0, intrinsic[2]], [0, intrinsic[1], intrinsic[3]], [0, 0, 1]])

    print("Load models...")
    detection_model = detection.get_model()
    sam_model = sam.get_model()
    orientation_model = orientation.get_model()

    print("Start inference...")
    output_image_files = []
    history_vector = None

    for image_path in tqdm(video_image_paths):
        depth_path = image_path.split(".")[0] + ".npy"

        image = Image.open(image_path).convert("RGB")
        depth = np.load(depth_path)
        pcd_camera, _ = depth2pcd(depth, intrinsic)
        scene_pcd = np.concatenate((pcd_camera.reshape(-1, 3), np.array(image).reshape(-1, 3)), axis=-1)

        detections = detection.get_detections(image, navigate_obj, detection_model, output_folder=output_folder, single=True)
        mask, ann_img, object_names = sam.get_mask(image, navigate_obj, sam_model, detections, output_folder=output_folder)

        image = np.array(image)
        object_mask = mask[0]
        segmented_object = pcd_camera[object_mask]
        segmented_image = image[object_mask]
        colored_object_pcd = np.concatenate((segmented_object.reshape(-1, 3), segmented_image.reshape(-1, 3)), axis=-1)
        center = colored_object_pcd.mean(axis=0)[:3] / 1000

        orientation = orientation.pred_orientation(orientation_model, colored_object_pcd, instruction)
        vector = np.array(orientation[0])
        vector[1] = 0
        if history_vector is None:
            history_vector = vector
        vector = history_vector * ema + vector * (1 - ema)

        image_with_vector = draw_vector_with_text(
            image=Image.open(image_path).convert("RGB"),
            start_camera_point=center,
            direction=vector,
            camera_matrix=camera_matrix,
            text=instruction[0])
        output_image_files.append(image_with_vector)

    print("Save videos...")
    output_path = os.path.join(output_folder, "navigation.mp4")
    save_images_as_video(output_image_files, output_path)
