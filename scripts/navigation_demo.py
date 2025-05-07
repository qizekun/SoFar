import os
import warnings
import numpy as np
from PIL import Image
from depth.utils import depth2pcd
from serve import pointso as orientation
from segmentation import sam, grounding_dino as detection


warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)


def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(theta), theta


if __name__ == "__main__":
    image_path = "navigation.png"
    depth_path = "navigation.npy"
    navigate_obj = ["microwave"]
    instruction = ["front view"]
    output_folder = "output"
    intrinsic = [385.327, 384.862, 322.151, 246.106]

    image = Image.open(image_path).convert("RGB")
    depth = np.load(depth_path)
    pcd_camera, _ = depth2pcd(depth, intrinsic)
    scene_pcd = np.concatenate((pcd_camera.reshape(-1, 3), np.array(image).reshape(-1, 3)), axis=-1)
    np.save(os.path.join(output_folder, "scene.npy"), scene_pcd)

    print("Load models...")
    detection_model = detection.get_model()
    sam_model = sam.get_model()
    orientation_model = orientation.get_model()

    print("Start Segment Anything...")
    detections = detection.get_detections(image, navigate_obj, detection_model, output_folder=output_folder, single=True)
    mask, ann_img, object_names = sam.get_mask(image, navigate_obj, sam_model, detections, output_folder=output_folder)

    image = np.array(image)
    object_mask = mask[0]
    segmented_object = pcd_camera[object_mask]
    segmented_image = image[object_mask]
    colored_object_pcd = np.concatenate((segmented_object.reshape(-1, 3), segmented_image.reshape(-1, 3)), axis=-1)
    np.save(os.path.join(output_folder, f"navigate_obj.npy"), colored_object_pcd)
    center = colored_object_pcd.mean(axis=0)[:3] / 1000
    print(f"Obj Center: {center}")

    orientation = orientation.pred_orientation(orientation_model, colored_object_pcd, instruction)
    print(f"Predict orientation: {instruction}--{orientation}")
    
    vector = orientation[0]
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])

    angle_x, theta_x = angle_between_vectors(vector, x_axis)
    angle_y, theta_y = angle_between_vectors(vector, y_axis)
    angle_z, theta_z = angle_between_vectors(vector, z_axis)

    print(f"Angle with the x-axis: {angle_x:.2f}°")
    print(f"Angle with the y-axis: {angle_y:.2f}°")
    print(f"Angle with the z-axis: {angle_z:.2f}°")
    print("\n\n\n")
    print(f"Turn right: {int(180 - angle_z)}°")
    target_z = center[2] + np.cos(theta_z) * 0.5
    target_x = center[0] + np.cos(theta_x) * 0.5
    print(f"Target position: {target_z, target_x}")
