import os
import numpy as np
from serve import pointso
from serve.utils import remove_outliers


def open6dor_scene_graph(image, pcd, mask, info, object_names, orientation_model, output_folder="output"):
    n, h, w = mask.shape
    image = np.array(image)

    picked_object_name = info["picked_object"]
    index = object_names.index(picked_object_name)

    object_mask = mask[index]
    segmented_object = pcd[object_mask]
    segmented_image = image[object_mask]
    colored_object_pcd = np.concatenate((segmented_object.reshape(-1, 3), segmented_image.reshape(-1, 3)), axis=-1)
    np.save(os.path.join(output_folder, "picked_obj_mask.npy"), object_mask)
    np.save(os.path.join(output_folder, "picked_obj.npy"), colored_object_pcd)

    segmented_object = remove_outliers(segmented_object)
    colored_object_pcd = remove_outliers(colored_object_pcd)
    min_values = segmented_object.min(axis=0)
    max_values = segmented_object.max(axis=0)
    mean_values = segmented_object.mean(axis=0)

    center = [round(mean_values[0], 2), round(mean_values[1], 2), round(mean_values[2], 2)]
    center_str = f"x: {mean_values[0]:.2f}, y: {mean_values[1]:.2f}, z: {mean_values[2]:.2f}"
    bbox = [
        [min_values[0], max_values[0]],
        [min_values[1], max_values[1]],
        [min_values[2], max_values[2]]
    ]
    bbox_str = {
        "x_min ~ x_max": f"{min_values[0]:.2f} ~ {max_values[0]:.2f}",
        "y_min ~ y_max": f"{min_values[1]:.2f} ~ {max_values[1]:.2f}",
        "z_min ~ z_max": f"{min_values[2]:.2f} ~ {max_values[2]:.2f}"
    }
    orientation = {}
    orientation_str = {}
    direction_attributes = info["direction_attributes"]
    if len(direction_attributes) > 0:
        pred = pointso.pred_orientation(orientation_model, colored_object_pcd, direction_attributes)
        for j, attribute in enumerate(direction_attributes):
            orientation[attribute] = pred[j]
            orientation_str[attribute] = f"direction vector: ({pred[j][0]:.2f}, {pred[j][1]:.2f}, {pred[j][2]:.2f})"
    picked_object_dict = {
        'object name': object_names[index],
        'center': center,
        'bounding box': bbox,
        'orientation': orientation
    }
    picked_object_info = {
        'object name': object_names[index],
        'center': center_str,
        'bounding box': bbox_str,
        # 'orientation': orientation_str
    }

    obj_id = 0
    other_objects_info = []
    for i in range(n):
        if i == index:
            continue
        else:
            obj_id += 1
        object_mask = mask[i]
        segmented_object = pcd[object_mask]
        segmented_image = image[object_mask]
        colored_object_pcd = np.concatenate((segmented_object.reshape(-1, 3), segmented_image.reshape(-1, 3)), axis=-1)
        np.save(os.path.join(output_folder, f"else_obj_{obj_id}.npy"), colored_object_pcd)

        segmented_object = remove_outliers(segmented_object)
        min_values = segmented_object.min(axis=0)
        max_values = segmented_object.max(axis=0)
        mean_values = segmented_object.mean(axis=0)
        center = f"x: {mean_values[0]:.2f}, y: {mean_values[1]:.2f}, z: {mean_values[2]:.2f}"
        bbox = {
            "x_min ~ x_max": f"{min_values[0]:.2f} ~ {max_values[0]:.2f}",
            "y_min ~ y_max": f"{min_values[1]:.2f} ~ {max_values[1]:.2f}",
            "z_min ~ z_max": f"{min_values[2]:.2f} ~ {max_values[2]:.2f}"
        }

        node = {
            'object name': object_names[i],
            'center': center,
            'bounding box': bbox
        }
        other_objects_info.append(node)

    return picked_object_info, other_objects_info, picked_object_dict


def open6dor_scene_graph_for_mismatch_pcd(image, pcd, mask, info, object_names, orientation_model, output_folder="output"):
    from skimage.morphology import disk
    from scipy.ndimage import binary_dilation
    n, h, w = mask.shape
    image = np.array(image)

    picked_object_name = info["picked_object"]
    index = object_names.index(picked_object_name)

    # pcd size is not equal to mask size, use relative scale to get the mask
    object_mask = mask[index]

    radius = 10
    struct_element = disk(radius)
    object_mask = binary_dilation(object_mask, structure=struct_element)

    H, W = image.shape[:2]
    x_min, x_max = pcd[:, 0].min(), pcd[:, 0].max()
    y_min, y_max = pcd[:, 1].min(), pcd[:, 1].max()
    segmented_object = []
    mask_index = []
    for p in pcd:
        i_index = (x_max - p[0]) / (x_max - x_min) * (W - 1)
        j_index = (p[1] - y_min) / (y_max - y_min) * (H - 1)
        i_index = int(i_index)
        j_index = int(j_index)
        if object_mask[j_index, i_index]:
            segmented_object.append(list(p) + list(image[j_index, i_index]))
            mask_index.append(True)
        else:
            mask_index.append(False)
    segmented_object = np.array(segmented_object)
    np.save(os.path.join(output_folder, "picked_obj_mask.npy"), object_mask)
    np.save(os.path.join(output_folder, "picked_obj.npy"), segmented_object.reshape(-1, 6))

    segmented_object = remove_outliers(segmented_object)
    min_values = segmented_object.min(axis=0).astype(np.float64).tolist()
    max_values = segmented_object.max(axis=0).astype(np.float64).tolist()
    mean_values = segmented_object.mean(axis=0).astype(np.float64).tolist()

    center = [round(mean_values[0], 2), round(mean_values[1], 2), round(mean_values[2], 2)]
    center_str = f"x: {mean_values[0]:.2f}, y: {mean_values[1]:.2f}, z: {mean_values[2]:.2f}"
    bbox = [
        [min_values[0], max_values[0]],
        [min_values[1], max_values[1]],
        [min_values[2], max_values[2]]
    ]
    bbox_str = {
        "x_min ~ x_max": f"{min_values[0]:.2f} ~ {max_values[0]:.2f}",
        "y_min ~ y_max": f"{min_values[1]:.2f} ~ {max_values[1]:.2f}",
        "z_min ~ z_max": f"{min_values[2]:.2f} ~ {max_values[2]:.2f}"
    }
    orientation = {}
    orientation_str = {}
    direction_attributes = info["direction_attributes"]
    if len(direction_attributes) > 0:
        pred = pointso.pred_orientation(orientation_model, segmented_object.reshape(-1, 6), direction_attributes)
        for j, attribute in enumerate(direction_attributes):
            orientation[attribute] = pred[j]
            orientation_str[attribute] = f"direction vector: ({pred[j][0]:.2f}, {pred[j][1]:.2f}, {pred[j][2]:.2f})"
    picked_object_dict = {
        'object name': object_names[index],
        'center': center,
        'bounding box': bbox,
        'orientation': orientation
    }
    picked_object_info = {
        'object name': object_names[index],
        'center': center_str,
        'bounding box': bbox_str,
        # 'orientation': orientation_str
    }

    obj_id = 0
    other_objects_info = []
    for i in range(n):
        if i == index:
            continue
        else:
            obj_id += 1
        object_mask = mask[i]

        radius = 10
        struct_element = disk(radius)
        object_mask = binary_dilation(object_mask, structure=struct_element)

        H, W = image.shape[:2]
        x_min, x_max = pcd[:, 0].min(), pcd[:, 0].max()
        y_min, y_max = pcd[:, 1].min(), pcd[:, 1].max()
        segmented_object = []
        for p in pcd:
            i_index = (x_max - p[0]) / (x_max - x_min) * (W - 1)
            j_index = (p[1] - y_min) / (y_max - y_min) * (H - 1)
            i_index = int(i_index)
            j_index = int(j_index)
            if object_mask[j_index, i_index]:
                segmented_object.append(list(p) + list(image[j_index, i_index]))
        segmented_object = np.array(segmented_object)
        np.save(os.path.join(output_folder, f"else_obj_{obj_id}.npy"), segmented_object.reshape(-1, 6))

        segmented_object = remove_outliers(segmented_object)
        min_values = segmented_object.min(axis=0).astype(np.float64).tolist()
        max_values = segmented_object.max(axis=0).astype(np.float64).tolist()
        mean_values = segmented_object.mean(axis=0).astype(np.float64).tolist()
        center = f"x: {mean_values[0]:.2f}, y: {mean_values[1]:.2f}, z: {mean_values[2]:.2f}"
        bbox = {
            "x_min ~ x_max": f"{min_values[0]:.2f} ~ {max_values[0]:.2f}",
            "y_min ~ y_max": f"{min_values[1]:.2f} ~ {max_values[1]:.2f}",
            "z_min ~ z_max": f"{min_values[2]:.2f} ~ {max_values[2]:.2f}"
        }

        node = {
            'object name': object_names[i],
            'center': center,
            'bounding box': bbox
        }
        other_objects_info.append(node)

    return picked_object_info, other_objects_info, picked_object_dict, np.array(mask_index)


def get_scene_graph(image, pcd, mask, info, object_names, orientation_model, output_folder="output"):
    if len(mask) == 0:
        return [], []
    n, h, w = mask.shape
    image = np.array(image)

    objects_info = []
    objects_dict = []
    for i in range(n):
        object_mask = mask[i]
        segmented_object = pcd[object_mask]
        segmented_image = image[object_mask]
        colored_object_pcd = np.concatenate((segmented_object.reshape(-1, 3), segmented_image.reshape(-1, 3)), axis=-1)
        np.save(os.path.join(output_folder, f"obj_{i + 1}.npy"), colored_object_pcd)

        segmented_object = remove_outliers(segmented_object)
        colored_object_pcd = remove_outliers(colored_object_pcd)
        min_values = segmented_object.min(axis=0)
        max_values = segmented_object.max(axis=0)
        mean_values = segmented_object.mean(axis=0)
        center = f"x: {mean_values[0]:.2f}, y: {mean_values[1]:.2f}, z: {mean_values[2]:.2f}"
        bbox = {
            "x_min ~ x_max": f"{min_values[0]:.2f} ~ {max_values[0]:.2f}",
            "y_min ~ y_max": f"{min_values[1]:.2f} ~ {max_values[1]:.2f}",
            "z_min ~ z_max": f"{min_values[2]:.2f} ~ {max_values[2]:.2f}"
        }
        orientation = {}
        orientation_str = {}
        direction_attributes = info[object_names[i]]
        if len(direction_attributes) > 0:
            pred = pointso.pred_orientation(orientation_model, colored_object_pcd, direction_attributes)
            for j, attribute in enumerate(direction_attributes):
                orientation[attribute] = pred[j]
                orientation_str[attribute] = f"direction vector: ({pred[j][0]:.2f}, {pred[j][1]:.2f}, {pred[j][2]:.2f})"

        node = {
            'id': i + 1,
            'object name': object_names[i],
            'center': center,
            'bounding box': bbox,
            'orientation': orientation_str
        }
        objects_info.append(node)

        node_dict = {
            'object name': object_names[i],
            'center': [round(mean_values[0], 2), round(mean_values[1], 2), round(mean_values[2], 2)],
            'bounding box': [
                [min_values[0], max_values[0]],
                [min_values[1], max_values[1]],
                [min_values[2], max_values[2]]
            ],
            'orientation': orientation
        }
        objects_dict.append(node_dict)

    return objects_info, objects_dict


def get_scene_graph_3d(image, pcd, mask, info, object_names, output_folder="output"):
    if len(mask) == 0:
        return [], []
    n, h, w = mask.shape
    image = np.array(image)

    objects_info = []
    objects_dict = []
    for i in range(n):
        object_mask = mask[i]
        segmented_object = pcd[object_mask]
        segmented_image = image[object_mask]
        colored_object_pcd = np.concatenate((segmented_object.reshape(-1, 3), segmented_image.reshape(-1, 3)), axis=-1)
        np.save(os.path.join(output_folder, f"obj_{i + 1}.npy"), colored_object_pcd)

        segmented_object = remove_outliers(segmented_object)
        min_values = segmented_object.min(axis=0)
        max_values = segmented_object.max(axis=0)
        mean_values = segmented_object.mean(axis=0)
        center = f"x: {mean_values[0]:.2f}, y: {mean_values[1]:.2f}, z: {mean_values[2]:.2f}"
        bbox = {
            "x_min ~ x_max": f"{min_values[0]:.2f} ~ {max_values[0]:.2f}",
            "y_min ~ y_max": f"{min_values[1]:.2f} ~ {max_values[1]:.2f}",
            "z_min ~ z_max": f"{min_values[2]:.2f} ~ {max_values[2]:.2f}"
        }
        node = {
            'id': i + 1,
            'object name': object_names[i],
            'center': center,
            'bounding box': bbox,
        }
        objects_info.append(node)

        node_dict = {
            'object name': object_names[i],
            'center': [round(mean_values[0], 2), round(mean_values[1], 2), round(mean_values[2], 2)],
            'bounding box': [
                [min_values[0], max_values[0]],
                [min_values[1], max_values[1]],
                [min_values[2], max_values[2]]
            ]
        }
        objects_dict.append(node_dict)

    return objects_info, objects_dict
