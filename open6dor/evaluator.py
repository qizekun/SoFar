"""
This file contains the evaluation metrics for Open6DOR Benchmark.
We are currently refining the rotation eval section for fairer evaluation and easier comparison.
Full version coming soon.
"""
import numpy as np
import math
import json
import os
import glob
from scipy.spatial.transform import Rotation as R


def projection(rot_mat_A, rot_mat_B, axis):
    """
    Project the relative rotation from A to B onto the axis.
    rot_mat: 3x3 rotation matrix
        A: ground truth rotation
        B: predicted rotation
    axis: 3x1 vector
    """
    det = np.linalg.det(rot_mat_A)
    assert det != 0  # rotation matrix should have determinant +1 or -1
    v = np.linalg.inv(rot_mat_A) @ axis

    w = rot_mat_B @ v
    angle = np.arccos(np.dot(axis, w) / (np.linalg.norm(axis) * np.linalg.norm(w)))
    return np.degrees(angle)


quat_gt = [
    -0.5333969593048096,
    -0.5333969593048096,
    0.4642065465450287,
    0.4642064869403839
]
rot_gt = R.from_quat(quat_gt).as_matrix()


def normalize_quat(quat):
    norm = math.sqrt(sum(q ** 2 for q in quat))
    return [q / norm for q in quat]


def angle_deviation(quat0, quat1):
    # Normalize the quaternions
    quat0 = normalize_quat(quat0)
    quat1 = normalize_quat(quat1)

    # Compute the dot product of the two quaternions
    dot_product = sum(q0 * q1 for q0, q1 in zip(quat0, quat1))

    # Ensure the dot product is within the range [-1, 1] to avoid numerical errors
    dot_product = max(-1.0, min(1.0, dot_product))

    # Compute the angle deviation (in radians)
    angle_deviation = 2 * math.acos(dot_product)

    # Convert the angle deviation to degrees
    angle_deviation_degrees = math.degrees(angle_deviation)
    if angle_deviation_degrees > 180:
        angle_deviation_degrees = 360 - angle_deviation_degrees

    return angle_deviation_degrees


def evaluate_rot(task_config_file, quat_pred):
    """
    Evaluate the predicted rotation.
    task_id: str
    quat_pred: list of 4 floats
    """
    # load the ground truth quaternion
    prompt = ""
    with open(task_config_file, "r") as f:
        task_config = json.load(f)
    init_quat = task_config["init_obj_pos"][-1][3:7]

    try:
        init_rot = R.from_quat(init_quat).as_matrix()
    except:
        return -1
    rotation_instruction = task_config["rotation_instruction"]
    annot = None

    for instruction in task_config["anno_target"]["annotation"].keys():
        if instruction in rotation_instruction:
            annot = task_config["anno_target"]["annotation"][instruction]
            prompt = instruction
            break
    if annot is None:
        return "No annotation found"
    quat_gt_list = annot["quat"]
    ax = annot["axis"]
    stage = annot["stage"]
    if stage == 2:
        return "Annotation stage 2"

    if ax == "x":
        axis = np.array([1, 0, 0])
    elif ax == "y":
        axis = np.array([0, 1, 0])
    elif ax == "z":
        axis = np.array([0, 0, 1])
    elif ax == "n":
        axis = None
    else:  # "0"
        axis = None
        return "No axis specified"
    rot_pred = R.from_quat(quat_pred).as_matrix()  # transformation
    rot_pred = rot_pred @ init_rot

    deviation_list = []

    for quat_gt in quat_gt_list:
        if not isinstance(quat_gt, list):
            quat_gt = quat_gt_list
        rot_gt = R.from_quat(quat_gt).as_matrix()
        deviation = -1
        if isinstance(axis, np.ndarray):
            try:
                deviation = projection(rot_gt, rot_pred, axis)
            except:
                import pdb
                pdb.set_trace()

        else:
            deviation = angle_deviation(quat_gt, quat_pred)
        deviation_list.append(deviation)
    deviation_min = min(deviation_list)

    return deviation_min


def evaluate_gt_rot_exp(output_root, task_root, eval_file):
    eval_dict = {}
    for root, dirs, files in os.walk(output_root):
        for dir in dirs:
            task = dir
            task_path = os.path.join(root, dir)
            if os.path.exists(os.path.join(task_path, "ammend_rotation.txt")):
                rot_file = os.path.join(task_path, "ammend_rotation.txt")
            else:
                rot_file = os.path.join(task_path, "proposed_rotation.txt")
            config_file = glob.glob(os.path.join(task_root, f"**/{task}/task_config_new2.json"))[0]
            with open(config_file, "r") as f:
                task_config = json.load(f)
            if not os.path.exists(rot_file):
                continue
            with open(rot_file, "r") as f:
                lines = f.readlines()
            quaternion_line = None

            for line in lines:
                if line.startswith("local rotation:"):
                    quaternion_line = lines[lines.index(line) + 1]

                    local_rotation_str = quaternion_line.split(']')[0] + ']'
                    break

            if local_rotation_str:  # quaternion_line:

                # Convert the string representation of the quaternion to a list of floats
                assert ',' in quaternion_line
                if "vlm" in quaternion_line:
                    quaternion = quaternion_line
                    deviation = -1
                else:
                    quaternion = [float(x) for x in local_rotation_str.strip('[]\n').split(',')]

                    deviation = evaluate_rot(config_file, quaternion)
                eval_dict[task] = {"deviation": deviation, "proposal": quaternion}
            else:
                import pdb
                pdb.set_trace()
        break
    with open(eval_file, "w") as f:
        json.dump(eval_dict, f, indent=4)
    return eval_dict


def evaluate_6dof_exp(output_root, task_root, eval_file):
    eval_dict = {}
    for root, dirs, files in os.walk(output_root):
        for dir in dirs:
            task = dir
            task_path = os.path.join(root, dir)
            if os.path.exists(os.path.join(task_path, "ammend_rotation_50.txt")):
                rot_file = os.path.join(task_path, "ammend_rotation_50.txt")
            else:
                rot_file = os.path.join(task_path, "proposed_rotation.txt")
            config_file = glob.glob(os.path.join(task_root, f"**/**/{task}/task_config_new4.json"))[0]
            with open(config_file, "r") as f:
                task_config = json.load(f)
            label = task_config["rot_tag_detail"]
            level = task_config["rot_tag_level"]
            if not os.path.exists(rot_file):
                continue
            with open(rot_file, "r") as f:
                lines = f.readlines()
            quaternion_line = None

            for line in lines:
                if line.startswith("proposed rotation:"):
                    quaternion_line = lines[lines.index(line) + 1]
                    break

            if quaternion_line:
                # Convert the string representation of the quaternion to a list of floats
                if ',' in quaternion_line:
                    quaternion = [float(x) for x in quaternion_line.strip('[]\n').split(',')]
                    deviation = evaluate_rot(config_file, quaternion)

                elif "vlm" in quaternion_line:
                    quaternion = quaternion_line
                    deviation = -1
                else:
                    quaternion = [float(x) for x in quaternion_line.strip('[]\n').split()]

                    deviation = evaluate_rot(config_file, quaternion)
                eval_dict[task] = {"deviation": deviation, "proposal": quaternion, "label": label, "level": level}
            else:
                import pdb
                pdb.set_trace()
        break
    with open(eval_file, "w") as f:
        json.dump(eval_dict, f, indent=4)
    return eval_dict


def evaluate_posi(tar_pos, mode, sel_pos=None, sel_pos_1=None, sel_pos_2=None, sel_pos_all=None):
    """
    Evaluate the predicted position.
    """
    succ = 0
    if mode in ["left", "right", "front", "back", "behind", "top"]:
        if mode == "left":
            succ += sel_pos[1] > tar_pos[1]
        elif mode == "right":
            succ += sel_pos[1] < tar_pos[1]
        elif mode == "front":
            succ += sel_pos[0] < tar_pos[0]
        elif mode == "back" or mode == "behind":
            succ += sel_pos[0] > tar_pos[0]
        elif mode == "top":
            succ += sel_pos[2] <= tar_pos[2]
    elif mode == "between":
        max_sel_pos_x = np.max([sel_pos_1[0], sel_pos_2[0]])
        max_sel_pos_y = np.max([sel_pos_1[1], sel_pos_2[1]])
        min_sel_pos_x = np.min([sel_pos_1[0], sel_pos_2[0]])
        min_sel_pos_y = np.min([sel_pos_1[1], sel_pos_2[1]])
        succ += (min_sel_pos_x < tar_pos[0] < max_sel_pos_x) or (min_sel_pos_y < tar_pos[0] < max_sel_pos_y)
    elif mode == "center":
        max_sel_pos_x = np.max(sel_pos_all, axis=0)[0]
        min_sel_pos_x = np.min(sel_pos_all, axis=0)[0]
        max_sel_pos_y = np.max(sel_pos_all, axis=0)[1]
        min_sel_pos_y = np.min(sel_pos_all, axis=0)[1]
        succ += (min_sel_pos_x < tar_pos[0] < max_sel_pos_x) and (min_sel_pos_y < tar_pos[1] < max_sel_pos_y)
    return succ
