import torch
import random
import numpy as np
from sklearn.cluster import DBSCAN


def calculate_rotation_matrix(vectors_orig, vectors_trans):
    """
    Kabsch algorithm to calculate the rotation matrix that rotates the vectors_orig to vectors_trans
    """
    vectors_orig = np.array(vectors_orig)
    vectors_trans = np.array(vectors_trans)

    H = np.dot(vectors_orig.T, vectors_trans)
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    return R


def single_transform_matrix(vectors_orig, vectors_trans):
    v1 = vectors_orig[0]
    v2 = vectors_trans[0]

    axis = np.cross(v1, v2)
    axis_norm = np.linalg.norm(axis)

    if axis_norm == 0:
        R = np.eye(3)
    else:
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        u = axis / axis_norm

        ux, uy, uz = u
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        R = np.array([
            [cos_t + ux ** 2 * (1 - cos_t), ux * uy * (1 - cos_t) - uz * sin_t, ux * uz * (1 - cos_t) + uy * sin_t],
            [uy * ux * (1 - cos_t) + uz * sin_t, cos_t + uy ** 2 * (1 - cos_t), uy * uz * (1 - cos_t) - ux * sin_t],
            [uz * ux * (1 - cos_t) - uy * sin_t, uz * uy * (1 - cos_t) + ux * sin_t, cos_t + uz ** 2 * (1 - cos_t)]
        ])

    return R


def generate_rotation_matrix(directions_orig, directions_trans, threshold=0.5):
    directions_orig = np.array([v / np.linalg.norm(v) for v in directions_orig])
    directions_trans = np.array([v / np.linalg.norm(v) for v in directions_trans])

    if len(directions_orig) == 1:
        return single_transform_matrix(directions_orig, directions_trans)

    cos_angle_orig = np.dot(directions_orig[0], directions_orig[1])
    cos_angle_trans = np.dot(directions_trans[0], directions_trans[1])

    if abs(cos_angle_orig - cos_angle_trans) > threshold:
        print("only use the first direction vector")
        return calculate_rotation_matrix([directions_orig[0]], [directions_trans[0]])

    return calculate_rotation_matrix(directions_orig, directions_trans)


def get_point_cloud_from_rgbd(camera_depth, camera_rgb, camera_view_matrix_inv, camera_proj_matrix):
    camera_depth_tensor = torch.tensor(camera_depth)
    camera_rgb_tensor = torch.tensor(camera_rgb)
    height, width = camera_depth_tensor.shape
    device = camera_depth_tensor.device
    depth_buffer = camera_depth_tensor.to(device)
    rgb_buffer = camera_rgb_tensor.to(device)

    vinv = torch.tensor(camera_view_matrix_inv).to(device).float()
    proj = torch.tensor(camera_proj_matrix).to(device)
    fu = 2 / proj[0, 0]
    fv = 2 / proj[1, 1]

    camera_u = torch.arange(0, width, device=device)
    camera_v = torch.arange(0, height, device=device)

    v, u = torch.meshgrid(
        camera_v, camera_u)

    centerU = width / 2
    centerV = height / 2

    Z = depth_buffer
    X = -(u - centerU) / width * Z * fu
    Y = (v - centerV) / height * Z * fv

    R = rgb_buffer[..., 0].view(-1)
    G = rgb_buffer[..., 1].view(-1)
    B = rgb_buffer[..., 2].view(-1)

    Z = Z.view(-1)
    X = X.view(-1)
    Y = Y.view(-1)

    position = torch.vstack((X, Y, Z, torch.ones(len(X), device=device), R, G, B))
    position = position.permute(1, 0)
    position[:, 0:4] = position[:, 0:4] @ vinv
    points = torch.cat((position[:, 0:3], position[:, 4:8]), dim=1)

    return points


def remove_outliers(points):

    if len(points) > 100000:
        indices = random.sample(range(len(points)), 100000)
        points = points[indices]

    xyzs = points[:, :3]

    centroid = xyzs.mean(axis=0)
    distances_to_centroid = np.linalg.norm(xyzs - centroid, axis=1)
    weights = 1 + distances_to_centroid / distances_to_centroid.max()
    weighted_xyzs = xyzs * weights[:, np.newaxis]

    # DBSCAN
    try:
        clustering = DBSCAN(eps=0.05, min_samples=6, n_jobs=-1).fit(weighted_xyzs)
    except ValueError:
        print("DBSCAN failed, return all points")
        return points

    labels = clustering.labels_
    total_points = len(labels)
    num_clusters = labels.max() + 1
    cluster_sizes = [(i, np.sum(labels == i)) for i in range(num_clusters)]
    threshold = total_points * 0.05
    valid_clusters = [i for i, size in cluster_sizes if size > threshold]

    indices = np.where(np.isin(labels, valid_clusters))[0]
    filtered_points = points[indices]

    if len(filtered_points) < 0.8 * len(points):
        filtered_points = points

    return filtered_points
