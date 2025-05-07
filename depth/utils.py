import os
import numpy as np
from PIL import ExifTags
from depth.metric_3d_v2.utils.transform import gray_to_colormap
from depth.metric_3d_v2.utils.unproj_pcd import reconstruct_pcd, save_point_cloud, ply_to_obj


def get_intrinsic(img):
    try:
        exif = img.getexif()
        exif.update(exif.get_ifd(ExifTags.IFD.Exif))
    except:
        exif = {}
    sensor_width = exif.get(ExifTags.Base.FocalPlaneYResolution, None)
    sensor_height = exif.get(ExifTags.Base.FocalPlaneXResolution, None)
    focal_length = exif.get(ExifTags.Base.FocalLength, None)

    w, h = img.size
    sensor_width = w / sensor_width * 25.4 if sensor_width is not None else None
    sensor_height = h / sensor_height * 25.4 if sensor_height is not None else None
    focal_length = focal_length * 1.0 if focal_length is not None else None

    if focal_length is None:
        focal_length = 6.76
    elif sensor_width is None and sensor_height is None:
        sensor_width = 16
        sensor_height = h / w * sensor_width

    # calculate focal length in pixels
    w, h = img.size
    fx = w / sensor_width * focal_length if sensor_width is not None else 1000
    fy = h / sensor_height * focal_length if sensor_height is not None else 1000

    return fx, fy


def depth2pcd(depth, intrinsic, transform_matrix=None):
    pcd_camera = reconstruct_pcd(depth, intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3])
    if transform_matrix is not None:
        pcd_base = transform_point_cloud(pcd_camera, np.array(transform_matrix))
    else:
        pcd_base = None
    return pcd_camera[:, :, :3], pcd_base


def transform_point_cloud(point_cloud, transform_matrix):
    H, W, _ = point_cloud.shape

    point_cloud_homogeneous = np.ones((H, W, 4))
    point_cloud_homogeneous[:, :, :3] = point_cloud

    H, W = point_cloud_homogeneous.shape[0], point_cloud_homogeneous.shape[1]
    points_reshaped = point_cloud_homogeneous.reshape(-1, 4)

    transformed_points = np.dot(transform_matrix, points_reshaped.T)
    transformed_points = transformed_points[:3, :]  # (3, 512*640)
    transformed_points_reshaped = transformed_points.reshape(3, H, W)

    return np.transpose(transformed_points_reshaped, (1, 2, 0))


def transform_point_cloud_nohw(point_cloud, transform_matrix):
    num, _ = point_cloud.shape

    point_cloud_homogeneous = np.ones((num, 4))
    point_cloud_homogeneous[:, :3] = point_cloud

    points_reshaped = point_cloud_homogeneous.reshape(-1, 4)

    transformed_points = np.dot(transform_matrix, points_reshaped.T)
    transformed_points = transformed_points[:3, :]  # (3, 512*640)

    return np.transpose(transformed_points, (1, 0))


def inverse_transform_point_cloud(point_cloud, transform_matrix):
    num, _ = point_cloud.shape

    point_cloud_homogeneous = np.ones((num, 4))
    point_cloud_homogeneous[:, :3] = point_cloud

    points_reshaped = point_cloud_homogeneous.reshape(-1, 4)

    transform_matrix_inv = np.linalg.inv(transform_matrix)

    transformed_points = np.dot(transform_matrix_inv, points_reshaped.T)
    transformed_points = transformed_points[:3, :]  # (3, N)
    return np.transpose(transformed_points, (1, 0))


def transform_obj_pts(points, extrinsic_matrix):
    n = points.shape[0]
    homogeneous_points = np.hstack((points, np.ones((n, 1))))  # (n, 4)

    transformed_points = homogeneous_points @ extrinsic_matrix.T  # (n, 4)

    world_points = transformed_points[:, :3] / transformed_points[:, 3:]

    return world_points
