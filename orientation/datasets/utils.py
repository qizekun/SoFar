import torch
import numpy as np
from plyfile import PlyData
from scipy.spatial import KDTree


def interpolate_point_cloud(points, k=2):
    N = points.shape[0]
    tree = KDTree(points[:, :3])

    dists, indices = tree.query(points[:, :3], k=k)
    alphas = np.random.rand(N, k - 1, 1)

    neighbor_points = points[indices[:, 1:], :3]
    neighbor_attrs = points[indices[:, 1:], 3:]

    original_points = points[:, :3].reshape(N, 1, 3)
    original_attrs = points[:, 3:].reshape(N, 1, -1)

    new_points = alphas * original_points + (1 - alphas) * neighbor_points
    new_attrs = alphas * original_attrs + (1 - alphas) * neighbor_attrs

    new_points = np.concatenate([new_points, new_attrs], axis=2).reshape(-1, points.shape[1])

    return np.vstack((points, new_points))


def filter_point_cloud(point_cloud, angle_threshold_degrees=90, fix=False):
    # Extract XYZ coordinates from the point cloud
    xyz = point_cloud[:, :3]

    # Center the point cloud at the origin and normalize
    center = np.mean(xyz, axis=0)
    normalized_xyz = (xyz - center) / np.linalg.norm(xyz - center, axis=1)[:, np.newaxis]

    # Randomly select a point within half a unit sphere
    if fix:
        phi = np.pi / 4
        theta = np.pi / 2
    else:
        phi = np.random.uniform(0, np.pi / 2)
        theta = np.random.uniform(0, np.pi)

    x = np.sin(phi) * np.sin(theta)
    y = np.sin(phi) * np.cos(theta)
    z = np.cos(phi)

    random_point = np.array([x, y, z])

    # Calculate angles between the random point and all points in the normalized point cloud
    dot_product = np.dot(normalized_xyz, random_point.T)
    clipped_dot_product = np.clip(dot_product, -1.0, 1.0)
    angles = np.arccos(clipped_dot_product)

    # Filter points based on the angle threshold
    filtered_indices = np.where(np.degrees(angles) < angle_threshold_degrees)[0]

    # Keep only the selected points
    selected_points = point_cloud[filtered_indices]

    return selected_points


def occlusion(point_cloud, num_points=10000, angle_threshold_degrees=90, fix=False):
    filtered_point_cloud = filter_point_cloud(point_cloud, angle_threshold_degrees, fix)
    filtered_num = filtered_point_cloud.shape[0]
    k = int(num_points / filtered_num) + 1
    if 1 < k < 5:
        point_cloud = interpolate_point_cloud(filtered_point_cloud, k)
    return point_cloud


def rotation(xyz, rotation_angle):
    x, y, z = rotation_angle
    x, y, z = int(x), int(y), int(z)
    x_rad, y_rad, z_rad = np.radians(x), np.radians(y), np.radians(z)

    rot_x = np.array([[1, 0, 0], [0, np.cos(x_rad), -np.sin(x_rad)], [0, np.sin(x_rad), np.cos(x_rad)]])
    rot_y = np.array([[np.cos(y_rad), 0, np.sin(y_rad)], [0, 1, 0], [-np.sin(y_rad), 0, np.cos(y_rad)]])
    rot_z = np.array([[np.cos(z_rad), -np.sin(z_rad), 0], [np.sin(z_rad), np.cos(z_rad), 0], [0, 0, 1]])

    rot_matrix = np.dot(np.dot(rot_z, rot_y), rot_x)
    xyz = np.matmul(xyz, rot_matrix)
    return xyz


def load_pts(path, separator=',', verbose=False):
    extension = path.split('.')[-1]
    if extension == 'npy':
        pcl = np.load(path, allow_pickle=True)
    elif extension == 'npz':
        pcl = np.load(path)
        pcl = pcl['pred']
    elif extension == 'ply':
        ply = PlyData.read(path)
        vertex = ply['vertex']
        (x, y, z) = (vertex[t] for t in ('x', 'y', 'z'))
        pcl = np.column_stack((x, y, z))
        if len(vertex.properties) == 6:
            (r, g, b) = (vertex[t] for t in ('red', 'green', 'blue'))
            pcl = np.column_stack((pcl, r, g, b))
            pcl[:, 3:] = pcl[:, 3:] / 255 if pcl[:, 3:].max() > 1.0 else pcl
    elif extension == 'txt':
        f = open(path, 'r')
        line = f.readline()
        data = []
        while line:
            x, y, z = line.split(separator)[:3]
            data.append([float(x), float(y), float(z)])
            line = f.readline()
        f.close()
        pcl = np.array(data)
    elif extension == 'pth' or extension == 'pt':
        pcl = torch.load(path, map_location='cpu', weights_only=True)
        pcl = pcl.detach().numpy()
    else:
        print('unsupported file format.')
        raise FileNotFoundError

    if len(pcl.shape) == 3:
        pcl = pcl.reshape(-1, pcl.shape[-1])

    if pcl.shape[0] == 3 or pcl.shape[0] == 6:
        pcl = pcl.T

    if verbose:
        print(f'point cloud shape: {pcl.shape}')
    assert pcl.shape[-1] == 3 or pcl.shape[-1] == 6

    return pcl


def pc_norm(pc):
    """ pc: NxC, return NxC """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    if m < 1e-6:
        pc = np.zeros_like(pc)
    else:
        pc = pc / m
    return pc


def random_sample(pc, num):
    ori_num = pc.shape[0]
    if ori_num > num:
        permutation = np.arange(ori_num)
        np.random.shuffle(permutation)
        pc = pc[permutation[:num]]
    return pc


def process_pts(pts, data_args):
    pts = random_sample(pts, data_args.sample_points_num)
    pts[:, :3] = pc_norm(pts[:, :3])
    if data_args.with_color:
        if pts[:, 3:].max() > 1.0 + 1e-2:
            pts[:, 3:] = pts[:, 3:] / 255.0
    else:
        pts = pts[:, :3]
    pts = torch.from_numpy(pts).float()
    return pts
