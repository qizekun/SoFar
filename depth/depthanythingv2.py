import torch
import numpy as np
from depth.depth_anything_v2.dpt import DepthAnythingV2
from depth.utils import get_intrinsic, reconstruct_pcd, gray_to_colormap

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_configs = {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
dataset = 'hypersim'  # 'hypersim' for indoor model, 'vkitti' for outdoor model
DEPTH_ANYTHING_CHECKPOINT_PATH = f'checkpoints/depth_anything_v2_metric_{dataset}_{model_configs["encoder"]}.pth'


def get_model():
    max_depth = 20  # 20 for indoor model, 80 for outdoor model
    model = DepthAnythingV2(**{**model_configs, 'max_depth': max_depth})
    model.load_state_dict(torch.load(DEPTH_ANYTHING_CHECKPOINT_PATH, map_location='cpu'))
    model.to(DEVICE).eval()
    return model


def depth_estimation(img):
    fx, fy = get_intrinsic(img)
    img = np.array(img)
    intrinsic = [fx, fy, img.shape[1] / 2, img.shape[0] / 2]

    model = get_model()
    pred_depth = model.infer_image(img)  # HxW depth map in meters in numpy
    pcd = reconstruct_pcd(pred_depth, intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3])

    pred_depth[pred_depth < 0] = 0
    img = gray_to_colormap(pred_depth)
    img.save("depth_estimation.jpg")

    return pred_depth, img, pcd
