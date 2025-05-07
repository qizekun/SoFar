import os
import torch
from mmengine import Config

import numpy as np
from PIL import Image
from depth.metric_3d_v2.utils.running import load_ckpt
from depth.utils import get_intrinsic, reconstruct_pcd, gray_to_colormap
from depth.metric_3d_v2.model.monodepth_model import get_configured_monodepth_model
from depth.metric_3d_v2.utils.do_test import transform_test_data_scalecano, get_prediction


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CFG_PATH = "depth/metric_3d_v2/configs/HourglassDecoder/vit.raft5.large.py"
CHECKPOINT_PATH = "checkpoints/metric_depth_vit_large_800k.pth"


def get_model():
    cfg = Config.fromfile(CFG_PATH)
    model = get_configured_monodepth_model(cfg, )
    model, _,  _, _ = load_ckpt(CHECKPOINT_PATH, model, strict_match=False)
    model.to(DEVICE).eval()
    return cfg, model


def predict_depth_normal(img, model, cfg, fx=1000.0, fy=1000.0):

    img = np.array(img)
    intrinsic = [fx, fy, img.shape[1] / 2, img.shape[0] / 2]
    rgb_input, cam_models_stacks, pad, label_scale_factor = transform_test_data_scalecano(img, intrinsic,
                                                                                          cfg.data_basic)

    with torch.no_grad():
        pred_depth, pred_depth_scale, scale, output, confidence = get_prediction(
            model=model,
            input=rgb_input,
            cam_model=cam_models_stacks,
            pad_info=pad,
            scale_info=label_scale_factor,
            gt_depth=None,
            normalize_scale=cfg.data_basic.depth_range[1],
            ori_shape=[img.shape[0], img.shape[1]],
        )

    pred_depth = pred_depth.squeeze().cpu().numpy()
    pred_depth[pred_depth < 0] = 0
    pred_color = gray_to_colormap(pred_depth)

    img = Image.fromarray(pred_color)
    pcd = reconstruct_pcd(pred_depth, intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3])

    return pred_depth, img, pcd


def depth_estimation(img, metriced_model, output_folder="output"):
    cfg, model = metriced_model
    fx, fy = get_intrinsic(img)
    depth, depth_img, pcd = predict_depth_normal(img, model, cfg, fx, fy)
    depth_img.save(os.path.join(output_folder, "depth_estimation.jpg"))
    return depth, depth_img, pcd
