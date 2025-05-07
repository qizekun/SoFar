import torch
import numpy as np
from depth.monocular_geometry.model import MoGeModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model():
    model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(DEVICE)
    return model


def depth_estimation(img, moge_model, output_folder="output"):
    # Read the input image and convert to tensor (3, H, W) and normalize to [0, 1]
    img = np.array(img)
    img = torch.tensor(img / 255, dtype=torch.float32, device=DEVICE).permute(2, 0, 1)

    output = moge_model.infer(img)
    # (H, W, 3) scale-invariant point map in OpenCV camera coordinate system (x right, y down, z forward)
    points = output["points"].cpu().numpy()
    depth = output["depth"].cpu().numpy() # (H, W) scale-invariant depth map
    mask = output["mask"].cpu().numpy()   # (H, W) a binary mask for valid pixels.
    intrinsics = output["intrinsics"].cpu().numpy()  # (3, 3) normalized camera intrinsics
    return depth, intrinsics, points
