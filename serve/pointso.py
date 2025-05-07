import yaml
import torch
import random
import numpy as np
from easydict import EasyDict
from orientation.datasets.utils import pc_norm
from orientation.models.PointSO import PointSO

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CFG_PATH = "orientation/cfgs/train/small.yaml"
CHECKPOINT_PATH = "checkpoints/small.pth"

# for Open6DOR tasks
# CFG_PATH = "orientation/cfgs/train/base.yaml"
# CHECKPOINT_PATH = "checkpoints/base_finetune.pth"


def get_model():
    with open(CFG_PATH, 'r') as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader)).model
    model = PointSO(config)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE)["base_model"], strict=True)
    model.to(DEVICE).eval()
    return model


def pred_orientation(model, pcd, instruction):
    assert pcd.shape[1] == 6
    pcd_list = []
    vote_num = 12
    for i in range(vote_num):
        if pcd.shape[0] < 10000:
            idx = random.choices(range(pcd.shape[0]), k=10000)
        else:
            idx = random.sample(range(pcd.shape[0]), k=10000)
        temp = pcd[idx]
        temp[:, :3] = pc_norm(temp[:, :3])
        pcd_list.append(temp)
    pcd = np.array(pcd_list)
    n = len(instruction)
    with torch.no_grad():
        pcd = torch.from_numpy(pcd).float().repeat(n, 1, 1).to(DEVICE)
        instruction = instruction * vote_num
        pred = model(pcd, instruction)
        pred = pred.cpu().numpy()

        pred = pred.reshape(n, vote_num, 3).mean(axis=1).reshape(n, 3)
        norms = np.linalg.norm(pred, axis=-1, keepdims=True)
        normalized_pred = pred / (norms + 1e-8)
        normalized_directions_list = normalized_pred.tolist()
    return normalized_directions_list


if __name__ == "__main__":
    model = get_model()
    pcd = np.load("output/picked_obj.npy")
    instruction = ["open"]
    orientation = pred_orientation(model, pcd, instruction)
    print(orientation)
