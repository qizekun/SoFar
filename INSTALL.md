## Installation
The code requires `python>=3.8`, as well as `pytorch>=2.0.1` and `torchvision>=0.15.0`.
Recommendation setting for the environment variable:
```bash
conda create -n sofar python=3.12 -y
conda activate sofar
export CUDA_HOME=/path/to/cuda-12.1/
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
# if use Metric3D V2
pip install xformers==0.0.27 --index-url https://download.pytorch.org/whl/cu121
```


Install SoFar package:
```bash
git clone https://github.com/qizekun/SoFar.git
cd SoFar
pip install -e .
pip install -e segmentation/SAM
```

Optional, install other packages:
```bash
# for grounding DINO
pip install --no-build-isolation -e segmentation/GroundingDINO
# for yolo-world
pip install inference[yolo-world]==0.9.13
# for training SoFar-LLaVA
pip install -e sofar_llava
pip install flash-attn --no-build-isolation
# for qwen inference
pip install qwen-vl-utils[decord]==0.0.8
pip install flash-attn --no-build-isolation
```

Download checkpoints:

Mainly we are using Florance-2 + Segment Anything for open-vocabulary object segmentation.
Note that Florance-2 performs poorly on predicting small objects and multi-objects, and Grounding DINO has better effects on these special tasks.
```bash
cd checkpoints
# Florence-2
huggingface-cli download microsoft/Florence-2-base
# Segment Anything
wget -c https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# PointSO
wget -c https://huggingface.co/qizekun/PointSO/resolve/main/small.pth
wget -c https://huggingface.co/qizekun/PointSO/resolve/main/base_finetune.pth
```
Optional, download other checkpoints:
```bash
# Metric3D V2
wget -c https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_large_800k.pth
# Grounding DINO
wget -c https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
# Depth Anything V2
wget -c https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
```
If connect huggingface has problems, use mirror:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```
