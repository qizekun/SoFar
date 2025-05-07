# SoFar: Language-Grounded Orientation Bridges Spatial Reasoning and Object Manipulation

## Open6DOR V2
Download the [open6dor_v2.zip](https://huggingface.co/datasets/qizekun/Open6DOR_V2/blob/main/open6dor_v2.zip) and extract it to `./datasets/open6dor_v2/`.
The overall directory structure should be:
```
│SoFar/datasets/open6dor_v2/
├── task_refine_pos/
│   ├── behind
│   └── ...
├── task_refine_rot/
│   └── rot_ins
└── task_refine_6dof/
    ├── behind
    │   ├── Place_the_apple_behind_the_bottle_on_the_table.__upright
    │   └── ...
    └── ...
```
| Data file name                                                                                   |     Size |
|--------------------------------------------------------------------------------------------------|---------:|
| [open6dor_v2.zip](https://huggingface.co/datasets/qizekun/Open6DOR_V2/blob/main/open6dor_v2.zip) |  1.89 GB |
| task_refine_pos                                                                                  | 20.41 GB |
| task_refine_rot                                                                                  | 12.84 GB |
| task_refine_6dof                                                                                 | 21.99 GB |


## 6-DoF SpatialBench
Download the [images.zip](https://huggingface.co/datasets/qizekun/6DoF-SpatialBench/blob/main/images.zip) and extract it to `./datasets/6dof_spatialbench/`.
The overall directory structure should be:
```
│SoFar/datasets/6dof_spatialbench/
├── spatial_data.json
└── images/
    ├── 0.png
    └── ...
```
| Data file name                                                                               |     Size |
|----------------------------------------------------------------------------------------------|---------:|
| [images.zip](https://huggingface.co/datasets/qizekun/6DoF-SpatialBench/blob/main/images.zip) | 531.7 MB |
| [spatial_data.json](./6dof_spatialbench/spatial_data.json)                                   |   0.8 KB |


## OrienText300K
Download the [OrienText300K.json](https://huggingface.co/datasets/qizekun/OrienText300K/blob/main/OrienText300K.json) or [OrienText300K_strict.json](https://huggingface.co/datasets/qizekun/OrienText300K/blob/main/OrienText300K_strict.json) and place it in `./orientation/data/SO/train.json`.
The strict version is a subset of the original dataset with more strict filtering, including smooth and texture surface. 
We use Cap3D as our point clouds data source. Cap3D_pcs data needs to be obtained in pt form from the [Cap3D repository](https://huggingface.co/datasets/tiange/Cap3D/tree/main).
The overall directory structure should be:
```
│orientation/data/SO/
├── train.json <---- OrienText300K.json
└── pcs <---- cap3d_pcs
    ├── 00000054c36d44a2a483bdbff31d8edf.pt
    └── ...
```
| Data file name                                                                                                         |     Size |
|------------------------------------------------------------------------------------------------------------------------|---------:|
| [OrienText300K.json](https://huggingface.co/datasets/qizekun/OrienText300K/blob/main/OrienText300K.json)               |   189 MB |
| [OrienText300K_strict.json](https://huggingface.co/datasets/qizekun/OrienText300K/blob/main/OrienText300K_strict.json) |   140 MB |
| [cap3d_pcs](https://huggingface.co/datasets/tiange/Cap3D/tree/main/PointCloud_pt_zips)                                 | 173.8 GB |