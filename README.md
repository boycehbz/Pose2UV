# \[TIP2022\] Pose2UV: Single-shot Multi-person Mesh Recovery with Deep UV Prior

The code for TIP 2022 paper "Pose2UV: Single-shot Multi-person Mesh Recovery with Deep UV Prior"<br>

[Buzhen Huang](http://www.buzhenhuang.com/), [Tianshu Zhang](https://scholar.google.com/citations?user=S5M_CncAAAAJ&hl=zh-CN&oi=ao), [Yangang Wang](https://www.yangangwang.com/)<br>
\[[Project](https://www.yangangwang.com/papers/HBZ-Pose2UV-2022-06.html)\] \[[Paper](https://ieeexplore.ieee.org/document/9817035)\] \[[Dataset](https://github.com/boycehbz/3DMPB-dataset)\]

![figure](/assets/pose2uv_pipeline.png)

## Installation 
Create conda environment and install dependencies.
```
conda create -n pose2uv python=3.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu111 # install pytorch

cd utils/nms
python setup_linux.py install
```
Due to the licenses, please download SMPL model file [here](http://smplify.is.tuebingen.mpg.de/).

Download trained models from [here](https://pan.baidu.com/s/1I3o_Qf12ZS0N65uVZAx-KQ?pwd=43j7).


## Demo
```
python demo.py --config cfg_files/demo.yaml
```

## Train
```
python main.py --config cfg_files/train.yaml
```

## Test
```
python main.py --config cfg_files/test.yaml
```

## Eval
```
python eval.py --config cfg_files/eval.yaml
```

## Pose-Mask Module
```
python main.py --config cfg_files/poseseg.yaml
```

## Dataset
[3DMPB](https://github.com/boycehbz/3DMPB-dataset) is a multi-person dataset in the outdoor sport field with human interaction occlusion and image truncation. This dataset provides annotations including bounding-box, human 2D pose, SMPL model annotations, instance mask and camera parameters.

## Citation
If you find this code or dataset useful for your research, please consider citing the paper.
```
@article{huang2022pose2uv,
  title={Pose2UV: Single-shot Multi-person Mesh Recovery with Deep UV Prior},
  author={Huang, Buzhen and Zhang, Tianshu and Wang, Yangang},
  journal={IEEE Transactions on Image Processing},
  year={2022},
  volume={31},
  pages={4679-4692}
}
```
