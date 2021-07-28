# Delving into Localization Errors for Monocular 3D Detection

By [Xinzhu Ma](https://scholar.google.com/citations?user=8PuKa_8AAAAJ), Yinmin Zhang, [Dan Xu](https://www.danxurgb.net/), [Dongzhan Zhou](https://scholar.google.com/citations?user=Ox6SxpoAAAAJ), [Shuai Yi](https://scholar.google.com/citations?user=afbbNmwAAAAJ), [Haojie Li](https://scholar.google.com/citations?user=pMnlgVMAAAAJ), [Wanli Ouyang](https://wlouyang.github.io/).


## Introduction

This repository is an official implementation of the paper ['Delving into Localization Errors for Monocular 3D Detection'](https://arxiv.org/abs/2103.16237). In this work, by intensive diagnosis experiments, we quantify the impact introduced by each sub-task and found the ‘localization error’ is the vital factor in restricting monocular 3D detection. Besides, we also investigate the underlying reasons behind localization errors, analyze the issues they might bring, and propose three strategies. 

<img src="resources/example.jpg" alt="vis" style="zoom:50%;" />




## Usage

### Installation
This repo is tested on our local environment (python=3.6, cuda=9.0, pytorch=1.1), and we recommend you to use anaconda to create a vitural environment:

```bash
conda create -n monodle python=3.6
```
Then, activate the environment:
```bash
conda activate monodle
```

Install  Install PyTorch:

```bash
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch
```

and other  requirements:
```bash
pip install -r requirements.txt
```

### Data Preparation
Please download [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and organize the data as follows:

```
#ROOT
  |data/
    |KITTI/
      |ImageSets/ [already provided in this repo]
      |object/			
        |training/
          |calib/
          |image_2/
          |label/
        |testing/
          |calib/
          |image_2/
```

### Training & Evaluation

Move to the workplace and train the network:

```sh
 cd #ROOT
 cd experiments/example
 python ../../tools/train_val.py --config kitti_example.yaml
```
The model will be evaluated automatically if the training completed. If you only want evaluate your trained model (or the provided [pretrained model](https://drive.google.com/file/d/1jaGdvu_XFn5woX0eJ5I2R6wIcBLVMJV6/view?usp=sharing)) , you can modify the test part configuration in the .yaml file and use the following command:

```sh
python ../../tools/train_val.py --config kitti_example.yaml --e
```

For ease of use, we also provide a pre-trained checkpoint, which can be used for evaluation directly. See the below table to check the performance.

|                   | AP40@Easy | AP40@Mod. | AP40@Hard |
| ----------------- | --------- | --------- | --------- |
| In original paper | 17.45     | 13.66     | 11.68     |
| In this repo      | 17.94     | 13.72     | 12.10     |

## Citation

If you find our work useful in your research, please consider citing:

```latex
@InProceedings{Ma_2021_CVPR,
author = {Ma, Xinzhu and Zhang, Yinmin, and Xu, Dan and Zhou, Dongzhan and Yi, Shuai and Li, Haojie and Ouyang, Wanli},
title = {Delving into Localization Errors for Monocular 3D Object Detection},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2021}}
```

## Acknowlegment

This repo benefits from the excellent work [CenterNet](https://github.com/xingyizhou/CenterNet). Please also consider citing it.

## License

This project is released under the MIT License.

## Contact

If you have any question about this project, please feel free to contact xinzhu.ma@sydney.edu.au.
