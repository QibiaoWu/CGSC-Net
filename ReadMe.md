# Spherical Modulated Convolution with Cosine Supervision for Airway Segmentation

The full code will be published after the paper has been accepted by the journal.

## Introduction
![Architecture of CGSC-Net]([fig/fig2.png](https://github.com/QibiaoWu/CGSC-Net/blob/main/fig/fig2.png))
The network consists of two parallel U-shaped networks for airway segmentation and orientation field learning,respectively. The cosine information learned by the CLM is used to guide the deformation of SMConv in the ITFFM.

![DAConv and SMConv]([fig/fig1.png](https://github.com/QibiaoWu/CGSC-Net/blob/main/fig/fig1.png))
(a) DAConv requires stacking three direction-aware convolutions to form a spherical receptive field, and each convolution needs to learn 4 angles. 
(b) Taking SMConv with kernel size 5 as an example, only one SMConv and learning 2 angles are needed to form a spherical receptive field.

## Setup environment
```bash
conda create --name CGSCNet python==3.8
conda activate CGSCNet
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Install SMConv
```bash
pip install SMConv-1.0.0-cp38-cp38-linux_x86_64.whl # Make sure you have CUDA 12.1 Toolkit installed
```

## Download our pretrained weight
checkpoint download link: https://drive.google.com/drive/folders/17-PHDG27s4fmZrGvqzISwE5oR-4Rbn5U?usp=drive_link
