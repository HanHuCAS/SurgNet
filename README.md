# SurgNet
This is the repository of the following work:

Jiachen Chen, Mengyang Li, Hu Han, Zhiming Zhao, and Xilin Chen. “SurgNet: Self-supervised Pretraining with Semantic Consistency for Vessel and Instrument Segmentation in Surgical Images”, IEEE Transactions on Medical Imaging, 2023 (Accepted). [[paper]](https://ieeexplore.ieee.org/abstract/document/10354412)

## 1. Pretrained Model and Inference Code:

### Pretrained Models
(1) [PTSurgNet](https://drive.google.com/file/d/1pgjtQLtmHBWXzMX7b595loDS9jDHQMsR/view?usp=sharing): This is the pretrained SurgNet model with 3M unlabeled images of IVIS-U. You can download the model and put it into ".\pretrain\\", and finetune it on a different dataset.

(2) [FTEV17SurgNet](https://drive.google.com/file/d/1eyrqZMRdyDRZ-2HnwFvBKBLZ-ZP-uyJA/view?usp=sharing): This is a finetuned SurgNet model on the EndoVisSub2017 training data. You can download the mode and put it into ".\work_dirs\\tmp\\", and test it on the testing set directly.

### Suggested Environment
```
conda create -n segenv python=3.8 pytorch==1.9.0 cudatoolkit=11.1 torchvision -c pytorch -y
conda activate segenv
pip install mmcv-full==1.3.0 
pip install mmseg==0.11.0
pip install scipy timm==0.3.2
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

### Data Prepare
Follow the guide in [mmseg](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/dataset_prepare.md) to prepare the Pascal-VOC dataset.

|----ImageSets

​		|----Segmentation

​				|----train.txt

​				|----val.txt

|----JPEGImages

​		|----10_frame000.jpg

​		|----10_frame001.jpg

​		|---- ......

|----SegmentationClassPNG

​		|----10_frame000.png

​		|----10_frame001.png

​		|---- ......



### Finetune Command
```
bash dist_train.sh ${CONFIG_FILE} ${GPUS}
# e.g., bash dist_train.sh configs/upernet/EndoVisSub2017_upernet.py 4
```

### Inference Command

```
Get quantitative results (mIoU):
bash dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE}  --eval mIoU
# e.g., python test.py configs/upernet/EndoVisSub2017_upernet.py work_dirs/tmp/finetune_pvt.pth  --eval mIoU

Get qualitative results:
bash dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE}  --show-dir=${SAVE_FILE}
# e.g., python test.py configs/upernet/EndoVisSub2017_upernet.py work_dirs/tmp/finetune_pvt.pth  --show-dir=work_dirs/val 
```

### Acknowledgement

The authors would like to thank Jiyang Tang for clearing up the code for this repository.
   
## 2. Application of the IVIS Intraoperative Vessel and Instrument Segmentation (IVIS) dataset: 
Individuals interested in accessing the IVIS dataset are required to consult the End User License Agreement [EULA](https://github.com/HanHuCAS/SurgNet/raw/main/IVIS%20Database%20EULA(1.1).docx) and submit an application. Upon approval of the completed and signed EULA by our Institutional Review Board (IRB), the applicant will receive a download link to the dataset.

## 3. References
[1] Xiang Li, Wenhai Wang, Lingfeng Yang, Jian Yang. Uniform Masking: Enabling MAE Pre-training for Pyramid-based Vision Transformers with Locality, arXiv:2205.10063, 2022. https://github.com/implus/UM-MAE

[2] MMSegmentation: https://github.com/open-mmlab/mmsegmentation
