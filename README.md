[![Build Status](http://ci.mxnet.io/job/gluon-cv/job/master/badge/icon)](http://ci.mxnet.io/job/gluon-cv/job/master/)
[![GitHub license](http://dmlc.github.io/img/apache2.svg)](./LICENSE)
[![Code Coverage](http://gluon-cv.mxnet.io/coverage.svg?)](http://gluon-cv.mxnet.io/coverage.svg)
[![PyPI](https://img.shields.io/pypi/v/gluoncv.svg)](https://pypi.python.org/pypi/gluoncv)

| [Installation](http://gluon-cv.mxnet.io) | [Documentation](http://gluon-cv.mxnet.io) | [Tutorials](http://gluon-cv.mxnet.io) |


# Cascade R-CNN 
## forked from [gluon-cv](https://github.com/zhreshold/gluon-cv)
edited by Yuerong Chen and Ning Li

## Benchmarking
1. PASCAL VOC 2007 (Train/Test: 2007+2012trainval/2007test, shorter size: 600)

model            | #GPUs | bs | lr | wd | epoch | decay | AP50  |  AP75  |  AP  | [Reference](https://github.com/zhaoweicai/cascade-rcnn) |
-----------------|--------|-----|--------|------|------|-------|-------|-------|-------|--------
VGG-RPN-baseline(pruned)| 2 | 1    |1e-3|5e-4|  20 |  14  | 75    | wait  | wait  | 42.9
VGG-RPN-cascade(pruned) | 2 | 1    |1e-3|5e-4|  20 |  14  | 76.20 | 57.79 | 53.05 | 51.2
RESNET50-RPN-cascade    | 8 | 1    |4e-3|5e-4|  20 |  14  | 79.71 | 60.00 | wait  | None
Res50-RFCN-Cascade      | 8 | 1    |2e-3|1e-4|  20 |  14  | 76.87 | 55.80 | 50.95 | 51.8 
Res101-RFCN-Cascade     | 8 | 1    |2e-3|1e-4|  20 |  14  | 79.12 | 59.45 | 54.79 | 54.2

## Developing Environment
**MXNet 1.3.0**

The repo is based on the 1.3.x version MXNet. You will probably need to go to the MXNet official github repo and compile it.

If your MXNet installed by using pip install(at least on August). It should be of MXNet version 1.2.x, which is too old for this repo.  You might need to solve the related problems because of the version mismatch.



## Installation

1. Clone the cascade_rcnn_gluon repository, and we'll call the directory that you cloned cascade_rcnn_gluon into CASCADE_ROOT

    ```Shell
	git clone https://github.com/zzdang/cascade_rcnn_gluon.git
    ```

2. Build cascade_rcnn_gluon

    ```Shell
	cd $CASCADE_ROOT/
	# Follow the gluon-cv installation instructions here:
	#   https://gluon-cv.mxnet.io/
	python setup.py install
    ```

## Training Cascade-RCNN

1. Get the training data
    ```Shell
    # This will download the pascal_voc dataset
	python scripts/datasets/pascal_voc.py
    ```

2. Download the pretrained models on ImageNet. For VGG-Net(we called vgg16_pruned), the FC layers are pruned and 2048 units per FC layer are remained. In addition, the two FC layers are copied three times for Cascade R-CNN training.

    ```Shell
    # Download pre-trained model(You can download it use dropbox or baiduyun link)
    -[dropbox link](https://www.dropbox.com/s/tjgcwqgber2tlxh/VGG_16_fc2048_prune.params?dl=0)
    -[baiduyun link](https://pan.baidu.com/s/1RgG33zy40ssdWHdhPx0-Kg) passwd: b7ev  

    # copy the pre-trained models to $CASCADE_ROOT/models/
    cp /PATH/TO/DOWNLOAD/MODEL $CASCADE_ROOT/models/
    ```

	```Shell
	# convert the vgg16_pruned pretained params to vgg 16_pruned_cascade params
	python load_params.py
	```

3. training for Cascade-RCNN

    ```Shell
    # training for pruned VGG16 
	python scripts/detection/cascade_rcnn/train_cascade_rcnn.py --network vgg16_pruned
    ```

4. Testing Demo

    ```Shell
    # testing for pruned VGG16 (VOC dataset)
	python scripts/detection/cascade_rcnn/demo_cascade_rcnn.py --network cascade_rcnn_vgg16_pruned_voc --pretrained /PATH/TO/TRAINED/MODEL
    ```

## Training Cascade-RFCN

1. training for Cascade-RFCN

    ```Shell
    # training for resnet101_v1b
    python scripts/detection/cascade_rcnn/train_cascade_rfcn.py --network resnet101_v1b --lr 0.002 --wd 0.0001 --save-prefix ./models/ --gpus 0
    ```

## Examples

- [Image Classification](http://gluon-cv.mxnet.io/build/examples_classification/index.html)

- [Object Detection](http://gluon-cv.mxnet.io/build/examples_detection/index.html)

- [Semantic Segmentation](http://gluon-cv.mxnet.io/build/examples_segmentation/index.html)


## To Do List

- [x] Train Cascade-RCNN with VGG16 and pruned vgg16 backbone
- [ ] Train Cascade-RCNN with pruned VGG16 backbone(2 batch pre gpu)
- [ ] Train faster-RCNN and Cascade-RCNN with resnet50 backbone 

