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

model     | #GPUs | bs | lr | epoch | decay | AP50 
---------|--------|-----|--------|------|------|-----
VGG-RPN-baseline     | 2 | 2    |1e-3|  20  |  10  | 72.3  
VGG-RPN-cascade      | 2 | 2    |1e-3|  20  |  14  | 75.3

## Training Cascade-RCNN

1. training for faster rcnn(vgg16)
    ```Shell
	python scripts/detection/faster_rcnn/train_faster_rcnn.py --network vgg16
    ```
2. training for cascade rcnn(vgg16)
    ```Shell
	python scripts/detection/cascade_rcnn/train_cascade_rcnn.py --network vgg16
    ```


## Examples

- [Image Classification](http://gluon-cv.mxnet.io/build/examples_classification/index.html)

- [Object Detection](http://gluon-cv.mxnet.io/build/examples_detection/index.html)

- [Semantic Segmentation](http://gluon-cv.mxnet.io/build/examples_segmentation/index.html)


## To Do List

- [x] Add VGG to faster-RCNN
- [ ] Train Cascade-RCNN with VGG16 backbone(2 batch pre gpu)
- [ ] Add VGG prune to faster-RCNN and cascade_rcnn
- [ ] Train faster-RCNN and Cascade-RCNN with resnet50 backbone 

