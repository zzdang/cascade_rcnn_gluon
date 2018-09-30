"""Faster RCNN Model."""
from __future__ import absolute_import

import os
import mxnet as mx
from mxnet import autograd
from mxnet.gluon import nn
from .rcnn_target import RCNNTargetSampler, RCNNTargetGenerator
from ..rcnn import RFCN_RCNN
from ..rpn import RPN

__all__ = ['rfcn', 'get_rfcn',
           'rfcn_resnet50_v2a_voc']


class rfcn(RFCN_RCNN):
    r"""Faster RCNN network.
    Parameters
    ----------
    features : gluon.HybridBlock
        Base feature extractor before feature pooling layer.
    top_features : gluon.HybridBlock
        Tail feature extractor after feature pooling layer.
    classes : iterable of str
        Names of categories, its length is ``num_class``.
    short : int, default is 600.
        Input image short side size.
    max_size : int, default is 1000.
        Maximum size of input image long side.
    train_patterns : str, default is None.
        Matching pattern for trainable parameters.
    nms_thresh : float, default is 0.3.
        Non-maximum suppression threshold. You can speficy < 0 or > 1 to disable NMS.
    nms_topk : int, default is 400
        Apply NMS to top k detection results, use -1 to disable so that every Detection
         result is used in NMS.
    post_nms : int, default is 100
        Only return top `post_nms` detection results, the rest is discarded. The number is
        based on COCO dataset which has maximum 100 objects per image. You can adjust this
        number if expecting more objects. You can use -1 to return all detections.
    roi_mode : str, default is align
        ROI pooling mode. Currently support 'pool' and 'align'.
    roi_size : tuple of int, length 2, default is (14, 14)
        (height, width) of the ROI region.
    stride : int, default is 16
        Feature map stride with respect to original image.
        This is usually the ratio between original image size and feature map size.
    clip : float, default is None
        Clip bounding box target to this value.
    rpn_channel : int, default is 1024
        Channel number used in RPN convolutional layers.
    base_size : int
        The width(and height) of reference anchor box.
    scales : iterable of float, default is (8, 16, 32)
        The areas of anchor boxes.
        We use the following form to compute the shapes of anchors:
        .. math::
            width_{anchor} = size_{base} \times scale \times \sqrt{ 1 / ratio}
            height_{anchor} = size_{base} \times scale \times \sqrt{ratio}
    ratios : iterable of float, default is (0.5, 1, 2)
        The aspect ratios of anchor boxes. We expect it to be a list or tuple.
    alloc_size : tuple of int
        Allocate size for the anchor boxes as (H, W).
        Usually we generate enough anchors for large feature map, e.g. 128x128.
        Later in inference we can have variable input sizes,
        at which time we can crop corresponding anchors from this large
        anchor map so we can skip re-generating anchors for each input.
    rpn_train_pre_nms : int, default is 12000
        Filter top proposals before NMS in training of RPN.
    rpn_train_post_nms : int, default is 2000
        Return top proposal results after NMS in training of RPN.
    rpn_test_pre_nms : int, default is 6000
        Filter top proposals before NMS in testing of RPN.
    rpn_test_post_nms : int, default is 300
        Return top proposal results after NMS in testing of RPN.
    rpn_nms_thresh : float, default is 0.7
        IOU threshold for NMS. It is used to remove overlapping proposals.
    train_pre_nms : int, default is 12000
        Filter top proposals before NMS in training.
    train_post_nms : int, default is 2000
        Return top proposal results after NMS in training.
    test_pre_nms : int, default is 6000
        Filter top proposals before NMS in testing.
    test_post_nms : int, default is 300
        Return top proposal results after NMS in testing.
    rpn_min_size : int, default is 16
        Proposals whose size is smaller than ``min_size`` will be discarded.
    num_sample : int, default is 128
        Number of samples for RCNN targets.
    pos_iou_thresh : float, default is 0.5
        Proposal whose IOU larger than ``pos_iou_thresh`` is regarded as positive samples.
    pos_ratio : float, default is 0.25
        ``pos_ratio`` defines how many positive samples (``pos_ratio * num_sample``) is
        to be sampled.
    additional_output : boolean, default is False
        ``additional_output`` is only used for Mask R-CNN to get internal outputs.
    Attributes
    ----------
    classes : iterable of str
        Names of categories, its length is ``num_class``.
    num_class : int
        Number of positive categories.
    short : int
        Input image short side size.
    max_size : int
        Maximum size of input image long side.
    train_patterns : str
        Matching pattern for trainable parameters.
    nms_thresh : float
        Non-maximum suppression threshold. You can speficy < 0 or > 1 to disable NMS.
    nms_topk : int
        Apply NMS to top k detection results, use -1 to disable so that every Detection
         result is used in NMS.
    post_nms : int
        Only return top `post_nms` detection results, the rest is discarded. The number is
        based on COCO dataset which has maximum 100 objects per image. You can adjust this
        number if expecting more objects. You can use -1 to return all detections.
    target_generator : gluon.Block
        Generate training targets with boxes, samples, matches, gt_label and gt_box.
    """
    def __init__(self, features, top_features, classes,
                 short=600, max_size=1000, train_patterns=None,
                 nms_thresh=0.3, nms_topk=400, post_nms=100,
                 roi_mode='align', roi_size=(14, 14), stride=16, clip=None,
                 rpn_channel=1024, base_size=16, scales=(8, 16, 32),
                 ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
                 rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
                 rpn_test_pre_nms=6000, rpn_test_post_nms=300, rpn_min_size=16,
                 num_sample=128, pos_iou_thresh=0.5, pos_ratio=0.25,
                 additional_output=False, **kwargs):
        super(rfcn, self).__init__(
            features=features, top_features=top_features, classes=classes,
            short=short, max_size=max_size, train_patterns=train_patterns,
            nms_thresh=nms_thresh, nms_topk=nms_topk, post_nms=post_nms,
            roi_mode=roi_mode, roi_size=roi_size, stride=stride, clip=clip, **kwargs)
        self._max_batch = 1  # currently only support batch size = 1
        self._num_sample = num_sample
        self._rpn_test_post_nms = rpn_test_post_nms
        self._target_generator = {RCNNTargetGenerator(self.num_class)}
        self._additional_output = additional_output
        with self.name_scope():
            self.rpn = RPN(
                channels=rpn_channel, stride=stride, base_size=base_size,
                scales=scales, ratios=ratios, alloc_size=alloc_size,
                clip=clip, nms_thresh=rpn_nms_thresh, train_pre_nms=rpn_train_pre_nms,
                train_post_nms=rpn_train_post_nms, test_pre_nms=rpn_test_pre_nms,
                test_post_nms=rpn_test_post_nms, min_size=rpn_min_size)
            self.sampler = RCNNTargetSampler(
                num_image=self._max_batch, num_proposal=rpn_train_post_nms,
                num_sample=num_sample, pos_iou_thresh=pos_iou_thresh, pos_ratio=pos_ratio)

    @property
    def target_generator(self):
        """Returns stored target generator
        Returns
        -------
        mxnet.gluon.HybridBlock
            The RCNN target generator
        """
        return list(self._target_generator)[0]

    def reset_class(self, classes):
        super(FasterRCNN, self).reset_class(classes)
        self._target_generator = {RCNNTargetGenerator(self.num_class)}

    def add_batchid(self, F, bbox):
        num_roi = self._num_sample if autograd.is_training() else self._rpn_test_post_nms
        with autograd.pause():
            roi_batchid = F.arange(0, self._max_batch, repeat=num_roi)
            # remove batch dim because ROIPooling require 2d input
            roi = F.concat(*[roi_batchid.reshape((-1, 1)), bbox.reshape((-1, 4))], dim=-1)
            roi = F.stop_gradient(roi)
            return roi

    def ROIExtraction(self, F, feature, bbox, output_dim):

        roi = self.add_batchid(F, bbox)

        # ROI features
        if self._roi_mode == 'pspool':
            pooled_feat = F.contrib.PSROIPooling(data=feature, rois=roi, spatial_scale=1. / self._stride, \
                output_dim=output_dim,pooled_size=self._roi_size[0],group_size=self._roi_size[0]) #,pooled_size=self._roi_size[0]
        else:
            raise ValueError("Invalid roi mode: {}".format(self._roi_mode))
        return pooled_feat

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x, gt_box=None):
        """Forward Faster-RCNN network.
        The behavior during traing and inference is different.
        Parameters
        ----------
        x : mxnet.nd.NDArray or mxnet.symbol
            The network input tensor.
        gt_box : type, only required during training
            The ground-truth bbox tensor with shape (1, N, 4).
        Returns
        -------
        (ids, scores, bboxes)
            During inference, returns final class id, confidence scores, bounding
            boxes.
        """
        def _split(x, axis, num_outputs, squeeze_axis):
            x = F.split(x, axis=axis, num_outputs=num_outputs, squeeze_axis=squeeze_axis)
            if isinstance(x, list):
                return x
            else:
                return [x]

        feat = self.features(x)
        # RPN proposals
        if autograd.is_training():
            rpn_score, rpn_box, raw_rpn_score, raw_rpn_box, anchors = \
                self.rpn(feat, F.zeros_like(x))
            rpn_box, samples, matches = self.sampler(rpn_box, rpn_score, gt_box)
        else:
            _, rpn_box = self.rpn(feat, F.zeros_like(x))

        res5c = self.top_features(feat)
        conv_new_1 = self.conv_new_1(res5c)
        rfcn_cls_feat = self.rfcn_cls(conv_new_1)
        rfcn_bbox_feat = self.rfcn_bbox(conv_new_1)
        # ROI features (ROI ps)
        # _,infer_shape,_ = rfcn_cls_feat.infer_shape(data0=(1,3,600,800) )
        # print("rfcn_cls_feat shape:{}".format(infer_shape))
        psroipooled_cls_rois = self.ROIExtraction(F=F, feature=rfcn_cls_feat, bbox=rpn_box, output_dim= self.num_class+1 )
        psroipooled_loc_rois = self.ROIExtraction(F=F, feature=rfcn_bbox_feat, bbox=rpn_box, output_dim=4)
        # _,infer_shape,_ = psroipooled_cls_rois.infer_shape(data0=(1,3,600,800),data1=(1,1,4) )
        # print("psroipooled_cls_rois shape:{}".format(infer_shape))
        cls_pred = F.Pooling(data=psroipooled_cls_rois, kernel=(7, 7), stride=(7, 7), pool_type='avg')
        box_pred = F.Pooling(data=psroipooled_loc_rois, kernel=(7, 7), stride=(7, 7), pool_type='avg')
        # _,infer_shape,_ = cls_pred.infer_shape(data0=(1,3,600,800),data1=(1,1,4) )
        # print("cls_pred shape:{}".format(infer_shape))
        num_roi = self._num_sample if autograd.is_training() else self._rpn_test_post_nms
        cls_pred = F.squeeze(cls_pred, axis=(2,3))
        box_pred = F.squeeze(box_pred, axis=(2,3))
        # cls_pred (B * N, C) -> (B, N, C)
        cls_pred = cls_pred.reshape((self._max_batch, num_roi, self.num_class + 1))
        # box_pred (B * N, C * 4) -> (B, N, C, 4)
        box_pred = box_pred.reshape((self._max_batch, num_roi, 1, 4))

        # no need to convert bounding boxes in training, just return
        if autograd.is_training():
            if self._additional_output:
                return (cls_pred, box_pred, rpn_box, samples, matches,
                        raw_rpn_score, raw_rpn_box, anchors, top_feat)
            return (cls_pred, box_pred, rpn_box, samples, matches,
                    raw_rpn_score, raw_rpn_box, anchors)

        # cls_ids (B, N, C), scores (B, N, C)
        cls_ids, scores = self.cls_decoder(F.softmax(cls_pred, axis=-1))
        # cls_ids, scores (B, N, C) -> (B, C, N) -> (B, C, N, 1)
        cls_ids = cls_ids.transpose((0, 2, 1)).reshape((0, 0, 0, 1))
        scores = scores.transpose((0, 2, 1)).reshape((0, 0, 0, 1))
        # box_pred (B, N, C, 4) -> (B, C, N, 4)
        box_pred = box_pred.transpose((0, 2, 1, 3))

        # rpn_boxes (B, N, 4) -> B * (1, N, 4)
        rpn_boxes = _split(rpn_box, axis=0, num_outputs=self._max_batch, squeeze_axis=False)
        # cls_ids, scores (B, C, N, 1) -> B * (C, N, 1)
        cls_ids = _split(cls_ids, axis=0, num_outputs=self._max_batch, squeeze_axis=True)
        scores = _split(scores, axis=0, num_outputs=self._max_batch, squeeze_axis=True)
        # box_preds (B, C, N, 4) -> B * (C, N, 4)
        box_preds = _split(box_pred, axis=0, num_outputs=self._max_batch, squeeze_axis=True)

        # per batch predict, nms, each class has topk outputs
        results = []
        for rpn_box, cls_id, score, box_pred in zip(rpn_boxes, cls_ids, scores, box_preds):
            # box_pred (C, N, 4) rpn_box (1, N, 4) -> bbox (C, N, 4)
            bbox = self.box_decoder(box_pred, self.box_to_center(rpn_box))
            bbox = F.repeat(bbox, repeats=self.num_class, axis=0)
            # res (C, N, 6)
            res = F.concat(*[cls_id, score, bbox], dim=-1)
            # res (C, self.nms_topk, 6)
            res = F.contrib.box_nms(
                res, overlap_thresh=self.nms_thresh, topk=self.nms_topk, valid_thresh=0.0001,
                id_index=0, score_index=1, coord_start=2, force_suppress=True)
            # res (C * self.nms_topk, 6)
            res = res.reshape((-3, 0))
            results.append(res)

        # result B * (C * topk, 6) -> (B, C * topk, 6)
        result = F.stack(*results, axis=0)
        ids = F.slice_axis(result, axis=-1, begin=0, end=1)
        scores = F.slice_axis(result, axis=-1, begin=1, end=2)
        bboxes = F.slice_axis(result, axis=-1, begin=2, end=6)
        if self._additional_output:
            return ids, scores, bboxes, feat
        return ids, scores, bboxes

def get_rfcn(name, dataset, pretrained=False, ctx=mx.cpu(),
                    root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    r"""Utility function to return faster rcnn networks.
    Parameters
    ----------
    name : str
        Model name.
    dataset : str
        The name of dataset.
    pretrained : bool, optional, default is False
        Load pretrained weights.
    ctx : mxnet.Context
        Context such as mx.cpu(), mx.gpu(0).
    root : str
        Model weights storing path.
    Returns
    -------
    mxnet.gluon.HybridBlock
        The Faster-RCNN network.
    """
    net = rfcn(**kwargs)
    if pretrained:
        from ..model_store import get_model_file
        full_name = '_'.join(('rfcn', name, dataset))
        net.load_parameters(get_model_file(full_name, root=root), ctx=ctx)
    return net

def rfcn_resnet50_v1b_voc(pretrained=False, pretrained_base=True, **kwargs):
    r"""Faster RCNN model from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"
    Parameters
    ----------
    pretrained : bool, optional, default is False
        Load pretrained weights.
    pretrained_base : bool, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `Ture`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    Examples
    --------
    >>> model = get_faster_rcnn_resnet50_v1b_voc(pretrained=True)
    >>> print(model)
    """
    from ..resnetv1b import resnet50_v1b
    from ...data import VOCDetection
    classes = VOCDetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet50_v1b(pretrained=pretrained_base, dilated=False, use_global_stats=True)
    features = nn.HybridSequential()
    top_features = nn.HybridSequential()
    for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
        features.add(getattr(base_network, layer))
    for layer in ['layer4']:
        top_features.add(getattr(base_network, layer))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
    return get_rfcn(
        name='resnet50_v1b', dataset='voc', pretrained=pretrained,
        features=features, top_features=top_features, classes=classes,
        short=600, max_size=1000, train_patterns=train_patterns,
        nms_thresh=0.3, nms_topk=400, post_nms=100,
        roi_mode='align', roi_size=(14, 14), stride=16, clip=None,
        rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16, 32),
        ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
        rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=300, rpn_min_size=16,
        num_sample=128, pos_iou_thresh=0.5, pos_ratio=0.25,
        **kwargs)

def rfcn_resnet50_v2a_voc(pretrained=False, pretrained_base=True, **kwargs):
    r"""Faster RCNN model from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"

    Parameters
    ----------
    pretrained : bool, optional, default is False
        Load pretrained weights.
    pretrained_base : bool, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `Ture`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_cascade_rcnn_resnet50_v2a_voc(pretrained=True)
    >>> print(model)
    """
    from .resnet50_v2a import resnet50_v2a
    from ...data import VOCDetection
    classes = VOCDetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet50_v2a(pretrained=pretrained_base)
    features = nn.HybridSequential()
    top_features = nn.HybridSequential()
    for layer in ['rescale'] + ['layer' + str(i) for i in range(4)]:
        features.add(getattr(base_network, layer))
    for layer in ['layer4']:
        top_features.add(getattr(base_network, layer))
    train_patterns = '|'.join(['.*rfcn0_conv','.*dense', '.*rpn', '.*stage(2|3|4)_conv'])  #
    # print("~~~~~")
    # print(features.collect_params())
    # print(top_features.collect_params())
    return get_rfcn(
        name='resnet50_v2a', dataset='voc', pretrained=pretrained,
        features=features, top_features=top_features, 
        classes=classes,
        short=600, max_size=1000, train_patterns=train_patterns,
        nms_thresh=0.3, nms_topk=400, post_nms=100,
        roi_mode='pspool', roi_size=(7, 7), stride=16, clip=None,
        rpn_channel=512, base_size=16, scales=( 8, 16, 32),
        ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
        rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=300, rpn_min_size=5,
        num_sample=128, pos_iou_thresh=0.5, pos_ratio=0.25,
        **kwargs)