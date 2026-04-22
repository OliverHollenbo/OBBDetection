from mmdet.models.builder import DETECTORS
from .obb_two_stage import OBBTwoStageDetector

@DETECTORS.register_module()
class OrientedRCNN(OBBTwoStageDetector):
    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 teacher_ckpt=None,
                 distill_alpha=0.0005):
        super(OrientedRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            teacher_ckpt=teacher_ckpt,
            distill_alpha=distill_alpha)
