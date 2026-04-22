import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


def focal_loss(student_feat, teacher_feat, gt_bboxes, img_metas):
    """Focal distillation loss — focuses on object regions.

    Args:
        student_feat (Tensor): Student FPN feature (B, C, H, W)
        teacher_feat (Tensor): Teacher FPN feature (B, C, H, W)
        gt_bboxes (list[Tensor]): Ground truth boxes for each image
        img_metas (list[dict]): Image metadata (contains img shape)

    Returns:
        Tensor: Scalar focal distillation loss
    """
    B, C, H, W = student_feat.shape
    device = student_feat.device

    # Build spatial attention mask from ground truth boxes
    mask = torch.ones(B, 1, H, W, device=device) * 0.5

    for i, bboxes in enumerate(gt_bboxes):
        if bboxes is None or len(bboxes) == 0:
            continue

        img_h, img_w = img_metas[i]['img_shape'][:2]
        scale_x = W / img_w
        scale_y = H / img_h

        for bbox in bboxes:
            # bbox format: (x1, y1, x2, y2) in image coords
            x1 = max(0, int(bbox[0].item() * scale_x))
            y1 = max(0, int(bbox[1].item() * scale_y))
            x2 = min(W, int(bbox[2].item() * scale_x) + 1)
            y2 = min(H, int(bbox[3].item() * scale_y) + 1)

            if x2 > x1 and y2 > y1:
                mask[i, 0, y1:y2, x1:x2] = 1.0

    # Normalise mask
    mask = mask / (mask.sum() + 1e-6)

    # Weighted MSE loss — penalise more near objects
    diff = (student_feat - teacher_feat.detach()) ** 2
    loss = (mask * diff.mean(dim=1, keepdim=True)).sum()

    return loss


def global_loss(student_feat, teacher_feat):
    """Global distillation loss — matches channel correlations (Gram matrix).

    Args:
        student_feat (Tensor): Student FPN feature (B, C, H, W)
        teacher_feat (Tensor): Teacher FPN feature (B, C, H, W)

    Returns:
        Tensor: Scalar global distillation loss
    """
    B, C, H, W = student_feat.shape

    # Reshape to (B, C, H*W)
    s = student_feat.reshape(B, C, -1)
    t = teacher_feat.detach().reshape(B, C, -1)

    # Compute Gram matrices (B, C, C)
    gram_s = torch.bmm(s, s.transpose(1, 2))
    gram_t = torch.bmm(t, t.transpose(1, 2))

    # Normalise by number of elements
    norm = C * H * W
    gram_s = gram_s / norm
    gram_t = gram_t / norm

    loss = F.mse_loss(gram_s, gram_t)

    return loss


@LOSSES.register_module()
class FGDLoss(nn.Module):
    """Focal and Global Knowledge Distillation Loss for object detection.

    Based on: Yang et al., "Focal and Global Knowledge Distillation
    for Detectors", CVPR 2022.

    Args:
        alpha_focal (float): Weight for focal distillation loss.
        alpha_global (float): Weight for global distillation loss.
        loss_weight (float): Overall weight of the FGD loss.
    """

    def __init__(self,
                 alpha_focal=0.0005,
                 alpha_global=0.0005,
                 loss_weight=1.0):
        super(FGDLoss, self).__init__()
        self.alpha_focal = alpha_focal
        self.alpha_global = alpha_global
        self.loss_weight = loss_weight

    def forward(self,
                student_feats,
                teacher_feats,
                gt_bboxes,
                img_metas):
        """Forward function.

        Args:
            student_feats (list[Tensor]): Student FPN features [P2,P3,P4,P5]
            teacher_feats (list[Tensor]): Teacher FPN features [P2,P3,P4,P5]
            gt_bboxes (list[Tensor]): Ground truth boxes per image
            img_metas (list[dict]): Image metadata per image

        Returns:
            dict: losses containing loss_focal and loss_global
        """
        assert len(student_feats) == len(teacher_feats)

        loss_focal_total = 0.0
        loss_global_total = 0.0

        for s_feat, t_feat in zip(student_feats, teacher_feats):
            loss_focal_total += focal_loss(
                s_feat, t_feat, gt_bboxes, img_metas)
            loss_global_total += global_loss(s_feat, t_feat)

        loss_focal_total = self.alpha_focal * loss_focal_total
        loss_global_total = self.alpha_global * loss_global_total

        return dict(
            loss_focal=self.loss_weight * loss_focal_total,
            loss_global=self.loss_weight * loss_global_total)
