from torch import nn

from ssd.modeling.backbone import build_backbone
from ssd.modeling.box_head import build_box_head
from ssd.modeling.box_head import build_box_predictor


class SSDDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.box_predictor = build_box_predictor(cfg)
        self.box_head = build_box_head(cfg)

    def forward(self, images, targets=None):
        features = self.backbone(images)
        cls_logits, bbox_pred = self.box_predictor(features)
        detections, detector_losses = self.box_head(cls_logits, bbox_pred, targets)
        if self.training:
            return detector_losses
        return detections
