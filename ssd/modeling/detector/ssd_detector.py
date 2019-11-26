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
        if self.cfg.EXPORT_SSD:
            import torch
            class SSD_before_postprocess(torch.nn.Module):
                def __init__(self, backbone, encoder):
                    super(SSD_before_postprocess, self).__init__()
                    self.backbone = backbone
                    self.encoder = encoder

                def forward(self, image):
                    features = self.backbone(image)
                    cls_logits, bbox_pred = self.encoder(features)   
                    return cls_logits, bbox_pred
            ssd_top_half = SSD_before_postprocess(self.backbone, self.box_predictor)
            ssd_top_half.eval()
            torch.onnx.export(ssd_top_half, images, "resnet50_ssd300_coco_trainval35k.onnx", verbose=True)
            # torch.onnx.export(ssd_top_half, images, "mobilenet_v2_ssd320_coco_trainval35k.onnx", verbose=True)
            if self.cfg.EXPORT_PRE_TENSORS:
                scores, bboxes = ssd_top_half(images)
                import numpy as np
                np.save("_model.input-0.npy", images)
                np.save("_model.output-0.npy", scores)
                np.save("_model.output-1.npy", bboxes)
                raise ValueError("aahahhhhh")

        features = self.backbone(images)
        cls_logits, bbox_pred = self.box_predictor(features)
        detections, detector_losses = self.box_head(cls_logits, bbox_pred, targets)
        if self.training:
            return detector_losses

        # Export end-to-end inputs and outputs
        if self.cfg.EXPORT_POST_TENSORS:
            import numpy as np
            # Only export the boxes that have scores over the threshold
            valid_indices = np.where(detections[0]['scores'] > self.cfg.TEST.NMS_THRESHOLD)
            boxes_export = detections[0]['boxes'][valid_indices]
            labels_export = detections[0]['labels'][valid_indices]
            scores_export = detections[0]['scores'][valid_indices]
            np.save("_model.input-0.npy", images)
            np.save("_model.output-0.npy", boxes_export)
            np.save("_model.output-1.npy", labels_export)
            np.save("_model.output-2.npy", scores_export)
            raise ValueError("aahahhhhh")

        return detections
