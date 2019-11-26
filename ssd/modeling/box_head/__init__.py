from ssd.modeling import registry
from .box_head import SSDBoxHead
from .box_predictor import BoxPredictor

__all__ = ['build_box_predictor', 'build_box_head', 'SSDBoxHead']

def build_box_predictor(cfg):
    return registry.BOX_PREDICTORS[cfg.MODEL.BOX_HEAD.PREDICTOR](cfg)

def build_box_head(cfg):
    return registry.BOX_HEADS[cfg.MODEL.BOX_HEAD.NAME](cfg)
