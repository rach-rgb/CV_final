from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from cross_output_layers import CrossOutputLayer
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.layers import ShapeSpec


@ROI_HEADS_REGISTRY.register()
class CrossROIHeads(StandardROIHeads):
    def __init__(self, cfg, input_shape):
        # build input_shape of box_predictor
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_head = build_box_head(cfg,
                                  ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution))

        super().__init__(cfg, input_shape, box_predictor=CrossOutputLayer(cfg, box_head.output_shape))



