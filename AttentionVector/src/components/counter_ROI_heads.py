import torch
from typing import Dict, List
from detectron2.layers import ShapeSpec
from detectron2.structures import Instances
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads

from counter_output_layers import CounterOutputLayer


@ROI_HEADS_REGISTRY.register()
# RoI head with counter part of RAVNet
class CounterROIHeads(StandardROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape,
                         box_predictor=CounterOutputLayer(cfg, self.predictor_input_shape(cfg, input_shape)))

    # calculate input shape of box_predictor
    # params - input_shape: input shape for RoIHeads
    def predictor_input_shape(self, cfg, input_shape):
        # build input_shape of box_predictor
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION

        # box_head output size
        head_output = (in_channels[0], pooler_resolution, pooler_resolution)

        num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM

        dict_head = {
            "conv_dims": [conv_dim] * num_conv,
            "fc_dims": [fc_dim] * num_fc,
        }

        for k, conv_dim in enumerate(dict_head['conv_dims']):
            head_output = (conv_dim, head_output[1], head_output[2])

        for k, fc_dim in enumerate(dict_head['fc_dims']):
            head_output = fc_dim

        o = head_output
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return ShapeSpec(channels=o[0], height=o[1], width=o[2])
