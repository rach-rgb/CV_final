import torch
import torch.nn as nn
from typing import Dict, Union
from detectron2.layers import ShapeSpec
from detectron2.config import configurable
from detectron2.modeling import FastRCNNOutputLayers


# output layer with counterpart of RAVNet
class CounterOutputLayer(FastRCNNOutputLayers):
    @configurable
    def __init__(self, input_shape: ShapeSpec, *, box2box_transform, num_classes: int, test_score_thresh: float = 0.0,
                 test_nms_thresh: float = 0.5, test_topk_per_image: int = 100, cls_agnostic_bbox_reg: bool = False,
                 smooth_l1_beta: float = 0.0, box_reg_loss_type: str = "smooth_l1",
                 loss_weight: Union[float, Dict[str, float]] = 1.0):

        super().__init__(input_shape, box2box_transform=box2box_transform, num_classes=num_classes,
                         test_score_thresh=test_score_thresh, test_nms_thresh=test_nms_thresh,
                         test_topk_per_image=test_topk_per_image, cls_agnostic_bbox_reg=cls_agnostic_bbox_reg,
                         smooth_l1_beta=smooth_l1_beta, box_reg_loss_type=box_reg_loss_type, loss_weight=loss_weight)

        # two FC layers as counter part of value and unpack FC layers
        self.fc1 = nn.Linear(num_classes+1, 64)
        self.fc2 = nn.Linear(64, num_classes+1)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        y = self.fc1(scores)
        y = self.fc2(y)
        scores = y + scores

        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas
