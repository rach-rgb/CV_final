import os
from detectron2 import model_zoo
from detectron2.config import get_cfg

from components.cross_ROI_heads import CrossROIHeads


def run(use_cfg=True, save_cfg=True, use_weight=False):
    cfg = get_cfg()

    # use custom cfg
    if use_cfg:
        cfg.merge_from_file('output.yaml')
    else:  # use baseline cfg
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NAME = 'CrossROIHeads'
        cfg.SOLVER.IMS_PER_BATCH = 4

    # use custom weight
    if use_weight:
        print("NOT IMPL")
    else:   # use baseline weight
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

    # save cfg
    if save_cfg:
        with open('output.yaml', 'w') as f:
            f.write(cfg.dump())

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg


if __name__ == '__main__':
    run()
