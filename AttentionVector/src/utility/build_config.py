from detectron2 import model_zoo
from detectron2.config import get_cfg

from components.rav_ROI_heads import RAVROIHeads
from components.counter_ROI_heads import CounterROIHeads


# build config and dump yaml file as output.yaml
if __name__ == "__main__":
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NAME = 'RAVROIHeads'
    cfg.SOLVER.IMS_PER_BATCH = 2
    # cfg.SOLVER.BASE_LR = base learning rates
    cfg.SOLVER.MAX_ITER = 15000
    cfg.SOLVER.CHECKPOINT_PERIOD = 5000
    cfg.TEST.EVAL_PERIOD = 1000
    cfg.SOLVER.STEPS = (3000, )
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

    # dump result as output.yaml
    with open('../output.yaml', 'w') as f:
        f.write(cfg.dump())
