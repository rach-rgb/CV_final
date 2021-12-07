from detectron2 import model_zoo
from detectron2.config import get_cfg

from components.cross_ROI_heads import CrossROIHeads
from components.counter_ROI_heads import CounterROIHeads

# build config and dump yaml file as output.yaml
if __name__ == "__main__":
    cfg = get_cfg()

    total = 41000
    batch = 2
    epoch = 3
    total_iter = int(total/batch)

    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NAME = 'CounterROIHeads'
    cfg.SOLVER.IMS_PER_BATCH = 2
    # cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR
    cfg.SOLVER.MAX_ITER = int(total / batch * epoch)
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    cfg.TEST.EVAL_PERIOD = 1000
    cfg.SOLVER.STEPS = (3000, )
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

    with open('../output.yaml', 'w') as f:
        f.write(cfg.dump())
