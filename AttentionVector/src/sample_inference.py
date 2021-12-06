import os, cv2, torch
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from components.cross_ROI_heads import CrossROIHeads
from components.counter_ROI_heads import CounterROIHeads


def run():
    cfg = get_cfg()
    cfg.merge_from_file('output.yaml')
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    cfg_cmp = get_cfg()
    cfg_cmp.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg_cmp.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

    predictor = DefaultPredictor(cfg)
    predictor_cmp = DefaultPredictor(cfg_cmp)

    for i in range(1, 8):
        im = cv2.imread("./sample/input" + str(i) + ".png")

        outputs = predictor(im)
        outputs_cmp = predictor_cmp(im)

        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        out_img = out.get_image()[:, :, ::-1]

        v_cmp = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg_cmp.DATASETS.TRAIN[0]), scale=1.2)
        out_cmp = v_cmp.draw_instance_predictions(outputs_cmp["instances"].to("cpu"))
        out_cmp_img = out_cmp.get_image()[:, :, ::-1]

        cv2.imshow('result', np.concatenate((out_img, out_cmp_img), axis=1))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    run()
