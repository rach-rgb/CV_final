import os, cv2, torch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from components.cross_ROI_heads import CrossROIHeads


def run():
    cfg = get_cfg()
    cfg.merge_from_file('output.yaml')
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    im = cv2.imread("./input.png")

    cv2.imshow('image', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    cv2.imshow('result', out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    run()
