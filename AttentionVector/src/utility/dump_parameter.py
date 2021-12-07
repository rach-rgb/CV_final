import cv2
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from components.cross_ROI_heads import CrossROIHeads


def run():
    cfg = get_cfg()
    cfg.merge_from_file('../output.yaml')
    cfg.MODEL.WEIGHTS = "../output/model_final.pth"

    predictor = DefaultPredictor(cfg)
    predictor.model.roi_heads.box_predictor.cross_net.track(True)

    for i in range(1, 8):
        im = cv2.imread("../sample/input" + str(i) + ".png")

        outputs = predictor(im)

        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow('image', out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        break


if __name__ == '__main__':
    run()
