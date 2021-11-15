import random, os, cv2
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
import sample_register_custom_dataset as add_balloon
from detectron2.utils.visualizer import Visualizer


def run(cfg, balloon_metadata, visualize):
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    if visualize:
        dataset_dicts = add_balloon.get_balloon_dicts("../prelim/data/balloon/val")
        for d in random.sample(dataset_dicts, 3):
            im = cv2.imread(d["file_name"])
            outputs = predictor(im)
            # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            v = Visualizer(im[:, :, ::-1],
                           metadata=balloon_metadata,
                           scale=0.5,
                           # instance_mode=ColorMode.IMAGE_BW
                           # remove the colors of unsegmented pixels. This option is only available for segmentation models
            )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imshow('image', out.get_image()[:, :, ::-1])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return predictor
